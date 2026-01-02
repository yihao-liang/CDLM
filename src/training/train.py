import os
import logging
import argparse
import torch
import torch.distributed as dist
import json
import math
from transformers import AutoTokenizer, EarlyStoppingCallback
from transformers.training_args import TrainingArguments
from transformers.trainer_callback import TrainerCallback
from datasets import load_from_disk

from src.model import LLaDAModelLM, StudentModel
from src.data import ConsistencyDistillationCollator, OpenCodeInstructCollator
from src.training.consistency_loss import ConsistencyTrainer

# Get the specific logger that produces this warning
c10d_logger = logging.getLogger("torch.distributed.distributed_c10d")
# Elevate its log level to ERROR so it only shows ERROR and higher level messages
c10d_logger.setLevel(logging.ERROR)

# ========== Logging Configuration ==========
def is_main_process():
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    return local_rank == 0

def setup_logging(output_dir):
    # Clear previous handlers to avoid duplicate configuration
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create an empty handlers list
    handlers = []
    
    # Only configure handlers for the main process
    if is_main_process():
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        log_file = os.path.join(output_dir, "logs", "train.log")
        
        # File logging: record all INFO and above
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
        
        # Console logging: only show WARNING and above
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)
    else:
        # Add a NullHandler for non-main processes to completely suppress log output
        null_handler = logging.NullHandler()
        handlers.append(null_handler)

    # Configure root logger
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)

logger = logging.getLogger(__name__)

# ========== Logging Callback ==========
class LoggingCallback(TrainerCallback):
    """A custom callback for writing training logs (like loss) to files."""
    def __init__(self):
        super().__init__()
        self.last_lr = None
        self.last_logged_step = -1
        self.step_log_count = {}  # Record the log count for each step
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Only execute on main process
        if not is_main_process() or logs is None:
            return
            
        current_step = state.global_step
        
        # Avoid duplicate logging for the same step
        # Only log entries containing important training metrics
        important_keys = ['loss', 'train_loss', 'eval_loss', 'learning_rate', 'grad_norm']
        has_important_metrics = any(k in logs for k in important_keys)
        
        if not has_important_metrics:
            return  # Skip unimportant logs
            
        # Check for duplicate logs from the same step
        if current_step in self.step_log_count:
            self.step_log_count[current_step] += 1
            # If training metrics have already been logged for this step, skip duplicate recording
            if self.step_log_count[current_step] > 1 and 'eval_loss' not in logs:
                return
        else:
            self.step_log_count[current_step] = 1
        
        # Get current learning rate
        trainer = kwargs.get("trainer")
        current_lr = None
        if trainer and hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
            current_lr = trainer.optimizer.param_groups[0]['lr']
        
        # Filter out unimportant logs to reduce output
        filtered_logs = {}
        for k, v in logs.items():
            # Only record important training metrics
            if k in ['loss', 'train_loss', 'eval_loss', 'learning_rate', 'grad_norm', 'epoch', 
                    'train_runtime', 'train_samples_per_second', 'train_steps_per_second']:
                filtered_logs[k] = f"{v:.6f}" if isinstance(v, float) else v
        
        # Add actual learning rate to logs
        if current_lr is not None:
            filtered_logs['actual_lr'] = f"{current_lr:.2e}"
            
            # Detect learning rate changes (only check on new steps)
            if (current_step != self.last_logged_step and 
                self.last_lr is not None and abs(current_lr - self.last_lr) > 1e-10):
                lr_change_ratio = current_lr / self.last_lr if self.last_lr > 0 else 0
                logger.info(f"🔄 Learning rate change detected: {self.last_lr:.2e} → {current_lr:.2e} (ratio: {lr_change_ratio:.3f})")
            
            self.last_lr = current_lr
        
        if filtered_logs:
            # Distinguish between training and evaluation logs
            log_type = "📊 EVAL" if 'eval_loss' in logs else "🏃 TRAIN"
            message = f"{log_type} Step {current_step} | {filtered_logs}"
            logger.info(message)
            
        self.last_logged_step = current_step
        
        # Clean up old step counts (keep the most recent 100 steps)
        if len(self.step_log_count) > 100:
            old_steps = [s for s in self.step_log_count.keys() if s < current_step - 50]
            for old_step in old_steps:
                del self.step_log_count[old_step]

def main():
    parser = argparse.ArgumentParser(description="DSCD (Discrete-Space Consistency Distillation) Training Script")
    parser.add_argument("--teacher_model_path", type=str, required=True, help="Path to the pretrained teacher model.")
    parser.add_argument("--data_dir", type=str, default="/data/gsm8k/", help="Directory containing the processed train/validation datasets.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and final model.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Batch size for evaluation.")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Number of warmup steps.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps.")
    parser.add_argument("--eval_steps", type=int, default=50, help="Evaluate every N steps.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps.")
    parser.add_argument("--total_diff_steps", type=int, default=1000, help="Total number of diffusion steps for the collator.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps.")
    parser.add_argument("--lambda_val", type=float, default=0.5, help="Weight for the consistency loss term (static lambda).")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Early stopping patience.")

    # Dynamic lambda parameters
    parser.add_argument("--use_dynamic_lambda", action="store_true", help="Enable dynamic lambda scheduling.")
    parser.add_argument("--initial_lambda", type=float, default=0.9, help="Initial lambda value at the start of training.")
    parser.add_argument("--final_lambda", type=float, default=0.5, help="Final lambda value at the end of training.")
    parser.add_argument("--lambda_warmup_ratio", type=float, default=0.1, help="Warmup ratio for lambda scheduling (0-1).")
    
    # Masking strategy parameters
    parser.add_argument("--use_nested_masking", action="store_true", help="Enable nested masking (teacher masks ⊆ student masks).")

    # Dataset selection
    parser.add_argument("--use_opencode_dataset", action="store_true", help="Use OpenCodeInstruct dataset for code generation training.")

    # Soft distillation temperature parameter
    parser.add_argument("--temperature", type=float, default=3.0, help="Temperature for soft distillation softmax scaling.")

    args = parser.parse_args()

    # Logging configuration
    setup_logging(args.output_dir)

    # ========== Log training parameters and DeepSpeed configuration ==========
    logger.info("Shell/Command-line arguments: %s", json.dumps(vars(args), indent=2, ensure_ascii=False))
    deepspeed_config_path = os.path.join("configs", "deepspeed_config.json")
    if os.path.exists(deepspeed_config_path):
        with open(deepspeed_config_path, 'r', encoding='utf-8') as f:
            deepspeed_config = json.load(f)
        logger.info("DeepSpeed config: %s", json.dumps(deepspeed_config, indent=2, ensure_ascii=False))
    else:
        logger.warning("DeepSpeed config file not found: %s", deepspeed_config_path)

    # ========== Args - Special configuration for DeepSpeed ==========
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        bf16=True,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        dataloader_num_workers=0,  # DeepSpeed recommended
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_total_limit=1,  # Only keep the latest 1 checkpoint
    )

    # ========== Tokenizer loading and special token handling ==========
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Remove code for adding extra mask tokens, use original model's <|mdm_mask|> token
    # Original LLaDA-1.5 model already contains dedicated mask token, no need to add extra tokens
    
    logger.info(f"Tokenizer info: vocab_size={tokenizer.vocab_size}, mask_token='{tokenizer.mask_token}' (ID: {tokenizer.mask_token_id})")

    # ========== Model loading and embedding resize checks ==========
    logger.info("Loading models and tokenizer...")
    teacher_model = LLaDAModelLM.from_pretrained(args.teacher_model_path, torch_dtype=torch.bfloat16)
    
    # Safety check: ensure teacher_model is loaded correctly
    if teacher_model is None:
        raise ValueError("LLaDAModelLM.from_pretrained() returned None, please check teacher_model_path")
    
    if not hasattr(teacher_model, 'config') or teacher_model.config is None:
        raise ValueError("teacher_model.config is None, teacher model loading may have failed")
    
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    student_model = StudentModel.from_pretrained(args.teacher_model_path, torch_dtype=torch.bfloat16)
    
    # Safety check: ensure student_model is loaded correctly
    if student_model is None:
        raise ValueError("StudentModel.from_pretrained() returned None, please check StudentModel implementation")
    
    if not hasattr(student_model, 'config') or student_model.config is None:
        raise ValueError("student_model.config is None, student model loading may have failed")

    # Check relationship between vocab_size and embedding_size
    model_vocab_size = getattr(student_model.config, 'vocab_size', 'N/A')
    model_embedding_size = getattr(student_model.config, 'embedding_size', 'N/A')
    current_tokenizer_size = len(tokenizer)
    
    if is_main_process():
        logger.info(f"Vocabulary size check:")
        logger.info(f"  - Tokenizer vocabulary size: {current_tokenizer_size}")
        logger.info(f"  - Model config vocab_size: {model_vocab_size}")
        logger.info(f"  - Model config embedding_size: {model_embedding_size}")
    
    # Actual embedding layer size is embedding_size or vocab_size
    actual_model_embedding_size = model_embedding_size if model_embedding_size is not None else model_vocab_size
    
    # 🔧 Modified strategy: keep teacher and student model structures completely consistent, no resize
    if actual_model_embedding_size != current_tokenizer_size:
        if is_main_process():
            logger.info(f"Detected embedding size mismatch:")
            logger.info(f"  - Model embedding layer size: {actual_model_embedding_size}")
            logger.info(f"  - Tokenizer vocabulary size: {current_tokenizer_size}")
            logger.info("🎯 New strategy: keep original model structure unchanged, avoid resize operations")
            logger.info("  - This ensures teacher and student models have completely consistent structures")
            logger.info("  - Avoids vocab size mixing issues after training")
            logger.info("  - Use dedicated StudentModelLoader for dimension matching during inference")
        
        # Validate current model structure
        if is_main_process():
            student_input_size = student_model.get_input_embeddings().weight.shape[0]
            student_output_size = student_model.get_output_embeddings().weight.shape[0]
            teacher_input_size = teacher_model.get_input_embeddings().weight.shape[0]
            teacher_output_size = teacher_model.get_output_embeddings().weight.shape[0]
            
            logger.info(f"Maintaining original model structure:")
            logger.info(f"  Student model - input embedding: {student_input_size}, output embedding: {student_output_size}")
            logger.info(f"  Teacher model - input embedding: {teacher_input_size}, output embedding: {teacher_output_size}")
            
            # Ensure teacher and student models have consistent structure
            if (student_input_size == teacher_input_size and 
                student_output_size == teacher_output_size):
                logger.info("✅ Teacher and student models have completely consistent structure")
            else:
                logger.warning("⚠️  Teacher and student models have inconsistent structure, this may cause training issues")
                
            logger.info("💡 Tip: Use utils/student_model_loader.py to correctly load the model after training")
    else:
        if is_main_process():
            logger.info(f"✅ Vocabulary size completely matched: {actual_model_embedding_size}")
    
    student_model.train()

    # ========== Dataset loading ==========
    train_dataset = load_from_disk(os.path.join(args.data_dir, "train"))
    validation_dataset = load_from_disk(os.path.join(args.data_dir, "validation"))
    logger.info("Loaded datasets: train=%d, validation=%d", len(train_dataset), len(validation_dataset))

    # ========== Data Collator ==========
    if args.use_opencode_dataset:
        # Use OpenCodeInstruct collator for code generation
        data_collator = OpenCodeInstructCollator(
            tokenizer=tokenizer,
            total_diff_steps=args.total_diff_steps,
            use_nested_masking=args.use_nested_masking,
            max_length=1024
        )
        logger.info("Using OpenCodeInstruct data collator")
    else:
        # Default: GSM8K collator for math problems
        data_collator = ConsistencyDistillationCollator(
            tokenizer=tokenizer,
            total_diff_steps=args.total_diff_steps,
            use_nested_masking=args.use_nested_masking
        )
        logger.info("Using default data collator")

    # ========== Callbacks ==========
    callbacks = [
        LoggingCallback(),
        EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
    ]

    # ========== Trainer ==========
    trainer = ConsistencyTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        lambda_val=args.lambda_val,
        use_dynamic_lambda=args.use_dynamic_lambda,
        initial_lambda=args.initial_lambda,
        final_lambda=args.final_lambda,
        lambda_warmup_ratio=args.lambda_warmup_ratio,
        temperature=args.temperature,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks
    )

    # ========== Log Lambda scheduling and temperature strategy ==========
    if args.use_dynamic_lambda:
        logger.info("Using dynamic Lambda scheduling: initial_lambda=%.3f, final_lambda=%.3f, warmup_ratio=%.3f", 
                   args.initial_lambda, args.final_lambda, args.lambda_warmup_ratio)
    else:
        logger.info("Using static Lambda: %.3f", args.lambda_val)
    
    logger.info("Soft distillation temperature: %.2f", args.temperature)
    logger.info("Early stopping patience: %d", args.early_stopping_patience)

    logger.info("Starting training...")

    # ========== Training ==========
    output = trainer.train()
    logger.info("Training completed. Final metrics: %s", str(output.metrics))

    # ========== Save model ==========
    final_checkpoint_path = os.path.join(args.output_dir, "final_checkpoint")
    trainer.save_model(final_checkpoint_path)
    tokenizer.save_pretrained(final_checkpoint_path)
    logger.info("Model saved to %s", final_checkpoint_path)

if __name__ == "__main__":
    main() 