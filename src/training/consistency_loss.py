import torch
import torch.nn.functional as F
import math
from transformers.trainer import Trainer
from accelerate.state import DistributedType

def compute_lambda_cosine_schedule(global_step, max_steps,
                                  warmup_ratio=0.1,      # First 10% warmup phase
                                  initial_lambda=0.9,    # Initial weight (at training start)
                                  final_lambda=0.5):     # Final weight (at training end)
    """
    A flexible Lambda scheduler with piecewise cosine scheduling.
    - Warmup phase: lambda remains at initial value.
    - Scheduling phase: lambda smoothly changes from initial to final value following cosine curve.
    
    Supports two modes:
    1. Ascending mode (initial < final): e.g., 0.0 → 0.5, learn standard answers first then teacher method
    2. Descending mode (initial > final): e.g., 0.8 → 0.2, learn teacher method first then standard answers
    """
    pct = global_step / max_steps
    if pct < warmup_ratio:           # step1 warmup phase
        return initial_lambda
    else:                            # step2 scheduling phase
        # Recalculate scheduling interval percentage (from 0 to 1)
        schedule_pct = (pct - warmup_ratio) / (1 - warmup_ratio)
        # Calculate standard cosine decay value (from 1 to 0)
        cos_val = 0.5 * (1 + math.cos(math.pi * schedule_pct))
        # Convert to smooth transition curve (from 0 to 1) through (1 - cos_val)
        smooth_transition = 1 - cos_val
        # Map to [initial_lambda, final_lambda] interval (supports ascending or descending)
        return initial_lambda + (final_lambda - initial_lambda) * smooth_transition

class ConsistencyTrainer(Trainer):
    """
    DeepSpeed version of ConsistencyTrainer
    Total_Loss = λ * Loss_consistency + (1 - λ) * Loss_reconstruction
    Supports dynamic lambda scheduling and soft distillation with temperature
    """
    def __init__(
        self,
        teacher_model: torch.nn.Module,
        lambda_val: float = None,  # Compatible with static lambda
        # Dynamic lambda parameters
        use_dynamic_lambda: bool = False,
        initial_lambda: float = 0.9,    # Lambda value at early training
        final_lambda: float = 0.5,      # Lambda value at training end
        lambda_warmup_ratio: float = 0.1,
        # Soft distillation temperature parameter
        temperature: float = 2.0,       # Temperature for softmax scaling
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        
        # Store tokenizer for in-training evaluation
        self.tokenizer = kwargs.get('processing_class', None)
        
        # Temperature parameter for soft distillation
        self.temperature = temperature
        
        # Lambda scheduling configuration
        self.use_dynamic_lambda = use_dynamic_lambda
        if use_dynamic_lambda:
            self.initial_lambda = initial_lambda
            self.final_lambda = final_lambda 
            self.lambda_warmup_ratio = lambda_warmup_ratio
            # Calculate total training steps
            self.total_steps = (
                len(self.get_train_dataloader()) * self.args.num_train_epochs // self.args.gradient_accumulation_steps
            )
        else:
            # Static lambda, maintain backward compatibility
            self.lambda_val = lambda_val if lambda_val is not None else 0.5
        
        # Ensure teacher model is on correct device
        if torch.cuda.is_available():
            self.teacher_model = self.teacher_model.to('cuda')
    
    def get_current_lambda(self):
        """Get current step's lambda value"""
        if not self.use_dynamic_lambda:
            return self.lambda_val
        
        return compute_lambda_cosine_schedule(
            global_step=self.state.global_step,
            max_steps=self.total_steps,
            warmup_ratio=self.lambda_warmup_ratio,
            initial_lambda=self.initial_lambda,
            final_lambda=self.final_lambda
        )
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute loss function - DeepSpeed optimized version, supports dynamic lambda
        """
        # 1. Separate inputs
        labels = inputs.pop("labels") 
        teacher_consistency_input_ids = inputs.pop("teacher_consistency_input_ids")
        answer_start_indices = inputs.pop("answer_start_indices")

        # 2. Ensure all inputs are on correct device
        device = next(model.parameters()).device
        if labels.device != device:
            labels = labels.to(device)
        if teacher_consistency_input_ids.device != device:
            teacher_consistency_input_ids = teacher_consistency_input_ids.to(device)
        if answer_start_indices.device != device:
            answer_start_indices = answer_start_indices.to(device)

        # 3. Student model forward pass
        student_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        student_logits = student_outputs.get("logits")

        # 4. Teacher model forward pass
        with torch.no_grad():
            # Ensure teacher model is on correct device
            if next(self.teacher_model.parameters()).device != device:
                self.teacher_model = self.teacher_model.to(device)
            teacher_outputs = self.teacher_model(input_ids=teacher_consistency_input_ids)
            target_logits = teacher_outputs.logits

        # Early NaN/Inf detection in logits - if model is corrupted, skip this batch
        has_nan = torch.isnan(student_logits).any() or torch.isinf(student_logits).any()
        if has_nan:
            # Replace NaN/Inf with zeros to create a valid zero-loss with gradient
            student_logits_clean = torch.nan_to_num(student_logits, nan=0.0, posinf=0.0, neginf=0.0)
            zero_loss = (student_logits_clean * 0.0).sum()
            if self.is_world_process_zero():
                self.log({
                    "loss_total": 0.0,
                    "loss_consistency": 0.0,
                    "loss_reconstruction": 0.0,
                    "lambda": self.get_current_lambda(),
                    "temperature": self.temperature,
                    "skipped_nan_logits": True
                })
            return (zero_loss, student_outputs) if return_outputs else zero_loss

        # 5. Reconstruction Loss
        loss_fct_ce = torch.nn.CrossEntropyLoss()
        actual_vocab_size = student_logits.shape[-1]
        loss_reconstruction = loss_fct_ce(student_logits.view(-1, actual_vocab_size), labels.view(-1))

        # 6. Consistency Loss with Temperature Scaling (Soft Distillation)
        # Use labels != -100 to determine valid positions (collator already filtered question/padding)
        loss_mask = (labels != -100)

        active_loss = loss_mask.view(-1)
        num_active = active_loss.sum().item()

        # Handle empty active positions (all labels are -100)
        if num_active == 0:
            loss_consistency = torch.tensor(0.0, device=student_logits.device, dtype=student_logits.dtype)
        else:
            active_student_logits = student_logits.view(-1, actual_vocab_size)[active_loss]
            active_target_logits = target_logits.view(-1, actual_vocab_size)[active_loss]

            # Apply temperature scaling to both student and teacher logits for soft distillation
            loss_consistency = F.kl_div(
                F.log_softmax(active_student_logits / self.temperature, dim=-1),
                F.softmax(active_target_logits / self.temperature, dim=-1),
                reduction='batchmean',
                log_target=False
            )

            # Scale the loss by T^2 to compensate for the temperature scaling
            loss_consistency = loss_consistency * (self.temperature ** 2)

        # 7. Get current lambda value
        current_lambda = self.get_current_lambda()

        # 8. Total loss with NaN/Inf protection
        # Clip both losses to prevent gradient explosion (use conservative max=5.0)
        loss_consistency = torch.clamp(loss_consistency, min=0.0, max=5.0)
        loss_reconstruction = torch.clamp(loss_reconstruction, min=0.0, max=5.0)

        total_loss = current_lambda * loss_consistency + (1 - current_lambda) * loss_reconstruction

        # Final NaN/Inf check on loss values
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            student_logits_clean = torch.nan_to_num(student_logits, nan=0.0, posinf=0.0, neginf=0.0)
            total_loss = (student_logits_clean * 0.0).sum()
            loss_consistency = torch.tensor(0.0, device=student_logits.device)
            loss_reconstruction = torch.tensor(0.0, device=student_logits.device)
        
        # Log losses (only on main process)
        if self.is_world_process_zero():
            self.log({
                "loss_total": total_loss.item(),
                "loss_consistency": loss_consistency.item(),
                "loss_reconstruction": loss_reconstruction.item(),
                "lambda": current_lambda,
                "temperature": self.temperature
            })

        return (total_loss, student_outputs) if return_outputs else total_loss 