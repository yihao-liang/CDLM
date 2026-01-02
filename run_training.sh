#!/bin/bash

export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Memory allocator settings to prevent OOM from fragmentation
# AMD GPUs (HIP)
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
# NVIDIA GPUs (CUDA) - uncomment if using NVIDIA
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. Path settings
TEACHER_MODEL_PATH="/models/LLaDA-8B-Instruct"
DATA_DIR="/data/gsm8k"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/exp/gsm8k_8b_${TIMESTAMP}"

# 2. Training hyperparameters
EPOCHS=10
TRAIN_BATCH_SIZE=2
EVAL_BATCH_SIZE=2
LEARNING_RATE=5e-6
GRAD_ACC_STEPS=4
WARMUP_STEPS=200

# 3. Early Stopping
EARLY_STOPPING_PATIENCE=15

# 4. Logging, evaluation and save frequency
LOGGING_STEPS=20
EVAL_STEPS=50
SAVE_STEPS=1000

# 5. Consistency distillation parameters
DIFFUSION_STEPS=1000
LAMBDA=0

# 6. Dynamic lambda scheduling
USE_DYNAMIC_LAMBDA=true
INITIAL_LAMBDA=0.9
FINAL_LAMBDA=0.5
LAMBDA_WARMUP_RATIO=0.1

# 7. Masking strategy
USE_NESTED_MASKING=true

# 8. Soft distillation temperature
TEMPERATURE=2.0

# Build arguments
DYNAMIC_LAMBDA_ARGS=""
if [ "$USE_DYNAMIC_LAMBDA" = true ]; then
    DYNAMIC_LAMBDA_ARGS="--use_dynamic_lambda --initial_lambda $INITIAL_LAMBDA --final_lambda $FINAL_LAMBDA --lambda_warmup_ratio $LAMBDA_WARMUP_RATIO"
fi

MASKING_ARGS=""
if [ "$USE_NESTED_MASKING" = true ]; then
    MASKING_ARGS="--use_nested_masking"
fi

accelerate launch --config_file ./configs/accelerate_config.yaml src/training/train.py \
    --teacher_model_path "$TEACHER_MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --learning_rate $LEARNING_RATE \
    --warmup_steps $WARMUP_STEPS \
    --logging_steps $LOGGING_STEPS \
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS \
    --total_diff_steps $DIFFUSION_STEPS \
    --lambda_val $LAMBDA \
    --temperature $TEMPERATURE \
    $DYNAMIC_LAMBDA_ARGS \
    $MASKING_ARGS

# Cleanup and copy config
[ $? -eq 0 ] && find "$OUTPUT_DIR" -maxdepth 1 -type d -name "checkpoint-*" -exec rm -rf {} +
[ $? -eq 0 ] && cp "$TEACHER_MODEL_PATH/configuration_llada.py" "$OUTPUT_DIR/final_checkpoint/" && cp "$TEACHER_MODEL_PATH/modeling_llada.py" "$OUTPUT_DIR/final_checkpoint/"
