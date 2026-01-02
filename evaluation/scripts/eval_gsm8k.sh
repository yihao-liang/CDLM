#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=0

# Model path
MODEL_PATH="/exp/math_distilled/final_checkpoint"
OUTPUT_ROOT="/exp/gsm8k_eval"

# Generation settings
GEN_LENGTH=256
BLOCK_LENGTH=32
BATCH_SIZE=1

# Dynamic decoding parameters
CONFIDENCE_THRESHOLD=0.95
MIN_TOKENS_PER_STEP=1
MAX_TOKENS_PER_STEP=32
EOS_THRESHOLD_RATIO=0.3

mkdir -p "$OUTPUT_ROOT"

OUTPUT_DIR="${OUTPUT_ROOT}/dynamic_b${BLOCK_LENGTH}"
mkdir -p "$OUTPUT_DIR"

GEN_KWARGS="use_dynamic=true,gen_length=${GEN_LENGTH},block_length=${BLOCK_LENGTH},temperature=0,confidence_threshold=${CONFIDENCE_THRESHOLD},min_tokens_per_step=${MIN_TOKENS_PER_STEP},max_tokens_per_step=${MAX_TOKENS_PER_STEP},eos_threshold_ratio=${EOS_THRESHOLD_RATIO}"

python evaluation/evaluation_script.py \
    -m dllm_eval \
    --model LLaDA \
    --tasks gsm8k \
    --batch_size ${BATCH_SIZE} \
    --model_args "pretrained=${MODEL_PATH},assistant_prefix=<reasoning> " \
    --gen_kwargs "${GEN_KWARGS}" \
    --num_fewshot 0 \
    --output_path "${OUTPUT_DIR}" \
    --log_samples

python evaluation/metrics/gsm8k.py --res_path "${OUTPUT_DIR}"


# Uncomment below to run fixed-step evaluation

# STEPS=256
# BLOCK_LENGTH=32  # Use 32 for faster inference, 256 for full block
# OUTPUT_DIR="${OUTPUT_ROOT}/fixed_s${STEPS}_b${BLOCK_LENGTH}"
# mkdir -p "$OUTPUT_DIR"
#
# GEN_KWARGS="use_dynamic=false,gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},temperature=0"
#
# python evaluation/evaluation_script.py \
#     -m dllm_eval \
#     --model LLaDA \
#     --tasks gsm8k \
#     --batch_size ${BATCH_SIZE} \
#     --model_args "pretrained=${MODEL_PATH},assistant_prefix=<reasoning> " \
#     --gen_kwargs "${GEN_KWARGS}" \
#     --num_fewshot 0 \
#     --output_path "${OUTPUT_DIR}" \
#     --log_samples
#
# python evaluation/metrics/gsm8k.py --res_path "${OUTPUT_DIR}"
