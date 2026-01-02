#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=0

# Model path
MODEL_PATH="/exp/code_distilled/final_checkpoint"
OUTPUT_ROOT="/exp/humaneval_eval"

# Generation settings
GEN_LENGTH=256
BLOCK_LENGTH=256
BATCH_SIZE=1

# Dynamic decoding parameters
CONFIDENCE_THRESHOLD=0.95
MIN_TOKENS_PER_STEP=1
MAX_TOKENS_PER_STEP=32
EOS_THRESHOLD_RATIO=0.3

mkdir -p "$OUTPUT_ROOT"

# ============================================
# HumanEval pass@1 (temperature=0, greedy decoding)
# ============================================
OUTPUT_DIR="${OUTPUT_ROOT}/humaneval_pass1"
mkdir -p "$OUTPUT_DIR"

GEN_KWARGS="use_dynamic=true,gen_length=${GEN_LENGTH},block_length=${BLOCK_LENGTH},temperature=0,confidence_threshold=${CONFIDENCE_THRESHOLD},min_tokens_per_step=${MIN_TOKENS_PER_STEP},max_tokens_per_step=${MAX_TOKENS_PER_STEP},eos_threshold_ratio=${EOS_THRESHOLD_RATIO}"

python evaluation/evaluation_script.py \
    -m dllm_eval \
    --model LLaDA \
    --tasks humaneval \
    --batch_size ${BATCH_SIZE} \
    --model_args "pretrained=${MODEL_PATH}" \
    --gen_kwargs "${GEN_KWARGS}" \
    --num_fewshot 0 \
    --output_path "${OUTPUT_DIR}" \
    --log_samples \
    --confirm_run_unsafe_code

python evaluation/metrics/humaneval.py --res_path "${OUTPUT_DIR}" --k_values "1"