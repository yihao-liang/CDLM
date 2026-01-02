# CDLM Evaluation

Evaluation framework for CDLM, built on `dllm_eval` from [preordinary/LLaDA](https://github.com/preordinary/LLaDA).

## Directory Structure

```
evaluation/
├── evaluation_script.py         # Main entry point
│
├── scripts/                     # Example evaluation scripts
│   ├── eval_gsm8k.sh            # GSM8K evaluation
│   ├── eval_math.sh             # MATH500 evaluation
│   ├── eval_mbpp.sh             # MBPP evaluation
│   └── eval_humaneval.sh        # HumanEval evaluation
│
├── dllm_eval/                   # Evaluation framework (fork of lm-eval-harness)
│   ├── __main__.py              # CLI entry point
│   ├── evaluator.py             # Core evaluation logic
│   ├── models/LLaDA.py          # LLaDA model with CAD support
│   └── tasks/                   # Task definitions
│       ├── gsm8k/
│       ├── math500/
│       ├── mbpp/
│       └── humaneval/
│
└── metrics/                     # Post-processing metrics
    ├── gsm8k.py                 # GSM8K answer extraction
    ├── math500.py               # MATH500 answer extraction
    ├── mbpp.py                  # MBPP code extraction
    └── humaneval.py             # HumanEval code extraction
```

## Model Checkpoints

All evaluation scripts expect `MODEL_PATH` to point to a local copy of [GSAI-ML/LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) (or to your distilled CD<sup>4</sup>LM checkpoint).

- Download the model with `git lfs clone https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct` or `huggingface-cli download GSAI-ML/LLaDA-8B-Instruct --local-dir /path/to/model`.
- Update `MODEL_PATH` (and optionally `OUTPUT_ROOT`, `CUDA_VISIBLE_DEVICES`, decoding flags, etc.) inside `evaluation/scripts/*.sh` before running them.
- When evaluating a newly trained student, set `MODEL_PATH` to `/path/to/your/OUTPUT_DIR/final_checkpoint` instead.

## Quick Start

### Using Example Scripts

```bash
# GSM8K with dynamic decoding (CAD)
./evaluation/scripts/eval_gsm8k.sh

# MATH500
./evaluation/scripts/eval_math.sh

# MBPP (pass@1)
./evaluation/scripts/eval_mbpp.sh

# HumanEval (pass@1)
./evaluation/scripts/eval_humaneval.sh
```

### Direct Usage

```bash
python evaluation/evaluation_script.py \
    -m dllm_eval \
    --model LLaDA \
    --tasks gsm8k \
    --batch_size 1 \
    --model_args "pretrained=/path/to/model" \
    --gen_kwargs "use_dynamic=true,gen_length=256,block_length=32,confidence_threshold=0.95" \
    --output_path ./results/gsm8k \
    --log_samples
```

## Decoding Modes

### Dynamic Decoding (CAD) - Recommended

Confidence-Adaptive Decoding with early stopping:

```bash
--gen_kwargs "use_dynamic=true,gen_length=256,block_length=32,confidence_threshold=0.95"
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_dynamic` | false | Enable CAD |
| `gen_length` | 256 | Maximum generation length |
| `block_length` | 32 | Block size for semi-autoregressive decoding |
| `confidence_threshold` | 0.95 | Token acceptance threshold |
| `temperature` | 0.0 | Sampling temperature |

### Fixed-Step Decoding

Original LLaDA decoding with fixed diffusion steps:

```bash
--gen_kwargs "gen_length=512,block_length=32,steps=512"
```

## Post-Processing

Run metrics extraction after evaluation:

```bash
python evaluation/metrics/gsm8k.py --res_path ./results/gsm8k/
python evaluation/metrics/math500.py --res_path ./results/math500/
python evaluation/metrics/mbpp.py --res_path ./results/mbpp/
python evaluation/metrics/humaneval.py --res_path ./results/humaneval/
```

## Credits

- `dllm_eval` framework from [preordinary/LLaDA](https://github.com/preordinary/LLaDA)
- Based on [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
