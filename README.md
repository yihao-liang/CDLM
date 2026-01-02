# CD<sup>4</sup>LM: Consistency Distillation and Adaptive Decoding for Diffusion Language Models

Official implementation of **CD<sup>4</sup>LM**, a framework for accelerating Diffusion Language Models (DLMs) via Discrete-Space Consistency Distillation (DSCD) and Confidence-Adaptive Decoding (CAD).

## Overview

CD<sup>4</sup>LM addresses the fundamental **static-to-dynamic misalignment** between training and inference in DLMs:
- **DSCD**: Trains a trajectory-invariant student model that maps diverse noisy states directly to clean distributions
- **CAD**: Dynamically allocates compute based on token confidence, aggressively skipping steps without quality collapse

### Key Results

| Benchmark | Baseline | CD<sup>4</sup>LM | Speedup |
|-----------|----------|-------|---------|
| GSM8K | 77.4% | 77.6% | **5.18×** |
| HumanEval | 38.7% | 40.9% | **3.30×** |
| MBPP | 36.9% | 39.0% | **2.96×** |
| MATH500 | 37.3% | 38.6% | **5.33×** |

**Average: 3.62× speedup while improving accuracy**

## Pretrained Models

CD<sup>4</sup>LM distills from the open-source [GSAI-ML/LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) checkpoint.  
- Download the checkpoint via `git lfs clone` or `huggingface-cli download`, then point `TEACHER_MODEL_PATH` in `run_training.sh` (and `MODEL_PATH` in inference/eval scripts) to the local directory.
- The same checkpoint can be used for zero-shot CAD inference if you only want to reproduce the paper results without re-training.

## Installation

```bash
git clone https://github.com/yihao-liang/CDLM.git
cd CDLM
pip install -r requirements.txt
```

### Requirements
- Python >= 3.10
- CUDA >= 11.8 (or ROCm >= 6.0)
- 8× GPUs with >= 40GB memory (for training)

## Project Structure

```
CDLM/
├── run_training.sh             # Training launcher
├── inference_demo.py           # Inference example
│
├── src/
│   ├── model/                  # LLaDA model architecture
│   ├── training/               # DSCD training
│   │   ├── train.py            # Training script
│   │   └── consistency_loss.py # Trainer and losses
│   └── data/                   # Data collators
│
├── scripts/
│   ├── prepare_data.py         # Dataset preparation
│   ├── LLaDA_generate.py       # Fixed-step generation
│   └── LLaDA_generate_dynamic.py  # CAD generation
│
├── evaluation/
│   ├── evaluation_script.py    # Main evaluation entry
│   ├── dllm_eval/              # Evaluation framework
│   ├── metrics/                # Task-specific metrics
│   └── scripts/                # Example evaluation scripts
│       ├── eval_gsm8k.sh
│       ├── eval_mbpp.sh
│       ├── eval_humaneval.sh
│       └── eval_math.sh
│
└── configs/                    # Training configurations
```

## Data Preparation

```bash
# GSM8K (math reasoning)
python scripts/prepare_data.py --dataset gsm8k --output_dir /data/gsm8k

# OpenCodeInstruct (code generation, 200k subset by default)
python scripts/prepare_data.py --dataset opencode --output_dir /data/opencode

# Full OpenCodeInstruct (5M samples, takes a while)
python scripts/prepare_data.py --dataset opencode --output_dir /data/opencode_full --max_samples 5000000
```

## Training (DSCD)

```bash
# Edit paths in run_training.sh first
bash run_training.sh
```

Key parameters in `run_training.sh`:
| Parameter | Description | Default |
|-----------|-------------|---------|
| `TEACHER_MODEL_PATH` | Pre-trained LLaDA model | - |
| `TEMPERATURE` | Distillation temperature | 2.0 |
| `INITIAL_LAMBDA` | Initial curriculum weight | 0.9 |
| `FINAL_LAMBDA` | Final curriculum weight | 0.5 |

## Inference (CAD)

Key parameters for generation:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `confidence_threshold` | Token acceptance threshold | 0.95 |
| `block_length` | Block size for semi-autoregressive decoding | 32 |
| `gen_length` | Maximum generation length | 256 |
| `temperature` | Sampling temperature (0 = greedy) | 0.0 |

## Evaluation

```bash
# GSM8K (math reasoning)
bash evaluation/scripts/eval_gsm8k.sh

# MATH500 (math reasoning)
bash evaluation/scripts/eval_math.sh

# MBPP (code generation)
bash evaluation/scripts/eval_mbpp.sh

# HumanEval (code generation)
bash evaluation/scripts/eval_humaneval.sh
```

See [evaluation/README.md](evaluation/README.md) for more details.

## Citation

```bibtex

```

## Acknowledgments

- [LLaDA](https://github.com/ML-GSAI/LLaDA) - Base diffusion language model
- [DAEDAL](https://github.com/Li-Jinsong/DAEDAL) - Evaluation framework

## License

Apache 2.0
