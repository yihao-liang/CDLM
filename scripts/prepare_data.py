#!/usr/bin/env python3
"""
Prepare datasets for CDLM training.

Supported datasets:
- gsm8k: Math reasoning (from openai/gsm8k)
- opencode: Code generation (from nvidia/OpenCodeInstruct)

Usage:
    python scripts/prepare_data.py --dataset gsm8k --output_dir /data/gsm8k
    python scripts/prepare_data.py --dataset opencode --output_dir /data/opencode --max_samples 200000
"""
import datasets
import os
import argparse


DATASET_CONFIGS = {
    "gsm8k": {
        "name": "gsm8k",
        "config": "main",
        "split": "train",
        "default_samples": None,  # Use all (~7.5k)
        "val_ratio": 0.1,
    },
    "opencode": {
        "name": "nvidia/OpenCodeInstruct",
        "config": None,
        "split": "train",
        "default_samples": 200000,  # Subset by default (full is 5M)
        "val_ratio": 0.05,
        "trust_remote_code": True,
    },
}


def prepare_dataset(dataset_type: str, output_dir: str, max_samples: int = None, val_ratio: float = None):
    """
    Download and prepare dataset for training.

    Args:
        dataset_type: One of 'gsm8k', 'opencode'
        output_dir: Directory to save the processed dataset
        max_samples: Maximum samples to use (None for all)
        val_ratio: Validation split ratio
    """
    if dataset_type not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_type}. Supported: {list(DATASET_CONFIGS.keys())}")

    config = DATASET_CONFIGS[dataset_type]

    # Use defaults if not specified
    if max_samples is None:
        max_samples = config["default_samples"]
    if val_ratio is None:
        val_ratio = config["val_ratio"]

    print(f"=== Preparing {dataset_type} dataset ===")
    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Build load arguments
    load_kwargs = {"trust_remote_code": config.get("trust_remote_code", False)}
    if config["config"]:
        load_kwargs["name"] = config["config"]

    # Load dataset
    if max_samples:
        print(f"Loading {max_samples:,} samples...")
        split_str = f'{config["split"]}[:{max_samples}]'
    else:
        print("Loading full dataset...")
        split_str = config["split"]

    dataset = datasets.load_dataset(config["name"], split=split_str, **load_kwargs)
    print(f"Loaded {len(dataset):,} samples")

    # Split into train/validation
    print(f"Splitting: {(1-val_ratio)*100:.0f}% train, {val_ratio*100:.0f}% validation")

    split_dataset = dataset.train_test_split(test_size=val_ratio, shuffle=True, seed=42)
    split_dataset["validation"] = split_dataset.pop("test")

    print(f"  Train: {len(split_dataset['train']):,}")
    print(f"  Validation: {len(split_dataset['validation']):,}")

    # Save
    for split_name, split_data in split_dataset.items():
        split_path = os.path.join(output_dir, split_name)
        print(f"Saving {split_name} to: {split_path}")
        split_data.save_to_disk(split_path)

    print(f"\nDone! Use --data_dir {output_dir} for training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare datasets for CDLM training")
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset to prepare: gsm8k, opencode"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        required=True,
        help="Directory to save the dataset"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples (default: all for gsm8k, 200k for opencode)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=None,
        help="Validation split ratio (default: 0.1 for gsm8k, 0.05 for opencode)"
    )

    args = parser.parse_args()

    prepare_dataset(
        dataset_type=args.dataset,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        val_ratio=args.val_ratio,
    )
