#!/usr/bin/env python3
"""
CDLM Inference Demo - Confidence-Adaptive Decoding (CAD)
"""
import time
import torch
from transformers import AutoTokenizer
from src.model import LLaDAModelLM
from scripts.LLaDA_generate_dynamic import generate_block

# ============ Configuration ============
MODEL_PATH = "/path/to/your/model"  # Change this
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CAD Parameters
GEN_LENGTH = 256
BLOCK_LENGTH = 32
CONFIDENCE_THRESHOLD = 0.95
TEMPERATURE = 0.0

# ============ Load Model ============
print(f"Loading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = LLaDAModelLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = model.to(DEVICE).eval()
print("Model loaded.\n")

# ============ Example Prompt ============
PROMPT = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

# ============ Inference ============
print(f"Prompt: {PROMPT}")
print("-" * 50)

# Tokenize
messages = [{"role": "user", "content": PROMPT}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"].to(DEVICE)

# Warmup
with torch.no_grad():
    _ = model(input_ids)

# Generate with CAD
torch.cuda.synchronize()
start_time = time.perf_counter()

output_ids, stats = generate_block(
    model=model,
    prompt=input_ids,
    gen_length=GEN_LENGTH,
    block_length=BLOCK_LENGTH,
    temperature=TEMPERATURE,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    tokenizer=tokenizer,
    verbose=False,
)

torch.cuda.synchronize()
end_time = time.perf_counter()

# Decode
generated_ids = output_ids[0, input_ids.shape[1]:]
response = tokenizer.decode(generated_ids, skip_special_tokens=True)

# Statistics
elapsed = end_time - start_time
num_tokens = len(generated_ids)
num_steps = len(stats) if isinstance(stats, list) else 0
tokens_per_step = num_tokens / num_steps if num_steps > 0 else 0
tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0

print(f"Response:\n{response}\n")
print("-" * 50)
print(f"Generated tokens: {num_tokens}")
print(f"Total steps:      {num_steps}")
print(f"Tokens/step:      {tokens_per_step:.2f}")
print(f"Time:             {elapsed:.2f}s")
print(f"Throughput:       {tokens_per_sec:.1f} tokens/s")
print("=" * 50)
