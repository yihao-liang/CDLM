import torch
import numpy as np
import torch.nn.functional as F
import logging
from datetime import datetime
from pathlib import Path
import argparse

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    """Numerically stable Gumbel-Max perturbation."""
    if temperature is None or temperature == 0:
        return logits

    logits = logits.to(torch.float32)

    # Clamp the uniform sample away from 0/1 to avoid log(0)
    noise = torch.rand_like(logits, dtype=logits.dtype)
    eps = torch.finfo(noise.dtype).eps
    noise = noise.clamp_(eps, 1.0 - eps)

    gumbels = -torch.log(-torch.log(noise))
    # Scale temperature as in classical Gumbel-max: argmax((logits + g)/tau)
    tau = max(temperature, eps)
    return (logits / tau) + gumbels


def get_dynamic_transfer_tokens(confidence, threshold, min_tokens=1, max_tokens=4):
    '''
    Dynamically determine how many tokens to accept based on confidence threshold.
    Inspired by SDLM's adaptive acceptance mechanism, but adapted for LLaDA's
    masked language model approach.

    Args:
        confidence: Tensor of shape (batch_size, num_positions) with confidence scores
        threshold: Minimum confidence threshold for accepting tokens
        min_tokens: Minimum number of tokens to accept per step (default: 1)
        max_tokens: Maximum number of tokens to accept per step (default: 4)

    Returns:
        num_accept: Tensor of shape (batch_size,) indicating number of tokens to accept
    '''
    batch_size = confidence.shape[0]
    num_accept = torch.zeros(batch_size, dtype=torch.int64, device=confidence.device)

    for i in range(batch_size):
        # Filter out invalid positions (confidence = -inf from EOS masking/blocking)
        valid_mask = torch.isfinite(confidence[i]) & (confidence[i] > -float('inf'))
        valid_confidences = confidence[i][valid_mask]

        if len(valid_confidences) == 0:
            # No valid tokens available - force accept min_tokens to ensure progress
            # This prevents deadlock when all tokens are blocked (e.g., all EOS)
            num_accept[i] = min_tokens
            continue

        # Sort valid confidence scores in descending order
        sorted_conf, _ = torch.sort(valid_confidences, descending=True)

        # Only consider up to max_tokens positions
        top_k_conf = sorted_conf[:min(max_tokens, len(sorted_conf))]

        # Count how many of the top-k exceed the threshold
        above_threshold = (top_k_conf >= threshold).sum().item()

        # Accept at least min_tokens, or more if above_threshold is higher
        num_accept[i] = max(above_threshold, min_tokens)
        # But don't exceed available valid positions
        num_accept[i] = min(num_accept[i], len(valid_confidences))

    return num_accept


@torch.no_grad()
def generate_block(model, prompt, gen_length=128, block_length=32, temperature=0.,
                   cfg_scale=0., remasking='low_confidence', mask_id=126336,
                   confidence_threshold=0.8, entropy_threshold=-0.3,
                   min_tokens_per_step=1, max_tokens_per_step=4,
                   max_steps_per_block=None, tokenizer=None, verbose=True, eos_threshold_ratio=0.5,
                   selection_method='confidence'):
    '''
    Semi-autoregressive block dynamic generation.

    Each block is decoded using dynamic confidence-based acceptance,
    but blocks are processed sequentially (autoregressive between blocks).

    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        gen_length: Total generated answer length.
        block_length: Length of each block.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of <|mdm_mask|> is 126336.
        confidence_threshold: Threshold for accepting tokens within block.
        min_tokens_per_step: Minimum tokens to decode per step within block.
        max_tokens_per_step: Maximum tokens to decode per step within block.
        max_steps_per_block: Maximum steps per block (None = block_length).
        tokenizer: Tokenizer for verbose output (optional).
        verbose: Whether to print detailed decode information.
        eos_threshold_ratio: Allow EOS only after this ratio of TOTAL generation is complete.
        selection_method: Token selection strategy for unmasking.
            - 'confidence': Select tokens with confidence >= confidence_threshold (default)
            - 'random': Random selection among masked tokens
            - 'entropy': Select tokens with neg_entropy >= entropy_threshold

    Returns:
        x: Generated sequence tensor.
        all_stats: List of step statistics for all blocks.
    '''
    batch_size = 1
    device = model.device

    assert gen_length % block_length == 0, f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
    num_blocks = gen_length // block_length

    if max_steps_per_block is None:
        max_steps_per_block = block_length  # Conservative default

    # Initialize full sequence with masks
    x = torch.full((batch_size, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)
    prompt_len = prompt.shape[1]

    # Common EOS token IDs
    eos_token_ids = [126081, 151643, 151645, 126348]

    all_stats = []
    total_steps = 0

    logging.info(f"Starting block generation: {gen_length} tokens ({num_blocks} blocks × {block_length})")
    logging.info(f"Threshold: {confidence_threshold}, Min/Max tokens per step: {min_tokens_per_step}/{max_tokens_per_step}")
    logging.info(f"Max steps per block: {max_steps_per_block}")

    for block_idx in range(num_blocks):
        block_start = prompt_len + block_idx * block_length
        block_end = prompt_len + (block_idx + 1) * block_length

        block_step = 0
        block_stats = []

        while True:
            # Check if current block is complete
            block_mask = (x[:, block_start:block_end] == mask_id)
            remaining_masks = block_mask.sum().item()

            if remaining_masks == 0:
                logging.info(f"Block {block_idx + 1}/{num_blocks} completed in {block_step} steps")
                break

            # Check max steps per block
            if block_step >= max_steps_per_block:
                logging.warning(f"Block {block_idx + 1}: max steps reached, force completing {remaining_masks} tokens")
                force_complete = True
            else:
                force_complete = False

            # Forward pass on full sequence
            mask_index = (x == mask_id)

            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits = logits.to(torch.float32)
            if not torch.isfinite(logits).all():
                logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))

            logits = logits - logits.amax(dim=-1, keepdim=True)
            logits = torch.clamp(logits, min=-50.0, max=50.0)

            # Sample tokens
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            # Calculate confidence/selection scores based on selection_method
            p = F.softmax(logits, dim=-1)

            if selection_method == 'entropy':
                # neg_entropy = sum(p * log(p)) = -entropy
                # Higher neg_entropy = lower entropy = more certain = higher priority
                epsilon = 1e-10
                log_p = torch.log(p + epsilon)
                x0_p = torch.sum(p * log_p, dim=-1)  # neg_entropy
            elif selection_method == 'random':
                # Random selection scores
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:  # 'confidence' (default)
                # Max probability of predicted token as confidence
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

            # Override with random for remasking if specified
            if remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)

            # CRITICAL: Only consider current block for selection
            # Set confidence to -inf for positions outside current block
            x0_p[:, :block_start] = -np.inf   # Prompt + previous blocks
            x0_p[:, block_end:] = -np.inf     # Future blocks

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            # Progressive EOS control based on TOTAL generation progress
            tokens_generated_total = gen_length - (x[:, prompt_len:prompt_len + gen_length] == mask_id).sum().item()
            progress = tokens_generated_total / gen_length

            if progress < eos_threshold_ratio:
                for eos_id in eos_token_ids:
                    eos_mask = (x0[:, block_start:block_end] == eos_id)
                    if eos_mask.any():
                        global_eos_positions = torch.where(eos_mask)[1] + block_start
                        confidence[0, global_eos_positions] = -np.inf

            # Select tokens to unmask
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)

            if force_complete:
                # Force mode: accept all remaining in this block
                block_confidence = confidence[0, block_start:block_end]
                valid_positions = torch.where(block_confidence > -np.inf)[0]
                if len(valid_positions) > 0:
                    global_indices = block_start + valid_positions
                    transfer_index[0, global_indices] = True
                num_transferred = transfer_index.sum().item()
            else:
                # Dynamic mode: select based on confidence
                block_mask_relative = mask_index[0, block_start:block_end]
                mask_positions_relative = torch.where(block_mask_relative)[0]

                if len(mask_positions_relative) == 0:
                    num_transferred = 0
                else:
                    mask_confidences = confidence[0, block_start + mask_positions_relative]
                    valid_mask = torch.isfinite(mask_confidences)
                    valid_mask_positions = mask_positions_relative[valid_mask]
                    valid_confidences = mask_confidences[valid_mask]

                    if len(valid_mask_positions) == 0:
                        # All blocked, force min tokens
                        k = min(min_tokens_per_step, len(mask_positions_relative))
                        _, top_k_indices = torch.topk(mask_confidences, k=k)
                        selected_relative = mask_positions_relative[top_k_indices]
                    else:
                        # Use appropriate threshold based on selection method
                        if selection_method == 'entropy':
                            threshold = entropy_threshold
                        else:
                            threshold = confidence_threshold

                        above_threshold = (valid_confidences >= threshold).sum().item()
                        k = max(above_threshold, min_tokens_per_step)
                        k = min(k, max_tokens_per_step)
                        k = min(k, len(valid_mask_positions))

                        _, top_k_in_valid = torch.topk(valid_confidences, k=k)
                        selected_relative = valid_mask_positions[top_k_in_valid]

                    selected_global = block_start + selected_relative
                    transfer_index[0, selected_global] = True

                num_transferred = transfer_index.sum().item()

            # Update sequence
            x[transfer_index] = x0[transfer_index]

            # Log progress every step
            logging.info(
                f"Step {total_steps} (Block {block_idx + 1}): Transferred {num_transferred} tokens, "
                f"Remaining: {remaining_masks - num_transferred}/{block_length}"
            )

            # Verbose output: show what was decoded this step
            if verbose and tokenizer:
                # Get the indices that were just decoded in the generation region
                gen_transfer_index = transfer_index[0, prompt_len:prompt_len + gen_length]
                decoded_positions = torch.where(gen_transfer_index)[0].cpu().tolist()

                # Get current state of generation region
                gen_region = x[0, prompt_len:prompt_len + gen_length].cpu()

                # Prepare tokens for display
                decoded_tokens = []
                decoded_text_parts = []
                confidences_for_decoded = []

                for pos in decoded_positions:
                    token_id = gen_region[pos].item()
                    token_text = tokenizer.decode([token_id])
                    decoded_tokens.append(token_id)
                    decoded_text_parts.append(token_text)
                    # Get confidence for this position
                    conf_val = confidence[0, prompt_len + pos].item()
                    confidences_for_decoded.append(conf_val)

                print(f"\nStep {total_steps}:")
                print(f"  Decoded {len(decoded_positions)} tokens at positions: {decoded_positions}")
                print(f"  Token IDs: {decoded_tokens}")
                print(f"  Confidences: {[f'{c:.4f}' for c in confidences_for_decoded]}")
                print(f"  Text: {''.join(decoded_text_parts)}")

                # Show current full generation state with <mask> markers
                print(f"  Current generation (with <mask>):")
                gen_text_parts = []
                for i in range(gen_length):
                    if gen_region[i].item() == mask_id:
                        gen_text_parts.append("<mask>")
                    else:
                        gen_text_parts.append(tokenizer.decode([gen_region[i].item()]))

                full_text = ''.join(gen_text_parts)
                print(f"    {full_text}")
                print(f"  Remaining masks: {remaining_masks - num_transferred}/{gen_length}")

            block_stats.append({
                'block': block_idx,
                'step': block_step,
                'tokens': num_transferred,
                'forced': force_complete
            })

            block_step += 1
            total_steps += 1

            if force_complete:
                break

        all_stats.extend(block_stats)

    # Print summary statistics
    logging.info(f"\n{'='*60}")
    logging.info(f"Generation Summary:")
    logging.info(f"Total blocks: {num_blocks}")
    logging.info(f"Total steps: {total_steps}")
    logging.info(f"Tokens generated: {gen_length}")
    if total_steps > 0:
        logging.info(f"Average tokens per step: {gen_length / total_steps:.2f}")

    # Show step distribution
    step_tokens = [s['tokens'] for s in all_stats if not s['forced']]
    if step_tokens:
        logging.info(f"Token distribution per step: min={min(step_tokens)}, max={max(step_tokens)}, avg={sum(step_tokens)/len(step_tokens):.2f}")

    forced_steps = sum(1 for s in all_stats if s['forced'])
    if forced_steps > 0:
        logging.info(f"Forced completions: {forced_steps}")
    logging.info(f"{'='*60}\n")

    return x, all_stats


@torch.no_grad()
def generate(model, prompt, gen_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336,
             confidence_threshold=0.8, entropy_threshold=-0.3,
             min_tokens_per_step=1, max_tokens_per_step=4,
             max_steps=None, tokenizer=None, verbose=True, eos_threshold_ratio=0.5,
             selection_method='confidence'):

    '''
    Simplified dynamic generation without block concept (global mode).

    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        gen_length: Generated answer length.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of <|mdm_mask|> is 126336.
        confidence_threshold: Threshold for accepting tokens
        min_tokens_per_step: Minimum tokens to decode per step
        max_tokens_per_step: Maximum tokens to decode per step
        max_steps: Maximum number of decoding steps (None = unlimited)
        tokenizer: Tokenizer for verbose output (optional)
        verbose: Whether to print detailed decode information
        eos_threshold_ratio: Allow EOS only after this ratio of generation is complete (0.5 = 50%)
    '''
    batch_size = 1
    device = model.device

    # Initialize full sequence with masks
    x = torch.full((batch_size, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    # Generation region
    gen_start = prompt.shape[1]
    gen_end = prompt.shape[1] + gen_length

    # Statistics tracking
    step = 0
    step_stats = []

    logging.info(f"Starting generation: {gen_length} tokens to generate")
    logging.info(f"Threshold: {confidence_threshold}, Min/Max tokens per step: {min_tokens_per_step}/{max_tokens_per_step}")
    if max_steps:
        logging.info(f"Max steps: {max_steps}")

    if verbose and tokenizer:
        print(f"\n{'='*80}")
        print(f"DECODE PROCESS (watching generation region only)")
        print(f"{'='*80}")

    while True:
        # Check if generation is complete
        mask_index = (x == mask_id)
        gen_region_mask = mask_index[:, gen_start:gen_end]
        remaining_masks = gen_region_mask.sum().item()

        if remaining_masks == 0:
            logging.info(f"Generation completed in {step} steps")
            break

        # Check if max steps exceeded
        if max_steps is not None and step >= max_steps:
            logging.warning(
                f"Reached max steps ({max_steps}). "
                f"Force completing {remaining_masks} remaining tokens."
            )
            force_complete = True
        else:
            force_complete = False

        # Model forward pass
        if cfg_scale > 0.:
            un_x = x.clone()
            un_x[prompt_index] = mask_id
            x_ = torch.cat([x, un_x], dim=0)
            logits = model(x_).logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = model(x).logits

        # Safety check for NaN/inf in logits
        logits = logits.to(torch.float32)

        if not torch.isfinite(logits).all():
            bad = (~torch.isfinite(logits)).sum().item()
            logging.warning(
                f"NaN/Inf detected in logits at step {step} (count={bad}). Clamping to safe range."
            )
            logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))

        # Stabilise dynamic range before sampling
        logits = logits - logits.amax(dim=-1, keepdim=True)
        logits = torch.clamp(logits, min=-50.0, max=50.0)

        if not torch.isfinite(logits).all():
            logging.warning(
                f"Post-normalisation logits still non-finite at step {step}. Falling back to zeros."
            )
            logits = torch.zeros_like(logits)

        # Sample tokens
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)  # (batch_size, seq_len)

        # Calculate confidence/selection scores based on selection_method
        p = F.softmax(logits, dim=-1)

        if selection_method == 'entropy':
            # neg_entropy = sum(p * log(p)) = -entropy
            # Higher neg_entropy = lower entropy = more certain = higher priority
            epsilon = 1e-10
            log_p = torch.log(p + epsilon)
            x0_p = torch.sum(p * log_p, dim=-1)  # neg_entropy
        elif selection_method == 'random':
            # Random selection scores
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        else:  # 'confidence' (default)
            # Max probability of predicted token as confidence
            x0_p = torch.squeeze(
                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # (batch_size, seq_len)

        # Override with random for remasking if specified
        if remasking == 'random':
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)

        # Only consider generation region for confidence
        x0_p[:, :gen_start] = -np.inf  # Don't modify prompt
        x0_p[:, gen_end:] = -np.inf    # Don't modify beyond generation region

        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, -np.inf)

        # Progressive EOS control: Only allow EOS after sufficient context is built
        # Calculate generation progress (0.0 = just started, 1.0 = complete)
        tokens_generated = gen_length - remaining_masks
        progress = tokens_generated / gen_length

        # Common EOS token IDs for LLaDA models
        eos_token_ids = [126081, 151643, 151645, 126348]  # <|endoftext|>, <|im_end|>, <|eot_id|>

        # Block EOS tokens if progress is below threshold
        if progress < eos_threshold_ratio:
            num_blocked_total = 0
            for eos_id in eos_token_ids:
                eos_mask = (x0[:, gen_start:gen_end] == eos_id)
                if eos_mask.any():
                    # Set confidence to -inf for these positions
                    global_eos_positions = torch.where(eos_mask)[1] + gen_start
                    confidence[0, global_eos_positions] = -np.inf
                    num_blocked_total += eos_mask.sum().item()

            if verbose and tokenizer and num_blocked_total > 0:
                print(f"  [Progress {progress*100:.1f}%] Blocked {num_blocked_total} EOS token(s) - building context")

        # Determine which tokens to transfer
        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

        if force_complete:
            # Force mode: accept ALL remaining tokens in generation region
            # Since batch_size=1, directly process the single sample
            gen_confidence = confidence[0, gen_start:gen_end]
            valid_positions = torch.where(gen_confidence > -np.inf)[0]
            if len(valid_positions) > 0:
                global_indices = gen_start + valid_positions
                transfer_index[0, global_indices] = True

            num_transferred = transfer_index.sum().item()
            logging.info(f"Step {step}: Force completed {num_transferred} tokens")
            step_stats.append({
                'step': step,
                'tokens': num_transferred,
                'forced': True
            })
        else:
            # Dynamic mode: select based on confidence threshold
            # IMPORTANT: Only select from MASK positions in generation region

            # Get mask positions in generation region (relative indices)
            gen_region_mask_relative = mask_index[0, gen_start:gen_end]
            mask_positions_relative = torch.where(gen_region_mask_relative)[0]

            if len(mask_positions_relative) == 0:
                # No masks remaining, this shouldn't happen but handle gracefully
                num_transferred = 0
            else:
                # Get confidence values for mask positions only
                mask_confidences = confidence[0, gen_start + mask_positions_relative]

                # Filter out blocked positions (confidence = -inf from EOS blocking)
                valid_mask = torch.isfinite(mask_confidences) & (mask_confidences > -float('inf'))
                valid_mask_positions = mask_positions_relative[valid_mask]
                valid_confidences = mask_confidences[valid_mask]

                if len(valid_mask_positions) == 0:
                    # All mask positions are blocked (e.g., all predict EOS)
                    # Force unmask min_tokens from highest confidence blocked positions
                    # This prevents deadlock when all positions are EOS
                    k = min(min_tokens_per_step, len(mask_positions_relative))
                    _, top_k_indices = torch.topk(mask_confidences, k=k)
                    selected_relative = mask_positions_relative[top_k_indices]
                else:
                    # Determine how many tokens to unmask based on confidence/entropy
                    # Use appropriate threshold based on selection method
                    if selection_method == 'entropy':
                        threshold = entropy_threshold
                    else:
                        threshold = confidence_threshold

                    # Count how many exceed threshold
                    above_threshold = (valid_confidences >= threshold).sum().item()

                    # Unmask at least min_tokens, more if confidence is high
                    k = max(above_threshold, min_tokens_per_step)
                    k = min(k, max_tokens_per_step)  # Cap at max
                    k = min(k, len(valid_mask_positions))  # Can't exceed available

                    # Select top-k confident positions from valid masks
                    _, top_k_in_valid = torch.topk(valid_confidences, k=k)
                    selected_relative = valid_mask_positions[top_k_in_valid]

                # Convert relative indices to global indices
                selected_global = gen_start + selected_relative
                transfer_index[0, selected_global] = True

            num_transferred = transfer_index.sum().item()
            step_stats.append({
                'step': step,
                'tokens': num_transferred,
                'forced': False
            })

            if step % 10 == 0 or remaining_masks <= 10:
                logging.info(
                    f"Step {step}: Transferred {num_transferred} tokens, "
                    f"Remaining: {remaining_masks - num_transferred}/{gen_length}"
                )

        # Update sequence
        x[transfer_index] = x0[transfer_index]

        # Verbose output: show what was decoded this step
        if verbose and tokenizer:
            # Get the indices that were just decoded in the generation region
            gen_transfer_index = transfer_index[0, gen_start:gen_end]
            decoded_positions = torch.where(gen_transfer_index)[0].cpu().tolist()

            # Get current state of generation region
            gen_region = x[0, gen_start:gen_end].cpu()

            # Prepare tokens for display
            decoded_tokens = []
            decoded_text_parts = []
            confidences_for_decoded = []

            for pos in decoded_positions:
                token_id = gen_region[pos].item()
                token_text = tokenizer.decode([token_id])
                decoded_tokens.append(token_id)
                decoded_text_parts.append(token_text)
                # Get confidence for this position
                conf_val = confidence[0, gen_start + pos].item()
                confidences_for_decoded.append(conf_val)

            print(f"\nStep {step}:")
            print(f"  Decoded {len(decoded_positions)} tokens at positions: {decoded_positions}")
            print(f"  Token IDs: {decoded_tokens}")
            print(f"  Confidences: {[f'{c:.4f}' for c in confidences_for_decoded]}")
            print(f"  Text: {''.join(decoded_text_parts)}")

            # Show current full generation state with <mask> markers
            print(f"  Current generation (with <mask>):")
            gen_text_parts = []
            for i in range(gen_length):
                if gen_region[i].item() == mask_id:
                    gen_text_parts.append("<mask>")
                else:
                    gen_text_parts.append(tokenizer.decode([gen_region[i].item()]))

            full_text = ''.join(gen_text_parts)
            # Print full text without truncation
            print(f"    {full_text}")
            print(f"  Remaining masks: {remaining_masks - len(decoded_positions)}/{gen_length}")

        step += 1

        # Break immediately after force completion
        if force_complete:
            break

    # Print summary statistics
    logging.info(f"\n{'='*60}")
    logging.info(f"Generation Summary:")
    logging.info(f"Total steps: {step}")
    logging.info(f"Tokens generated: {gen_length}")
    logging.info(f"Average tokens per step: {gen_length / step:.2f}")

    # Show step distribution
    step_tokens = [s['tokens'] for s in step_stats if not s['forced']]
    if step_tokens:
        logging.info(f"Token distribution per step: min={min(step_tokens)}, max={max(step_tokens)}, avg={sum(step_tokens)/len(step_tokens):.2f}")

    forced_steps = sum(1 for s in step_stats if s['forced'])
    if forced_steps > 0:
        logging.info(f"Forced completions: {forced_steps}")
    logging.info(f"{'='*60}\n")

    return x, step_stats


def main():
    parser = argparse.ArgumentParser(description='Simplified LLaDA Dynamic Decode')
    parser.add_argument('--model_path', type=str,
                       default='./models/LLaDA-1.5',
                       help='Path to pretrained model')
    parser.add_argument('--prompt', type=str,
                       default='How are you',
                       help='Input prompt text')
    parser.add_argument('--gen_length', type=int, default=128,
                       help='Generated answer length')
    parser.add_argument('--temperature', type=float, default=0.,
                       help='Sampling temperature')
    parser.add_argument('--cfg_scale', type=float, default=0.,
                       help='Classifier-free guidance scale')
    parser.add_argument('--remasking', type=str, default='low_confidence',
                       choices=['low_confidence', 'random'],
                       help='Remasking strategy')

    # Dynamic decode parameters
    parser.add_argument('--confidence_threshold', type=float, default=0.8,
                       help='Confidence threshold for accepting tokens')
    parser.add_argument('--min_tokens_per_step', type=int, default=1,
                       help='Minimum tokens to decode per step')
    parser.add_argument('--max_tokens_per_step', type=int, default=4,
                       help='Maximum tokens to decode per step')
    parser.add_argument('--max_steps', type=int, default=None,
                       help='Maximum number of decoding steps (None = unlimited)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed decode process')
    parser.add_argument('--eos_threshold_ratio', type=float, default=0.5,
                       help='Allow EOS tokens only after this ratio of generation is complete (0.5 = 50%%)')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info(f"Loading model from {args.model_path}")
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    prompt = args.prompt

    # Check if it's an Instruct model or Base model
    if 'Instruct' in args.model_path or 'instruct' in args.model_path:
        # Add special tokens for Instruct model
        m = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    print(f"\n{'='*60}")
    print(f"Simplified Dynamic Decode (No Blocks, No Fixed Steps)")
    print(f"{'='*60}")
    print(f"Model: {args.model_path}")
    print(f"Generation length: {args.gen_length}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    print(f"Tokens per step: {args.min_tokens_per_step}-{args.max_tokens_per_step}")
    if args.max_steps:
        print(f"Max steps: {args.max_steps}")
    else:
        print(f"Max steps: Unlimited")
    print(f"{'='*60}\n")

    import time
    start_time = time.time()

    out, stats = generate(
        model,
        input_ids,
        gen_length=args.gen_length,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        remasking=args.remasking,
        confidence_threshold=args.confidence_threshold,
        min_tokens_per_step=args.min_tokens_per_step,
        max_tokens_per_step=args.max_tokens_per_step,
        max_steps=args.max_steps,
        tokenizer=tokenizer,
        verbose=args.verbose,
        eos_threshold_ratio=args.eos_threshold_ratio
    )

    elapsed_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Generation completed in {elapsed_time:.2f}s")
    print(f"Total steps: {len(stats)}")
    print(f"{'='*60}\n")
    print("Generated text:")
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()
