import torch
import math
from typing import Any, Dict, List
from transformers import PreTrainedTokenizer


class ConsistencyDistillationCollator:
    """
    Consistency distillation collator with nested masking strategy.

    Key features:
    1. Teacher masks are always a subset of student masks (teacher_mask ⊆ student_mask)
    2. Question parts are preserved as unchanging conditions
    3. Loss computation only on answer regions
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, total_diff_steps: int = 100,
                 use_nested_masking: bool = True):
        self.tokenizer = tokenizer
        self.total_diff_steps = total_diff_steps
        self.use_nested_masking = use_nested_masking

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id

        # LLaDA models use <|mdm_mask|> (ID 126336) but tokenizer.mask_token_id may be None
        self.mask_token_id = self.tokenizer.mask_token_id
        if self.mask_token_id is None:
            # Fallback: try to encode the mask token directly
            mask_token_ids = self.tokenizer.encode("<|mdm_mask|>", add_special_tokens=False)
            if len(mask_token_ids) == 1:
                self.mask_token_id = mask_token_ids[0]
            else:
                # Last resort: use hardcoded LLaDA mask token ID
                self.mask_token_id = 126336

    def _get_mask_ratios(self) -> tuple:
        """
        Get student and teacher mask ratios.

        Student: U(0.40, 0.90)
        Teacher: clip(r_s × U(0.30, 0.70), 0.10, 0.60)

        Returns:
            (student_ratio, teacher_ratio)
        """
        import random
        r_s = random.uniform(0.40, 0.90)
        ratio = random.uniform(0.30, 0.70)
        r_t = max(0.10, min(0.60, r_s * ratio))
        return r_s, r_t

    def _mask_sequence_nested(self, ids: torch.Tensor, ratio_student: float, ratio_teacher: float, condition_len: int):
        """Apply nested masking ensuring teacher_mask ⊆ student_mask."""
        seq_len = ids.size(0)
        if seq_len <= condition_len:
            return ids.clone(), ids.clone()

        # Get answer region indices
        answer_indices = torch.arange(condition_len, seq_len)
        answer_length = len(answer_indices)
        
        # Short answer protection
        if answer_length < 20:
            ratio_student = min(ratio_student, 0.50)  # Force r_s <= 0.50 for L < 20
        
        if answer_length <= 10:
            # Direct setting for very short answers
            num_student_mask = 2
            num_teacher_mask = 1
        else:
            # Calculate number of tokens to mask
            ratio_student = max(0.0, min(1.0, ratio_student))
            ratio_teacher = max(0.0, min(1.0, ratio_teacher))
            
            num_student_mask = int(answer_length * ratio_student)
            num_teacher_mask = int(answer_length * ratio_teacher)
            
            # Apply minimum mask count: m_s = max(2, ceil(0.10 * L))
            min_student_mask = max(2, math.ceil(0.10 * answer_length))
            num_student_mask = max(num_student_mask, min_student_mask)
            
            num_teacher_mask = min(num_teacher_mask, num_student_mask)  # Ensure subset
        
        # Ensure we don't exceed available tokens
        num_student_mask = min(num_student_mask, answer_length)
        num_teacher_mask = min(num_teacher_mask, num_student_mask)
        
        if num_student_mask == 0:
            return ids.clone(), ids.clone()
        
        # Apply nested masking
        student_ids = ids.clone()
        teacher_ids = ids.clone()
        
        # Randomly select indices for student
        perm = torch.randperm(len(answer_indices))
        student_mask_indices = answer_indices[perm[:num_student_mask]]
        teacher_mask_indices = student_mask_indices[:num_teacher_mask]  # Subset of student
        
        student_ids[student_mask_indices] = self.mask_token_id
        teacher_ids[teacher_mask_indices] = self.mask_token_id
        
        return student_ids, teacher_ids

    def _cosine_schedule_original(self, t: int) -> float:
        """Original cosine schedule from the very first version."""
        s = 0.008
        t_scaled = t / self.total_diff_steps
        return math.sin((t_scaled + s) / (1 + s) * math.pi * 0.5)**2

    def _mask_sequence_original(self, ids: torch.Tensor, ratio: float, condition_len: int) -> torch.Tensor:
        """Original independent masking strategy - completely random and independent."""
        seq_len = ids.size(0)
        answer_len = seq_len - condition_len
        if answer_len <= 0: 
            return ids.clone()
        
        ratio = max(0.0, min(1.0, ratio))
        num_to_mask = int(answer_len * ratio)
        if num_to_mask == 0: 
            return ids.clone()
        
        # No special token protection - mask any answer tokens
        answer_indices = torch.arange(condition_len, seq_len)
        indices_to_mask = answer_indices[torch.randperm(len(answer_indices))[:num_to_mask]]
        
        masked_ids = ids.clone()
        masked_ids[indices_to_mask] = self.mask_token_id
        return masked_ids

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        questions = [ex.get('question', '') for ex in examples]
        answers = [ex.get('answer', '') for ex in examples]
        combined_texts = [f"{q} {a}" for q, a in zip(questions, answers)]
        
        # Tokenize everything once to get the original sequences (x0)
        # We don't pad here yet because we need the original lengths
        tokenized_combined = self.tokenizer(
            combined_texts, padding=False, truncation=True, max_length=512
        )
        x0_id_list = [torch.tensor(ids) for ids in tokenized_combined['input_ids']]
        
        # Use robust method for both nested and independent masking
        # Calculate answer start indices using "question + space" tokenization for precise alignment
        q_with_space = [f"{q} " for q in questions]
        answer_start_indices = [
            len(self.tokenizer(qs, padding=False, truncation=True, max_length=512)['input_ids'])
            for qs in q_with_space
        ]
        
        batch_size = len(x0_id_list)
        
        student_inputs_t_batch = []
        teacher_inputs_t_prime_batch = []

        for i in range(batch_size):
            current_x0_ids = x0_id_list[i]
            condition_len = answer_start_indices[i]
            
            if self.use_nested_masking:
                # Current advanced strategy: nested masking with GSM8K protection
                # Get mask ratios: Student [0.40, 0.90], Teacher [0.10, 0.60]
                mask_ratio_t, mask_ratio_t_prime = self._get_mask_ratios()
                
                # Apply nested masking
                xt_ids, xt_prime_ids = self._mask_sequence_nested(
                    current_x0_ids, mask_ratio_t, mask_ratio_t_prime, condition_len
                )
            else:
                # Original strategy: completely independent random masking
                t = torch.randint(1, self.total_diff_steps + 1, (1,)).item()
                t_prime = torch.randint(0, t, (1,)).item()
                mask_ratio_t = self._cosine_schedule_original(t)
                mask_ratio_t_prime = self._cosine_schedule_original(t_prime)
                
                # Apply independent masking (teacher and student completely independent)
                xt_ids = self._mask_sequence_original(current_x0_ids, mask_ratio_t, condition_len)
                xt_prime_ids = self._mask_sequence_original(current_x0_ids, mask_ratio_t_prime, condition_len)
            
            student_inputs_t_batch.append(xt_ids)
            teacher_inputs_t_prime_batch.append(xt_prime_ids)

        # First, pad student and teacher inputs to the maximum length in the batch
        student_input_ids = torch.nn.utils.rnn.pad_sequence(
            student_inputs_t_batch, batch_first=True, padding_value=self.pad_token_id
        )
        teacher_consistency_input_ids = torch.nn.utils.rnn.pad_sequence(
            teacher_inputs_t_prime_batch, batch_first=True, padding_value=self.pad_token_id
        )

        # Create attention mask for student input
        student_attention_mask = (student_input_ids != self.pad_token_id).long()

        # ========================= Key fix: process labels after all sequences are handled =========================
        # Now, create labels tensor with correct final padding length
        final_padded_length = student_input_ids.shape[1]
        labels = torch.full((batch_size, final_padded_length), self.pad_token_id, dtype=torch.long)
        for i in range(batch_size):
            original_len = len(x0_id_list[i])
            labels[i, :original_len] = x0_id_list[i]

        # Create mask for student masked positions
        student_mask_bool = (student_input_ids == self.mask_token_id)
        if self.use_nested_masking:
            # Only need teacher mask in nested mode
            teacher_mask_bool = (teacher_consistency_input_ids == self.mask_token_id)
        
        # Set labels to -100 for: question part, padding, and positions where loss should not be computed
        for i in range(batch_size):
            # Ignore question part when calculating loss
            labels[i, :answer_start_indices[i]] = -100
        # Ignore padding tokens when calculating loss
        labels[labels == self.pad_token_id] = -100
        
        # Choose loss computation strategy based on masking mode
        if self.use_nested_masking:
            # Nested masking: teacher_mask ⊆ student_mask, compute loss on student masked positions
            labels[torch.logical_not(student_mask_bool)] = -100
        else:
            # Independent masking (original strategy): compute loss on entire answer region
            # Don't filter by mask positions - let model learn to predict full answer regardless of mask pattern
            pass  # No additional masking - compute loss on all answer tokens
        # =========================================================================================
            
        return {
            "input_ids": student_input_ids,
            "attention_mask": student_attention_mask,
            "teacher_consistency_input_ids": teacher_consistency_input_ids,
            "labels": labels, 
            "answer_start_indices": torch.tensor(answer_start_indices, dtype=torch.long),
        }