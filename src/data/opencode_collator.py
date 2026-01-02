"""
OpenCodeInstruct Consistency Distillation Collator

Handles data preparation for consistency distillation training on OpenCodeInstruct dataset.
OpenCodeInstruct contains instruction-following code generation tasks with:
- input: Programming question/instruction
- output: Solution code
- unit_tests: Generated unit tests (optional)
"""
import torch
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase


@dataclass
class OpenCodeInstructCollator:
    """
    Data collator for OpenCodeInstruct consistency distillation.

    For code generation tasks, we mask parts of the code solution while
    keeping the instruction/question as context.
    """
    tokenizer: PreTrainedTokenizerBase
    total_diff_steps: int = 1000
    max_length: int = 1024  # OpenCodeInstruct has longer sequences
    pad_to_multiple_of: Optional[int] = None
    use_nested_masking: bool = True

    def __post_init__(self):
        # Ensure we have a mask token
        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({"mask_token": "<|mdm_mask|>"})

    def get_mask_ratios(self) -> tuple:
        """
        Get student and teacher mask ratios.

        Student: U(0.40, 0.90)
        Teacher: clip(r_s × U(0.30, 0.70), 0.10, 0.60)

        Returns:
            (student_ratio, teacher_ratio)
        """
        r_s = random.uniform(0.40, 0.90)
        ratio = random.uniform(0.30, 0.70)
        r_t = max(0.10, min(0.60, r_s * ratio))
        return r_s, r_t

    def format_input(self, example: Dict[str, Any]) -> str:
        """
        Format OpenCodeInstruct example into a single string.
        Combines instruction with code solution.

        OpenCodeInstruct fields:
        - input: Programming question/instruction
        - output: Solution code
        """
        # Create a prompt with the instruction
        prompt = f"Instruction: {example['input']}\n\n"

        # Add the solution
        prompt += "Solution:\n"
        prompt += example['output']

        return prompt

    def find_code_start_index(
        self,
        input_ids: List[int],
        tokenizer: PreTrainedTokenizerBase
    ) -> int:
        """
        Find where the code solution starts in the tokenized input.
        We look for "Solution:\n" pattern.

        Args:
            input_ids: Tokenized input
            tokenizer: Tokenizer instance

        Returns:
            Index where code starts
        """
        # Tokenize the marker
        solution_marker = tokenizer.encode("Solution:\n", add_special_tokens=False)
        marker_len = len(solution_marker)

        # Find the marker in input_ids
        for i in range(len(input_ids) - marker_len + 1):
            if input_ids[i:i+marker_len] == solution_marker:
                return i + marker_len

        # If not found, use a heuristic (mask last 60% of tokens)
        # OpenCodeInstruct instructions tend to be longer
        return int(len(input_ids) * 0.4)

    def apply_masking(
        self,
        input_ids: List[int],
        mask_ratio: float,
        code_start_idx: int
    ) -> List[int]:
        """
        Apply masking to the code portion of the input.

        Args:
            input_ids: Tokenized input
            mask_ratio: Proportion of code tokens to mask
            code_start_idx: Index where code starts

        Returns:
            Masked input_ids
        """
        masked_ids = input_ids.copy()

        # Only mask the code portion
        code_length = len(input_ids) - code_start_idx
        if code_length <= 0:
            return masked_ids

        # Calculate number of tokens to mask
        num_to_mask = int(code_length * mask_ratio)
        if num_to_mask == 0:
            return masked_ids

        # Get indices of code tokens
        code_indices = list(range(code_start_idx, len(input_ids)))

        # Randomly select tokens to mask
        mask_indices = random.sample(code_indices, min(num_to_mask, len(code_indices)))

        # Apply masking
        mask_token_id = self.tokenizer.mask_token_id
        for idx in mask_indices:
            masked_ids[idx] = mask_token_id

        return masked_ids

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch for OpenCodeInstruct consistency distillation.

        Creates two versions of each example:
        - x_t: Student input (higher noise, more masking)
        - x_t_prime: Teacher input (lower noise, less masking)

        Args:
            examples: List of dataset examples

        Returns:
            Dictionary with student/teacher inputs and labels
        """
        batch_size = len(examples)

        # Format all examples
        formatted_texts = [self.format_input(ex) for ex in examples]

        # Tokenize all examples
        tokenized = self.tokenizer(
            formatted_texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        # Find code start indices for each example
        code_start_indices = []
        for i in range(batch_size):
            input_ids = tokenized['input_ids'][i].tolist()
            code_start_idx = self.find_code_start_index(input_ids, self.tokenizer)
            code_start_indices.append(code_start_idx)

        # Initialize outputs
        student_input_ids = []
        teacher_input_ids = []

        for i in range(batch_size):
            input_ids = tokenized['input_ids'][i].tolist()
            code_start_idx = code_start_indices[i]

            # Get mask ratios: Student [0.40, 0.90], Teacher [0.10, 0.60]
            mask_ratio_t, mask_ratio_t_prime = self.get_mask_ratios()

            # Apply masking
            if self.use_nested_masking:
                # Nested masking: ensure teacher_mask ⊆ student_mask
                # Generate student mask first
                x_t = self.apply_masking(input_ids, mask_ratio_t, code_start_idx)

                # Teacher mask is subset of student mask
                x_t_prime = input_ids.copy()
                mask_token_id = self.tokenizer.mask_token_id

                # Find all student masked positions
                student_mask_positions = [j for j in range(code_start_idx, len(x_t))
                                         if x_t[j] == mask_token_id]

                # Calculate how many teacher masks we need
                code_length = len(input_ids) - code_start_idx
                num_teacher_mask = int(code_length * mask_ratio_t_prime)
                num_teacher_mask = min(num_teacher_mask, len(student_mask_positions))

                # Randomly select subset of student mask positions for teacher
                if num_teacher_mask > 0 and len(student_mask_positions) > 0:
                    teacher_mask_positions = random.sample(student_mask_positions, num_teacher_mask)
                    for j in teacher_mask_positions:
                        x_t_prime[j] = mask_token_id
            else:
                # Independent masking: teacher and student completely independent
                x_t = self.apply_masking(input_ids, mask_ratio_t, code_start_idx)
                x_t_prime = self.apply_masking(input_ids, mask_ratio_t_prime, code_start_idx)

            student_input_ids.append(x_t)
            teacher_input_ids.append(x_t_prime)

        # Convert to tensors
        student_input_ids = torch.tensor(student_input_ids, dtype=torch.long)
        teacher_input_ids = torch.tensor(teacher_input_ids, dtype=torch.long)

        # Create attention masks
        student_attention_mask = (student_input_ids != self.tokenizer.pad_token_id).long()
        teacher_attention_mask = (teacher_input_ids != self.tokenizer.pad_token_id).long()

        # Labels are the original unmasked input_ids
        labels = tokenized['input_ids'].clone()

        # Filter labels to compute loss only on code region
        # 1. Set instruction/question part to -100 (ignore in loss)
        for i in range(batch_size):
            labels[i, :code_start_indices[i]] = -100

        # 2. Set padding tokens to -100 (ignore in loss)
        labels[labels == self.tokenizer.pad_token_id] = -100

        # 3. If nested masking, only compute loss on student masked positions
        if self.use_nested_masking:
            student_mask_bool = (student_input_ids == self.tokenizer.mask_token_id)
            labels[torch.logical_not(student_mask_bool)] = -100

        return {
            'input_ids': student_input_ids,
            'attention_mask': student_attention_mask,
            'teacher_consistency_input_ids': teacher_input_ids,
            'labels': labels,
            'answer_start_indices': torch.tensor(code_start_indices, dtype=torch.long)
        }
