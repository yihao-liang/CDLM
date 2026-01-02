from .modeling_llada import LLaDAModelLM
import torch
from transformers import AutoConfig

class StudentModel(LLaDAModelLM):
    """
    Inherits from teacher model (LLaDAModelLM)
    """
    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Directly use the parent class's from_pretrained method
        # LLaDAModelLM has already correctly implemented model loading logic
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
