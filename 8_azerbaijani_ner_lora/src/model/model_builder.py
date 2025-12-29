from transformers import AutoModelForTokenClassification, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
import torch

def load_base_model(model_name, num_labels):
    """
    Load the base model (XLM-R / mBERT) for token classification.
    """
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)
    return model

def apply_lora(model, r, lora_alpha, lora_dropout):
    """
    Apply LoRA adapters to the model for efficient fine-tuning.
    """
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_lin", "v_lin"],
        lora_dropout=lora_dropout,
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    return model

def get_optimizer(model, learning_rate):
    """
    Return AdamW optimizer for the model.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    return optimizer
