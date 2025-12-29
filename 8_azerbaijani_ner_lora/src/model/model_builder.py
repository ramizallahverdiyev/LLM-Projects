from transformers import AutoModelForTokenClassification, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
import torch

def load_base_model(model_name="xlm-roberta-base", num_labels=25):
    """
    Load the base model (XLM-R / mBERT) for token classification.
    """
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)
    return model

def apply_lora(model, r=8, lora_alpha=16, lora_dropout=0.1):
    """
    Apply LoRA adapters to the model for efficient fine-tuning.
    """
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,  # Token classification task
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["query", "value"],  # Apply LoRA to attention Q/V matrices
        lora_dropout=lora_dropout,
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    return model

def get_optimizer(model, learning_rate=5e-5):
    """
    Return AdamW optimizer for the model.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    return optimizer

if __name__ == "__main__":
    # Load base model
    model = load_base_model()
    print("Base model loaded.")

    # Apply LoRA
    model = apply_lora(model)
    print("LoRA adapter applied.")

    # Prepare optimizer
    optimizer = get_optimizer(model)
    print("Optimizer ready.")
