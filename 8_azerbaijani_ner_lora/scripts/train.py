from src.utils.helpers import set_seed, make_dirs
from src.utils.logging import get_logger
from src.data.dataset_loader import load_hf_dataset, split_dataset
from src.data.preprocessing import get_tokenizer, tokenize_and_align_labels
from src.model.model_builder import load_base_model, apply_lora, get_optimizer
from src.model.trainer import train
import torch

def main():
    # Set seed & logger
    set_seed(42)
    logger = get_logger()
    
    # Load dataset
    ds = load_hf_dataset()
    dataset = split_dataset(ds)
    logger.info(f"Dataset loaded. Train size: {len(dataset['train'])}, Val size: {len(dataset['validation'])}")

    # Tokenizer & preprocessing
    tokenizer = get_tokenizer()
    tokenized_dataset = tokenize_and_align_labels(dataset, tokenizer)
    logger.info("Dataset tokenized and labels aligned.")

    # Model + LoRA
    model = load_base_model()
    model = apply_lora(model)
    optimizer = get_optimizer(model)

    # Train
    train(model, tokenized_dataset, tokenizer, optimizer, batch_size=4, epochs=3, device="cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training complete.")

if __name__ == "__main__":
    main()
