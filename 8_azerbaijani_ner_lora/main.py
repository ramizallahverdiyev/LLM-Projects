from src.utils.helpers import set_seed, make_dirs, load_config
from src.utils.logging import get_logger
from src.data.dataset_loader import load_hf_dataset, get_label_mapping
from src.data.preprocessing import get_tokenizer, tokenize_and_align_labels
from src.model.model_builder import load_base_model, apply_lora, get_optimizer
from src.model.trainer import train

import torch

def main():
    # 1. Setup
    model_config = load_config("configs/model_config.yaml")
    training_config = load_config("configs/training_config.yaml")
    
    set_seed(training_config["seed"])
    logger = get_logger("NER_LoRA")
    output_dir = training_config["output_dir"]
    make_dirs(output_dir)
    
    # 2. Load dataset
    dataset = load_hf_dataset(model_config["dataset_name"])
    dataset["train"] = dataset["train"].shuffle(seed=training_config["seed"]).select(range(int(len(dataset["train"]) * training_config["train_subset_size"])))
    logger.info(f"Dataset loaded. Train size: {len(dataset['train'])}, Val size: {len(dataset['validation'])}")

    # Label mapping
    id2label, label2id = get_label_mapping(model_config["label_list"])

    # 3. Tokenizer & preprocessing
    tokenizer = get_tokenizer(model_config["model_name"])
    tokenized_dataset = dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True, remove_columns=["tokens", "ner_tags"])
    logger.info("Dataset tokenized and labels aligned.")

    # 4. Model + LoRA + optimizer
    model = load_base_model(model_config["model_name"], model_config["num_labels"])
    model = apply_lora(model, training_config["lora_r"], training_config["lora_alpha"], training_config["lora_dropout"])
    optimizer = get_optimizer(model, training_config["learning_rate"])

    # 5. Train
    train(
        model,
        tokenized_dataset,
        tokenizer,
        optimizer,
        id2label,
        output_dir=output_dir,
        batch_size=training_config["batch_size"],
        epochs=training_config["epochs"],
        lr_scheduler_type=training_config["lr_scheduler_type"],
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info("Training complete.")

if __name__ == "__main__":
    main()