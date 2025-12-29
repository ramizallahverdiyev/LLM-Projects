import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification, get_scheduler
from tqdm import tqdm
from evaluate import load as load_metric
import os

def train(
    model,
    tokenized_datasets,
    tokenizer,
    optimizer,
    output_dir="experiments/run_001",
    batch_size=8,
    epochs=3,
    device=None,
    lr_scheduler_type="linear",
):
    """
    Train the model with tokenized datasets and LoRA applied.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # DataLoaders
    train_loader = DataLoader(
        tokenized_datasets["train"], batch_size=batch_size, shuffle=True, collate_fn=data_collator
    )
    val_loader = DataLoader(
        tokenized_datasets["validation"], batch_size=batch_size, shuffle=False, collate_fn=data_collator
    )

    # Metric
    metric = load_metric("seqeval")

    # Scheduler
    num_training_steps = epochs * len(train_loader)
    scheduler = get_scheduler(
        lr_scheduler_type, optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Ensure output directories
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    best_f1 = 0.0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc="Training"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Average training loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        all_predictions = []
        all_labels = []

        for batch in tqdm(val_loader, desc="Validation"):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).cpu().tolist()
            labels = batch["labels"].cpu().tolist()

            # Flatten and remove -100
            for pred_seq, label_seq in zip(predictions, labels):
                filtered_pred, filtered_label = [], []
                for p, l in zip(pred_seq, label_seq):
                    if l != -100:
                        filtered_pred.append(p)
                        filtered_label.append(l)
                all_predictions.append(filtered_pred)
                all_labels.append(filtered_label)

        results = metric.compute(predictions=all_predictions, references=all_labels)
        val_f1 = results["overall_f1"]
        print(f"Validation F1: {val_f1:.4f}")

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = os.path.join(output_dir, "checkpoints", "best_model")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"Best model saved to {save_path}")

    print("Training complete.")

# =====================================
# 2. Example usage
# =====================================
if __name__ == "__main__":
    from src.data.preprocessing import get_tokenizer, tokenize_and_align_labels
    from src.data.dataset_loader import load_hf_dataset, split_dataset
    from src.model.model_builder import load_base_model, apply_lora, get_optimizer

    # Load dataset and tokenizer
    ds = load_hf_dataset()
    dataset = split_dataset(ds)
    tokenizer = get_tokenizer()

    # Tokenize
    tokenized_dataset = tokenize_and_align_labels(dataset, tokenizer)

    # Load model + LoRA + optimizer
    model = load_base_model()
    model = apply_lora(model)
    optimizer = get_optimizer(model)

    # Train
    train(model, tokenized_dataset, tokenizer, optimizer, batch_size=4, epochs=1)
