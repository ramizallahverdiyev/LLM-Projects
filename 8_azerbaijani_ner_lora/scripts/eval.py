from src.utils.helpers import set_seed
from src.utils.logging import get_logger
from src.data.dataset_loader import load_hf_dataset, split_dataset, get_label_mapping
from src.data.preprocessing import get_tokenizer, tokenize_and_align_labels
from src.model.model_builder import load_base_model, apply_lora
from src.evaluation.metrics import compute_metrics
from src.evaluation.error_analysis import analyze_errors

import torch

def main():
    set_seed(42)
    logger = get_logger("Eval")

    # Load dataset & label mapping
    ds = load_hf_dataset()
    dataset = split_dataset(ds)
    id2label, label2id = get_label_mapping()
    logger.info("Dataset loaded for evaluation.")

    # Tokenizer & preprocessing
    tokenizer = get_tokenizer()
    tokenized_dataset = tokenize_and_align_labels(dataset, tokenizer)
    logger.info("Dataset tokenized for evaluation.")

    # Load trained model
    model_path = "experiments/run_001/checkpoints/best_model"
    model = load_base_model()
    model = apply_lora(model)
    model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin", map_location="cpu"))
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Run evaluation
    all_predictions, all_labels = [], []

    for split_name in ["validation", "test"]:
        split_data = tokenized_dataset[split_name]
        for example in split_data:
            input_ids = example["input_ids"].unsqueeze(0).to(device)
            attention_mask = example["attention_mask"].unsqueeze(0).to(device)
            labels = example["labels"]

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).squeeze().cpu().tolist()
            # Remove ignored tokens
            filtered_preds, filtered_labels = [], []
            for p, l in zip(preds, labels):
                if l != -100:
                    filtered_preds.append(p)
                    filtered_labels.append(l)
            all_predictions.append(filtered_preds)
            all_labels.append(filtered_labels)

    # Compute metrics
    results = compute_metrics(all_predictions, all_labels)
    logger.info(f"Evaluation results: {results}")

    # Error analysis
    errors = analyze_errors(all_predictions, all_labels, id2label)
    logger.info(f"Error analysis: {errors}")

if __name__ == "__main__":
    main()
