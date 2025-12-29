import sys
import os
import torch
from tqdm import tqdm
from src.utils.helpers import load_config
from src.data.dataset_loader import load_hf_dataset, get_label_mapping
from src.data.preprocessing import get_tokenizer, tokenize_and_align_labels
from src.inference.predict import load_inference_model
from src.evaluation.metrics import compute_metrics
from src.evaluation.error_analysis import analyze_errors

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def evaluate():

    # 1. Setup
    model_config = load_config("configs/model_config.yaml")
    training_config = load_config("configs/training_config.yaml") # Load training config
    inference_config = load_config("configs/inference_config.yaml")
    device = inference_config["device"] if torch.cuda.is_available() else "cpu"

    # 2. Load dataset and tokenizer
    dataset = load_hf_dataset(model_config["dataset_name"])
    if "train" in dataset and "train_subset_size" in training_config and training_config["train_subset_size"] < 1.0:

        dataset["train"] = dataset["train"].shuffle(seed=training_config["seed"]).select(range(int(len(dataset["train"]) * training_config["train_subset_size"])))

    tokenizer = get_tokenizer(inference_config["tokenizer_path"])
    tokenized_dataset = dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True, remove_columns=["tokens","ner_tags"])
    id2label, _ = get_label_mapping(model_config["label_list"])

    # 3. Load model
    model = load_inference_model(model_config["model_name"], inference_config["model_path"], model_config["num_labels"])
    model.to(device)
    model.eval()

    # 4. Evaluation
    all_predictions, all_labels = [], []
    for split_name in ["validation", "test"]:
        split_data = tokenized_dataset[split_name]
        for example in tqdm(split_data, desc=f"Evaluating on {split_name}"):
            input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(example["attention_mask"]).unsqueeze(0).to(device)
            labels = example["labels"]
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).squeeze().cpu().tolist()
            filtered_preds, filtered_labels = [], []
            for p, l in zip(preds, labels):
                if l != -100:
                    filtered_preds.append(id2label[p])
                    filtered_labels.append(id2label[l])
            all_predictions.append(filtered_preds)
            all_labels.append(filtered_labels)
            
    results = compute_metrics(all_predictions, all_labels)
    print(f"Validation/Test results: {results}")
    
    errors = analyze_errors(all_predictions, all_labels, id2label)
    print(f"Error analysis: {errors}")

if __name__ == "__main__":
    evaluate()