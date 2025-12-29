import torch
from transformers import AutoTokenizer
from src.model.model_builder import load_base_model, apply_lora
from datasets import Dataset

def load_model_and_tokenizer(model_path: str, model_name="xlm-roberta-base"):
    """
    Load trained LoRA model and tokenizer from a checkpoint directory.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = load_base_model(model_name=model_name)
    model = apply_lora(model)
    model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin", map_location="cpu"))
    model.eval()
    return model, tokenizer

def predict_entities(text_tokens, model, tokenizer, id2label, device=None):
    """
    Input: list of tokens
    Output: list of dicts with entity info
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize input
    encoding = tokenizer(text_tokens, is_split_into_words=True, return_tensors="pt", truncation=True)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).squeeze().cpu().tolist()
    word_ids = encoding.word_ids()

    # Align predictions to original tokens
    entities = []
    current_entity = None
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        label_id = predictions[idx]
        label = id2label[label_id]

        if label != "O":
            if current_entity is None:
                current_entity = {"entity": label, "start": word_idx, "end": word_idx}
            elif current_entity["entity"] == label:
                current_entity["end"] = word_idx
            else:
                entities.append(current_entity)
                current_entity = {"entity": label, "start": word_idx, "end": word_idx}
        else:
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None
    if current_entity is not None:
        entities.append(current_entity)

    # Map start/end indices to token strings
    for e in entities:
        e["text"] = " ".join(text_tokens[e["start"]: e["end"] + 1])

    return entities

if __name__ == "__main__":
    from src.data.dataset_loader import get_label_mapping

    # Example tokens
    tokens = ["Prezidentin", "müvafiq", "sərəncamlarına", "uyğun", "Bibiheybət", "məscid-ziyarətgah", "də", "təmir", "işləri", "aparıldı", "."]

    # Load label mapping
    id2label, label2id = get_label_mapping()

    model_path = "experiments/run_001/checkpoints/best_model"
    model, tokenizer = load_model_and_tokenizer(model_path)

    entities = predict_entities(tokens, model, tokenizer, id2label)
    print("Predicted entities:", entities)
