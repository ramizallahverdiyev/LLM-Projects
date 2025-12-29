import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig
from peft import PeftModel, PeftConfig
from datasets import Dataset

def load_inference_model(model_name: str, model_path: str, num_labels: int):
    """
    Load the base model and then the LoRA adapters for inference.
    """
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    base_model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    return model

def predict_entities(text: str, model, tokenizer, id2label, device=None):
    """
    Input: raw text string
    Output: list of dicts with entity info
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize input
    tokens = text.split()
    encoding = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True)
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
    
    # Initialize with a dummy token for consistent word_idx handling at the start
    previous_word_idx = -1 
    
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        
        # Only consider the first subword of a word for prediction
        if word_idx != previous_word_idx:
            label_id = predictions[idx]
            label = id2label[label_id]

            if label != "O":
                # If a new entity starts or current entity changes type
                if current_entity is None or current_entity["entity"] != label:
                    if current_entity is not None: # Save previous entity if exists
                        entities.append(current_entity)
                    current_entity = {"entity": label, "start_word_idx": word_idx, "end_word_idx": word_idx}
                else: # Continue current entity
                    current_entity["end_word_idx"] = word_idx
            else: # Current token is "O"
                if current_entity is not None: # Save previous entity if exists
                    entities.append(current_entity)
                    current_entity = None
        
        previous_word_idx = word_idx
            
    # Save any remaining entity
    if current_entity is not None:
        entities.append(current_entity)

    # Map start/end word indices to token strings
    for e in entities:
        e["text"] = " ".join(tokens[e["start_word_idx"]: e["end_word_idx"] + 1])
        del e["start_word_idx"]
        del e["end_word_idx"]

    return entities

if __name__ == "__main__":
    from src.data.dataset_loader import get_label_mapping
    from src.utils.helpers import load_config

    model_config = load_config("configs/model_config.yaml")
    inference_config = load_config("configs/inference_config.yaml")

    # Example text
    sample_text = "Prezidentin müvafiq sərəncamlarına uyğun Bibiheybət məscid-ziyarətgah da təmir işləri aparıldı."

    # Load label mapping
    id2label, label2id = get_label_mapping(model_config["label_list"])

    # Load model and tokenizer
    model = load_inference_model(model_config["model_name"], inference_config["model_path"], model_config["num_labels"])
    tokenizer = AutoTokenizer.from_pretrained(inference_config["tokenizer_path"])

    entities = predict_entities(sample_text, model, tokenizer, id2label, device="cpu")
    print("Predicted entities:", entities)
