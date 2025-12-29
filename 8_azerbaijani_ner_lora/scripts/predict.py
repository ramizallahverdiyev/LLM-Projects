import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.utils.helpers import load_config
from src.data.dataset_loader import get_label_mapping
from src.data.preprocessing import get_tokenizer
from src.inference.predict import load_inference_model, predict_entities as predict_entities_from_inference_module

def predict(text):
    # 1. Setup
    model_config = load_config("configs/model_config.yaml")
    inference_config = load_config("configs/inference_config.yaml")
    
    device = inference_config["device"] if torch.cuda.is_available() else "cpu"

    # 2. Load tokenizer and model
    tokenizer = get_tokenizer(inference_config["tokenizer_path"])
    model = load_inference_model(model_config["model_name"], inference_config["model_path"], model_config["num_labels"])
    
    id2label, _ = get_label_mapping(model_config["label_list"])

    # 3. Predict
    entities = predict_entities_from_inference_module(text, model, tokenizer, id2label, device)

    return entities

if __name__ == "__main__":
    sample_text = "Prezidentin müvafiq sərəncamlarına uyğun Bibiheybət məscid-ziyarətgah da təmir işləri aparıldı."
    entities = predict(sample_text)
    print(entities)