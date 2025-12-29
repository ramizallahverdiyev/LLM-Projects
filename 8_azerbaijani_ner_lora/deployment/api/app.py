from fastapi import FastAPI
from pydantic import BaseModel
import torch
import os
import sys

# Add the project root to sys.path to import local modules
# Assuming app.py is in 8_azerbaijani_ner_lora/deployment/api
# and the project root is 8_azerbaijani_ner_lora
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.helpers import load_config
from src.data.dataset_loader import get_label_mapping
from src.data.preprocessing import get_tokenizer
from src.inference.predict import load_inference_model, predict_entities as predict_entities_from_inference_module

app = FastAPI(
    title="Azerbaijani NER LoRA API",
    description="API for Named Entity Recognition in Azerbaijani using a LoRA-tuned model.",
    version="0.1.0",
)

class TextInput(BaseModel):
    text: str

# Global variables for model and tokenizer to be loaded once
model = None
tokenizer = None
id2label = None
device = None

@app.on_event("startup")
async def load_model():
    global model, tokenizer, id2label, device

    # 1. Setup
    # Adjust paths relative to the project root (where the configs folder is)
    model_config = load_config("configs/model_config.yaml")
    inference_config = load_config("configs/inference_config.yaml")
    
    device = inference_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load tokenizer and model
    # tokenizer_path and model_path in configs are likely relative to project root or specified explicitly
    # Assuming model_path in inference_config.yaml points to a directory containing the model artifacts
    tokenizer = get_tokenizer(inference_config["tokenizer_path"])
    
    # Load the fine-tuned model (LoRA adapter) on top of the base model
    num_labels = model_config["num_labels"]
    model_name = model_config["model_name"]
    lora_model_path = inference_config["model_path"] # This should point to the saved LoRA model dir

    model = load_inference_model(model_name, lora_model_path, num_labels)
    
    id2label, _ = get_label_mapping(model_config["label_list"])

@app.post("/predict")
async def predict_ner(input: TextInput):
    if model is None or tokenizer is None or id2label is None:
        return {"error": "Model not loaded. Please ensure startup event completed successfully."}, 500
    
    try:
        entities = predict_entities_from_inference_module(input.text, model, tokenizer, id2label, device)
        return {"text": input.text, "entities": entities}
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
