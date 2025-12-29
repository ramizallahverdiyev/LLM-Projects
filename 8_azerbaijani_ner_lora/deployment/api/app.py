from fastapi import FastAPI
from pydantic import BaseModel
import torch
import os
import sys

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

@app.get("/")
async def root():
    return {"message": "Welcome to the Azerbaijani NER LoRA API. Please refer to /docs for API documentation."}

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

    # Load model configuration
    model_config = load_config(os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'model_config.yaml'))

    # 1. Setup
    # Paths for deployment
    current_dir = os.path.dirname(__file__)
    lora_model_path = os.path.join(current_dir, "model_artifacts")
    tokenizer_path = os.path.join(current_dir, "model_artifacts")
    base_model_name = model_config["model_name"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Load tokenizer and model
    tokenizer = get_tokenizer(tokenizer_path)
    
    # Load the fine-tuned model (LoRA adapter) on top of the base model
    num_labels = model_config["num_labels"]
    model = load_inference_model(base_model_name, lora_model_path, num_labels=num_labels)
    
    label_list = model_config["label_list"]
    id2label, _ = get_label_mapping(label_list)

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
    uvicorn.run(app, host="127.0.0.1", port=8000)