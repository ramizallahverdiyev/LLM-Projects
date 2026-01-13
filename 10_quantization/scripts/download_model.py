# scripts/download_model.py

import os
from huggingface_hub import snapshot_download

def download_model():
    """
    Downloads a model from the Hugging Face Hub.
    """
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # The directory where the model will be saved.
    # We save it in the 'models/base' directory, which is at the root of the project.
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(script_dir, "models", "base")
    
    # Create the directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    print(f"Downloading model: {model_id} to {model_dir}")

    # Download the model snapshot from Hugging Face
    snapshot_download(
        repo_id=model_id,
        local_dir=model_dir,
        local_dir_use_symlinks=False,  # Set to False on Windows
        ignore_patterns=["*.safetensors"],  # We want the pytorch .bin files
    )

    print("Model download complete.")

if __name__ == "__main__":
    download_model()
