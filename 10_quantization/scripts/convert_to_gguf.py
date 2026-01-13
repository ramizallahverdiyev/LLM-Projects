# scripts/convert_to_gguf.py

import os
from llama_cpp.converter import convert_hf_to_gguf

def convert_model_to_gguf():
    """
    Converts a Hugging Face model to the GGUF format.
    """
    # The directory where the base model is saved.
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(script_dir, "models", "base")
    
    # The output directory for the GGUF model.
    output_dir = os.path.join(script_dir, "models", "base")
    os.makedirs(output_dir, exist_ok=True)
    
    # The path for the output GGUF file.
    # We are converting to FP16, so we'll name it accordingly.
    output_path = os.path.join(output_dir, "tinyllama-1.1b-chat-v1.0.F16.gguf")

    print(f"Converting model from {model_dir} to GGUF...")

    # Convert the model to GGUF format
    # The `model_dir` should contain the pytorch_model.bin, config.json, and tokenizer files.
    # The `outfile_type` is set to 1 for FP16. 0 would be for FP32.
    convert_hf_to_gguf(
        model_dir,
        output_path,
        outfile_type=1 # 0 = fp32, 1 = fp16
    )

    print(f"Model converted to GGUF format at: {output_path}")

if __name__ == "__main__":
    convert_model_to_gguf()
