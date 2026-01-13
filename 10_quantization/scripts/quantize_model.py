# scripts/quantize_model.py

import os
from llama_cpp import llama_quantize

def quantize_model():
    """
    Quantizes a GGUF model to various formats.
    """
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_model_dir = os.path.join(script_dir, "models", "base")
    quantized_model_dir = os.path.join(script_dir, "models", "quantized")
    os.makedirs(quantized_model_dir, exist_ok=True)

    input_gguf_path = os.path.join(base_model_dir, "tinyllama-1.1b-chat-v1.0.F16.gguf")

    if not os.path.exists(input_gguf_path):
        print(f"Error: Input GGUF model not found at {input_gguf_path}")
        print("Please ensure you have run the download_model.py and convert_to_gguf.py scripts first.")
        return

    # Define the quantization methods to apply
    # Q8_0: 8-bit integer quantization (no grouping)
    # Q4_K_M: 4-bit quantization using QK_K method (k-group size)
    quantization_methods = ["Q8_0", "Q4_K_M"]

    for method in quantization_methods:
        output_gguf_path = os.path.join(quantized_model_dir, f"tinyllama-1.1b-chat-v1.0.{method}.gguf")
        print(f"Quantizing {input_gguf_path} to {method} format...")
        print(f"Output will be saved to: {output_gguf_path}")

        try:
            llama_quantize(
                input_gguf_path,
                output_gguf_path,
                qtype=method
            )
            print(f"Quantization to {method} complete.")
        except Exception as e:
            print(f"Error quantizing to {method}: {e}")

if __name__ == "__main__":
    quantize_model()
