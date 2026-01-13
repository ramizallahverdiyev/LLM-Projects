# Project 10: Local LLM Quantization with GGUF and Ollama

This project provides a complete, step-by-step workflow for taking a standard FP16 Large Language Model, quantizing it using the GGUF format, and serving it locally for inference with Ollama. It is designed for engineers and data scientists looking to understand and implement model optimization techniques for local or on-device deployment.

## Project Goal
The primary goal is to make LLMs practical to run on consumer-grade hardware (CPU-first). We achieve this by converting a base model into the highly efficient GGUF format and applying post-training quantization to significantly reduce its memory footprint and accelerate inference speed.

## Key Features
- **Model Conversion**: Convert a Hugging Face model to the GGUF format.
- **Quantization**: Apply various quantization levels (e.g., Q4_K_M, Q8_0) to the GGUF model.
- **Local Deployment**: Serve the quantized model using Ollama.
- **Benchmarking**: Measure and compare the performance (latency, throughput, memory usage) of different quantization levels.
- **Comprehensive Documentation**: Includes theoretical explanations of quantization, GGUF, and the trade-offs involved.

## What this Project Does NOT Cover
- Model training or fine-tuning.
- Cloud-based deployment or APIs.
- Advanced, GPU-specific inference optimizations.

---

## How to Use This Project

### 1. Prerequisites
- Python 3.8+
- [Ollama](https://ollama.com/) installed and running.
- The required Python packages, which can be installed via:
  ```bash
  pip install -r requirements.txt
  ```

### 2. The Workflow

This project follows a 4-step process. The scripts in the `scripts/` directory are numbered to guide you.

**Step 1: Download a Base Model**
First, download a LLaMA-compatible model from the Hugging Face Hub. The `download_model.py` script is configured to download a small, manageable model for demonstration purposes.
```bash
python scripts/download_model.py
```
This will save the model to the `models/base/` directory.

**Step 2: Convert to GGUF**
Next, convert the downloaded FP16 model to the GGUF format. The `convert_to_gguf.py` script handles this conversion.
```bash
python scripts/convert_to_gguf.py
```
This creates a full-precision (FP16) GGUF file in `models/quantized/`.

**Step 3: Quantize the GGUF Model**
Now, apply quantization to the FP16 GGUF model. The `quantize_model.py` script can generate multiple quantized versions (e.g., 8-bit and 4-bit).
```bash
python scripts/quantize_model.py
```
This saves the quantized models (e.g., `Q4_K_M-model.gguf`) to the `models/quantized/` directory.

**Step 4: Serve with Ollama**
With your quantized models ready, use the provided `Modelfile` in the `ollama/` directory to create a local Ollama model. Follow the instructions in `ollama/run_ollama.md` to create and run your custom model.
```bash
# Following instructions in ollama/run_ollama.md
ollama create -f ollama/Modelfile my-custom-model:q4
ollama run my-custom-model:q4
```

**Step 5: Benchmark Performance**
Finally, use the `benchmark.py` script to measure the performance of your running Ollama model.
```bash
python scripts/benchmark.py --model my-custom-model:q4
```
The results for latency and memory usage will be saved to the `benchmarks/` directory.

---

## Project Structure
- `README.md`: This file, providing a complete overview of the project.
- `requirements.txt`: Python dependencies.
- `models/`: Contains the LLMs.
    - `base/`: Original, high-precision models from Hugging Face.
    - `quantized/`: Converted and quantized GGUF models.
- `scripts/`: A collection of Python scripts that form the core workflow.
    - `download_model.py`: Downloads the base model.
    - `convert_to_gguf.py`: Converts the base model to GGUF FP16.
    - `quantize_model.py`: Quantizes the GGUF model.
    - `benchmark.py`: Measures the performance of a model served via Ollama.
- `ollama/`: Configuration for serving models with Ollama.
    - `Modelfile`: Defines how to package a GGUF file into an Ollama model.
    - `run_ollama.md`: Step-by-step instructions for using Ollama.
- `benchmarks/`: Stores performance measurement results.
    - `latency.txt`: Inference speed and throughput metrics.
    - `memory.txt`: Peak memory usage.
    - `quality_notes.md`: Subjective notes on model quality.
- `docs/`: In-depth documentation.
    - `theory.md`: Explains the theory behind model quantization.
    - `gguf_explained.md`: A deep dive into the GGUF file format.
    - `quantization_tradeoffs.md`: Discusses the trade-offs between size, speed, and quality.