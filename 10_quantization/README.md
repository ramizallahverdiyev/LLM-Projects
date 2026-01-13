# Project 11 – LLM Quantization and Deployment with GGUF and Ollama

## Project Goal
This project aims to demonstrate a complete, local-first workflow for optimizing and deploying Large Language Models (LLMs) through quantization and serving them locally using the GGUF format and Ollama.

## What this Project Demonstrates
- Conversion of a LLaMA-compatible LLM to the GGUF format.
- Post-Training Quantization (PTQ) of the model to various bit depths (e.g., Q8_0, Q4_K_M).
- Local serving of these quantized GGUF models using Ollama.
- Benchmarking of memory usage, inference speed (tokens/sec), and startup time.
- Analysis of memory, speed, and quality trade-offs associated with different quantization levels.

## What this Project Does NOT Cover
- Training or fine-tuning of LLMs.
- Usage of cloud-based APIs or services for LLM deployment.
- GPU-specific optimizations (focus is primarily CPU-first, with optional GPU consideration).

## Folder Structure Explanation
- `README.md`: Overview of the project.
- `requirements.txt`: Python dependencies for the project.
- `models/`: Stores the base FP16 models and their quantized GGUF versions.
    - `base/`: Original FP16 models.
    - `quantized/`: Quantized GGUF models.
- `scripts/`: Python scripts for model download, conversion, quantization, and benchmarking.
    - `download_model.py`: Script to download a pre-trained LLaMA-compatible model.
    - `convert_to_gguf.py`: Script to convert a Hugging Face model to GGUF format.
    - `quantize_model.py`: Script to quantize a GGUF model to different bit depths.
    - `benchmark.py`: Script to measure performance metrics of served models.
- `ollama/`: Configuration and instructions for Ollama.
    - `Modelfile`: Ollama Modelfile for custom model definitions.
    - `run_ollama.md`: Instructions for running Ollama with the quantized models.
- `benchmarks/`: Stores the results of the benchmarking process.
    - `latency.txt`: Latency measurements.
    - `memory.txt`: Memory usage measurements.
    - `quality_notes.md`: Qualitative observations on model output at different quantization levels.
- `docs/`: Comprehensive documentation and theoretical explanations.
    - `theory.md`: General theory of LLM quantization.
    - `gguf_explained.md`: Detailed explanation of the GGUF format.
    - `quantization_tradeoffs.md`: Discussion of trade-offs between model size, speed, and quality.

## Roadmap
- Step 1: Project Setup (Completed: Directory structure and README outline)
- Step 2: Add theory documentation (docs/)
- Step 3: Implement model download script
- Step 4: Implement FP16 → GGUF conversion script
- Step 5: Implement quantization script (Q8_0, Q4_K_M)
- Step 6: Integrate with Ollama runtime (Modelfile and run instructions)
- Step 7: Implement benchmarking scripts and record results
- Step 8: Finalize README and project documentation
