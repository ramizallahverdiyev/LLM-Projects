# The GGUF File Format

GGUF stands for **Georgi Gerganov Universal Format**. It is a file format for storing large language models for inference, designed to be used with the `llama.cpp` library and its ecosystem.

## Key Goals and Features

GGUF was created to replace the previous GGML format with a more extensible and feature-rich alternative. Its key features include:

1.  **Extensibility:** GGUF is designed to be easily extended with new information without breaking compatibility with older clients. It uses a key-value structure for metadata, allowing new fields to be added in the future.

2.  **Metadata:** It can store arbitrary metadata about the model, such as the model architecture, tokenizer information, special tokens, and quantization details. This makes the model file self-contained and easy to use.

3.  **Simplicity:** The format is designed to be simple to read and write, with a focus on making it easy for developers to implement support for GGUF in their own tools.

4.  **Mmap-able:** The tensor data in a GGUF file is aligned in a way that allows it to be memory-mapped (`mmap`). This means the operating system can load parts of the model directly from disk into memory as they are needed, without requiring the entire model to be read into RAM first. This is crucial for running large models on machines with limited RAM.

## Why GGUF for this Project?

Using GGUF allows us to:

-   Run state-of-the-art LLMs on standard consumer hardware (CPUs).
-   Leverage the highly optimized `llama.cpp` engine for fast inference.
-   Easily experiment with different quantization levels to find the right balance between performance and quality.
-   Package the quantized model into a single, portable file that can be used by various tools, including Ollama.
