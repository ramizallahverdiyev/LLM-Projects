# Running Quantized Models with Ollama

This guide explains how to integrate and run the quantized GGUF models from this project using Ollama. Ollama provides a simple and powerful way to run large language models locally.

## 1. Install Ollama

If you don't have Ollama installed, download and install it from the official website:
[https://ollama.com/download](https://ollama.com/download)

Follow the instructions for your operating system.

## 2. Prepare Your Quantized Model

Before proceeding, ensure you have:
- Downloaded the base model using `python scripts/download_model.py`
- Converted the model to GGUF format using `python scripts/convert_to_gguf.py`
- Quantized the model using `python scripts/quantize_model.py`

This will ensure you have the `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` file (or other quantized versions) in the `models/quantized/` directory.

## 3. Create a Custom Ollama Model

Ollama uses `Modelfile`s to define custom models. We have a `Modelfile` prepared for you in the `ollama/` directory.

Navigate to the project's root directory in your terminal and then into the `ollama` directory:

```bash
cd ollama
```

Now, create the Ollama model using the `Modelfile`:

```bash
ollama create tinyllama-quantized -f Modelfile
```

This command tells Ollama to create a new model named `tinyllama-quantized` using the configuration and GGUF file specified in the `Modelfile`.

## 4. Run the Custom Ollama Model

Once the model is created, you can run it directly:

```bash
ollama run tinyllama-quantized
```

You can then start interacting with the model in your terminal. For example:

```
>>> How are you?
```

To exit the model interaction, type `/bye` or press `Ctrl+D`.

## 5. Explore Other Quantized Versions

You can create different `Modelfile`s pointing to `tinyllama-1.1b-chat-v1.0.Q8_0.gguf` or other quantized versions to compare their performance and quality. Just update the `FROM` line in the `Modelfile` to point to the desired GGUF file, and then create a new Ollama model with a different name.

## 6. Remove the Custom Ollama Model

If you wish to remove the model from Ollama, use the `rm` command:

```bash
ollama rm tinyllama-quantized
```
