# How to Run a Custom GGUF Model with Ollama

This guide provides the steps to serve your own quantized GGUF model using Ollama.

## Prerequisites

1.  **Ollama Installed**: Ensure Ollama is installed on your system. Visit the [Ollama website](https://ollama.com/) for installation instructions.
2.  **Quantized GGUF Model**: You must have a GGUF model file (e.g., `q4_k_m-model.gguf`) located in the `models/quantized/` directory.
3.  **Modelfile**: A `Modelfile` must be present in the `ollama/` directory, correctly configured to point to your GGUF file.

## Step 1: Copy the Model and Modelfile

Ollama needs the GGUF model file and the `Modelfile` to be in the same directory when creating a new model.

1.  Navigate to the `ollama/` directory in your terminal.
2.  Copy your GGUF model from `../models/quantized/` into the current (`ollama/`) directory. For example:

    ```bash
    # Make sure you are in the 'ollama' directory
    cp ../models/quantized/Q4_K_M-model.gguf ./
    ```

## Step 2: Update the `Modelfile`

Open the `Modelfile` in the `ollama/` directory and ensure the `FROM` instruction points to the correct GGUF file name you just copied.

**Example `Modelfile`:**

```modelfile
# Point to the GGUF file you copied into this directory
FROM ./Q4_K_M-model.gguf

# Parameters and template...
PARAMETER stop "</s>"
TEMPLATE """
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
{{ .Prompt }}
<|im_end|>
<|im_start|>assistant
"""
```

## Step 3: Create the Ollama Model

Use the `ollama create` command to build the model from your `Modelfile`. This packages your GGUF file into an Ollama-managed model.

-   `-f Modelfile`: Specifies the Modelfile to use.
-   `<your-model-name>`: The custom name you want to give your model (e.g., `my-custom-model:q4`).

```bash
ollama create -f Modelfile my-custom-model:q4
```

Ollama will process the file and create the model. On success, you will see a "success" message.

## Step 4: Run the Model

Once created, you can run your model just like any other Ollama model using `ollama run`.

```bash
ollama run my-custom-model:q4
```

You can now interact with your custom quantized model directly in the terminal.

## Step 5: (Optional) Clean Up

You can remove the GGUF file you copied into the `ollama/` directory to save space. Ollama now manages its own copy.

```bash
rm ./Q4_K_M-model.gguf
```