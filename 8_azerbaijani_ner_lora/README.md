# Azerbaijani Named Entity Recognition (NER) with LoRA

This project implements a Named Entity Recognition (NER) system for the Azerbaijani language, leveraging a LoRA (Low-Rank Adaptation) fine-tuned `distilbert-base-multilingual-cased` model. It includes data processing, model training, evaluation, and a FastAPI-based deployment for inference via a RESTful API. The goal is to identify and classify named entities (e.g., persons, organizations, locations) within Azerbaijani text.

## Features

*   **LoRA Fine-tuning:** Utilizes Low-Rank Adaptation (LoRA) for efficient fine-tuning of a `distilbert-base-multilingual-cased` model for Azerbaijani NER.
*   **Data Preprocessing:** Includes scripts for preparing raw Azerbaijani text data for NER training.
*   **Model Training & Evaluation:** Provides scripts to train the LoRA model and evaluate its performance on test datasets.
*   **Configurable Parameters:** Uses YAML configuration files (`model_config.yaml`, `training_config.yaml`, `inference_config.yaml`) for flexible project setup.
*   **FastAPI REST API:** Deploys the trained NER model as a high-performance, easy-to-use RESTful API for inference.
*   **Interactive API Documentation:** Automatically generates Swagger UI (`/docs`) and ReDoc (`/redoc`) for easy API exploration and testing.
*   **Deployment Ready:** Structured for deployment, including a `Dockerfile` for containerization.

## Installation

To get started with this project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/8_azerbaijani_ner_lora.git
    cd 8_azerbaijani_ner_lora
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    This project requires the following libraries. It's recommended to install them all for full functionality:

    ```bash
    pip install -r deployment/api/requirements.txt
    ```
    *(Note: This `requirements.txt` should cover most of the core dependencies for both the API and model interactions.)*

## Usage

The project can be used for training, evaluation, and inference via a RESTful API.

### 4.1. Running the FastAPI Application (Inference API)

To run the NER inference API:

1.  **Ensure you have installed the dependencies** (see [Installation](#installation)).
2.  **Navigate to the `deployment/api` directory:**
    ```bash
    cd deployment/api
    ```
3.  **Start the FastAPI server:**
    ```bash
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
    ```
    (Note: `--reload` is useful for development; remove it for production deployments.)

    The API will be accessible at `http://127.0.0.1:8000`.

### 4.2. Interacting with the API

*   **API Documentation:** Once the server is running, open your web browser and go to `http://127.0.0.1:8000/docs` for the interactive Swagger UI documentation, or `http://127.0.0.1:8000/redoc` for ReDoc documentation.
*   **Root Endpoint:** Access `http://127.0.0.1:8000` for a welcome message.
*   **`POST /predict` Endpoint:**
    *   **Method:** `POST`
    *   **URL:** `http://127.0.0.1:8000/predict`
    *   **Headers:** `Content-Type: application/json`
    *   **Request Body Example:**
        ```json
        {
          "text": "Tofiq Bəhramov adına Respublika Stadionu Bakı şəhərində yerləşir."
        }
        ```
    *   **Python Example:**
        ```python
        import requests
        import json

        url = "http://127.0.0.1:8000/predict"
        headers = {"Content-Type": "application/json"}
        data = {"text": "Tofiq Bəhramov adına Respublika Stadionu Bakı şəhərində yerləşir."}

        response = requests.post(url, headers=headers, data=json.dumps(data))
        print(response.json())
        ```

### 4.3. Training the Model

To train the NER model, you can use the provided training script:

1.  **Ensure you are in the project root directory.**
2.  **Run the training script:**
    ```bash
    python scripts/train.py
    ```
    (Note: Training parameters are configured in `configs/training_config.yaml`.)

### 4.4. Evaluating the Model

To evaluate the trained model:

1.  **Ensure you are in the project root directory.**
2.  **Run the evaluation script:**
    ```bash
    python scripts/eval.py
    ```
    (Note: Evaluation parameters are configured in `configs/inference_config.yaml` or `configs/model_config.yaml` depending on what `eval.py` expects.)

### 4.5. Making Predictions (Command-Line/Script)

To make predictions using a trained model outside of the API:

1.  **Ensure you are in the project root directory.**
2.  **Run the prediction script:**
    ```bash
    python scripts/predict.py --text "Some Azerbaijani text here."
    ```
    (Note: Check `scripts/predict.py` for exact arguments.)

## Project Structure

The project is organized into the following main directories:

```
8_azerbaijani_ner_lora/
├── main.py                     # Main entry point for various operations (e.g., orchestrator)
├── README.md                   # This README file
├── configs/                    # Configuration files for model, training, and inference
│   ├── inference_config.yaml
│   ├── model_config.yaml
│   └── training_config.yaml
├── data/                       # Stores raw, processed, and split datasets
│   ├── processed/
│   ├── raw/
│   │   ├── az_ner_dataset_test.csv
│   │   ├── az_ner_dataset_train.csv
│   │   └── az_ner_dataset_val.csv
│   └── splits/
├── deployment/                 # Contains API and deployment-related files
│   ├── api/                    # FastAPI application for inference
│   │   ├── app.py              # Main FastAPI application
│   │   ├── requirements.txt    # Python dependencies for the API
│   │   └── model_artifacts/    # Stored model weights, tokenizer, adapter config
│   └── docker/                 # Dockerfile for containerizing the API
│       └── Dockerfile
├── experiments/                # Stores results and checkpoints from training runs
│   └── run_001/
│       ├── checkpoints/
│       │   └── best_model/
│       └── logs/
├── notebooks/                  # Jupyter notebooks for EDA, experimentation, etc.
│   └── eda.ipynb
├── scripts/                    # Standalone scripts for various tasks
│   ├── eval.py                 # Script for model evaluation
│   ├── predict.py              # Script for command-line prediction
│   └── train.py                # Script for model training
└── src/                        # Source code for modular components
    ├── data/                   # Data loading and preprocessing utilities
    │   ├── dataset_loader.py
    │   └── preprocessing.py
    ├── evaluation/             # Metrics and error analysis for model evaluation
    │   ├── error_analysis.py
    │   └── metrics.py
    ├── inference/              # Prediction logic for the NER model
    │   └── predict.py
    ├── model/                  # Model definition, building, and training logic
    │   ├── model_builder.py
    │   └── trainer.py
    └── utils/                  # Utility functions (helpers, logging, config loading)
        ├── helpers.py
        └── logging.py
```

## Configuration

The project uses YAML files located in the `configs/` directory to manage various parameters for the model, training, and inference. This allows for flexible configuration without modifying the code directly.

*   **`configs/model_config.yaml`:** Defines model-specific parameters such as the base model name, the number of labels for NER, and the list of NER tags.
    ```yaml
    # Example content from model_config.yaml
    model_name: "distilbert-base-multilingual-cased"
    num_labels: 51
    label_list:
      - "O"
      - "B-PER"
      # ... other labels
    ```

*   **`configs/training_config.yaml`:** Contains parameters related to the training process, such as learning rate, batch size, number of epochs, etc. (You would need to define these based on your `train.py` script's expectations).

*   **`configs/inference_config.yaml`:** (If used by `predict.py` or `eval.py` for inference-specific settings, like output format or device.)

**How to Modify:**
To change any configuration, simply open the respective `.yaml` file in a text editor and adjust the values. These changes will be picked up by the scripts and API when they are run.

## API Endpoints

The FastAPI application exposes the following endpoints:

#### `GET /`

*   **Description:** Provides a welcome message and directs users to the API documentation.
*   **Method:** `GET`
*   **URL:** `http://127.0.0.1:8000/`
*   **Response (JSON):**
    ```json
    {
      "message": "Welcome to the Azerbaijani NER LoRA API. Please refer to /docs for API documentation."
    }
    ```

#### `GET /docs`

*   **Description:** Serves the interactive Swagger UI documentation for the API.
*   **Method:** `GET`
*   **URL:** `http://127.0.0.1:8000/docs`
*   **Response:** HTML content displaying the Swagger UI.

#### `GET /redoc`

*   **Description:** Serves the interactive ReDoc documentation for the API.
*   **Method:** `GET`
*   **URL:** `http://127.0.0.1:8000/redoc`
*   **Response:** HTML content displaying the ReDoc UI.

#### `POST /predict`

*   **Description:** Performs Named Entity Recognition on the provided Azerbaijani text.
*   **Method:** `POST`
*   **URL:** `http://127.0.0.1:8000/predict`
*   **Request Body (JSON):**
    ```json
    {
      "text": "Tofiq Bəhramov adına Respublika Stadionu Bakı şəhərində yerləşir."
    }
    ```
    *   **`text` (string, required):** The Azerbaijani text on which to perform NER.

*   **Response (JSON):**
    ```json
    {
      "text": "Tofiq Bəhramov adına Respublika Stadionu Bakı şəhərində yerləşir.",
      "entities": [
        {
          "entity": "B-PER",
          "text": "Tofiq Bəhramov"
        },
        {
          "entity": "B-ORG",
          "text": "Respublika"
        },
        {
          "entity": "I-PRO",
          "text": "Stadionu"
        },
        {
          "entity": "I-LOC",
          "text": "Bakı"
        }
      ]
    }
    ```
    *   **`text` (string):** The original input text.
    *   **`entities` (array of objects):** A list of identified named entities.
        *   **`entity` (string):** The type of named entity (e.g., "B-PER", "I-LOC").
        *   **`text` (string):** The extracted text span corresponding to the entity.

## Model Details

This project utilizes a Named Entity Recognition (NER) model fine-tuned for the Azerbaijani language.

*   **Base Model:** The foundation of our NER system is `distilbert-base-multilingual-cased` from Hugging Face Transformers. This model is chosen for its multilingual capabilities and efficiency, providing a good balance between performance and computational cost.
*   **LoRA (Low-Rank Adaptation):** To efficiently adapt the base model to the Azerbaijani NER task, we employ LoRA. LoRA significantly reduces the number of trainable parameters during fine-tuning by injecting low-rank matrices into the transformer layers. This approach allows for faster training, reduced memory usage, and easier storage and deployment of fine-tuned adapters without modifying the large base model.
*   **NER Task:** The model is trained to identify and classify specific categories of entities in Azerbaijani text, such as persons (PER), organizations (ORG), locations (LOC), products (PRO), and others, as defined in the `label_list` within `configs/model_config.yaml`. The output uses the IOB2 (Inside, Outside, Beginning) tagging scheme.

## Data

The project uses a custom dataset for Azerbaijani Named Entity Recognition.

*   **Dataset Location:** Raw datasets are expected to be placed in the `data/raw/` directory. The example provided includes:
    *   `az_ner_dataset_train.csv`
    *   `az_ner_dataset_val.csv`
    *   `az_ner_dataset_test.csv`
*   **Format:** The datasets are expected to be in a format suitable for NER tasks, likely tokenized text and corresponding NER tags. (Further details on expected format can be added here if known, e.g., "CoNLL-2003 format" or "CSV with 'tokens' and 'ner_tags' columns").
*   **Preprocessing:** The `src/data/preprocessing.py` script is responsible for converting raw data into a format usable by the model, including tokenization and alignment of labels. The `src/data/dataset_loader.py` handles loading these datasets.
*   **Processed Data:** Processed and split datasets are stored in `data/processed/` and `data/splits/` respectively.
