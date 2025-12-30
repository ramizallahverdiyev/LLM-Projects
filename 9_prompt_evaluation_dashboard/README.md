# Prompt Evaluation Dashboard

A web-based dashboard for evaluating the performance of large language models (LLMs) on question-answering tasks. This application allows users to select prompts from the SQuAD 2.0 dataset, generate answers using a local LLaMA model, and evaluate the quality of the generated answers.

## Features

*   **Dark Theme:** A modern and professional dark theme for a better user experience.
*   **Prompt Selection:** Select from a random sample of 500 prompts from the SQuAD 2.0 dataset.
*   **LLM Integration:**  Generate answers using a local LLaMA model (via Ollama).
*   **Evaluation:** Evaluate the generated answers against reference answers using a simple accuracy metric.
*   **Interactive Results Table:** View the evaluation results in a clear and interactive table.
*   **Loading Indicator:** A loading spinner provides feedback to the user while the model is generating an answer.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

*   Python 3.8+
*   [Ollama](https://ollama.ai/) installed and running with a LLaMA model (e.g., `llama2`).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/prompt-evaluation-dashboard.git
    cd prompt-evaluation-dashboard
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

Once the dependencies are installed, you can run the application with the following command:

```bash
python -m src.dashboard
```

The application will be available at [http://127.0.0.1:8050/](http://127.0.0.1:8050/).

## Usage

1.  **Select a prompt:** Choose a prompt from the dropdown menu.
2.  **Evaluate:** Click the "Evaluate" button to generate an answer from the LLM.
3.  **View results:** The evaluation results, including the prompt, model output, reference answers, and accuracy, will be displayed in the table below.

## Project Structure

```
.
├── data/
│   └── squad-train-v2.0.json   # SQuAD 2.0 dataset
├── models/
│   └── llama_local/            # (Placeholder for local model files)
├── notebooks/
│   └── exploration.ipynb       # Jupyter notebook for data exploration
├── prompts/
│   └── prompts.json            # (Not used in the current implementation)
├── src/
│   ├── __init__.py
│   ├── assets/
│   │   └── custom.css          # Custom CSS for styling
│   ├── dashboard.py            # Main application file
│   ├── evaluate.py             # Evaluation logic
│   ├── run_model.py            # Logic for running the LLM
│   └── storage.py              # Logic for saving and retrieving evaluations
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
