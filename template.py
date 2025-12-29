import os

# TEMPLATE FOR A THIRD PROJECT STRUCTURE
# PROJECT_NAME = "3.Embedding Similarity Search"

# STRUCTURE = {
#     "data": [
#         "raw_texts.py",
#         "queries.py",
#     ],
#     "embeddings": [
#         "embedder.py",
#         "embedding_store.py",
#     ],
#     "similarity": [
#         "metrics.py",
#         "search.py",
#     ],
#     "evaluation": [
#         "manual_checks.md",
#     ],
#     "config": [
#         "settings.py",
#     ],
#     "": [
#         "main.py",
#         "requirements.txt",
#         "README.md",
#     ]
# }



# TEMPLATE FOR FOURTH PROJECT STRUCTURE
# PROJECT_NAME = "4. FAISS Vector Search"

# STRUCTURE = {
#     "data/raw": [
#         "sentences.txt",
#     ],
#     "data/processed": [
#         "embeddings.npy",
#     ],
#     "src": [
#         "__init__.py",
#         "data_loader.py",
#         "embedding.py",
#         "indexer.py",
#         "search.py",
#         "utils.py",
#     ],
#     "experiments": [
#         "manual_similarity_check.py",
#     ],
#     "notes": [
#         "observations.md",
#     ],
#     "": [
#         "main.py",
#         "README.md",
#     ],
# }


# TEMPLATE FOR FIFTH PROJECT STRUCTURE
# PROJECT_NAME = "5.IMDB Sentiment Analysis"
# STRUCTURE = {
#     "": [
#         "README.md",
#         "requirements.txt",
#         ".gitignore",
#     ],
#     "data/raw": [
#         "imdb_hf_cache/",
#     ],
#     "data/processed": [],
#     "notebooks": [
#         "01_data_exploration.ipynb",
#         "02_error_analysis.ipynb",
#     ],
#     "src": [
#         "__init__.py",
#         "data_loader.py",
#         "preprocessing.py",
#         "evaluation.py",
#         "inference.py",
#         "baseline/__init__.py",
#         "baseline/train_baseline.py",
#         "baseline/evaluate_baseline.py",
#         "transformer/__init__.py",
#         "transformer/train_transformer.py",
#         "transformer/evaluate_transformer.py",
#     ],
#     "models": [
#         "baseline/vectorizer.pkl",
#         "baseline/classifier.pkl",
#         "transformer/imdb_sentiment_model/",
#     ],
#     "scripts": [
#         "run_baseline.py",
#         "run_transformer.py",
#     ],
# }

# TEMPLATE FOR SIXTH PROJECT STRUCTURE
# PROJECT_NAME = "6_query_rewriting_reranking"

# STRUCTURE = {
#     "data": [
#         "queries.json",
#         "documents.json",
#     ],

#     "src": [
#         "__init__.py",
#     ],

#     "src/embeddings": [
#         "__init__.py",
#         "build_index.py",
#     ],
#     "src/embeddings/index_files": [
#         # FAISS index files
#     ],

#     "src/query_rewriting": [
#         "__init__.py",
#         "hyde_rewriter.py",
#     ],

#     "src/reranking": [
#         "__init__.py",
#         "bge_reranker.py",
#     ],

#     "src/context_selection": [
#         "__init__.py",
#         "select_context.py",
#     ],

#     "src/utils": [
#         "__init__.py",
#         "preprocessing.py",
#         "metrics.py",
#     ],

#     "tests": [
#         "test_pipeline.py",
#     ],

#     "": [
#         "main.py",
#         "requirements.txt",
#         "README.md",
#     ],
# }

# TEMPLATE FOR SEVENTH PROJECT STRUCTURE
# PROJECT_NAME = "7_calculator_agent"

# STRUCTURE = {
#     "": [
#         "README.md",      # Project overview and documentation
#         "main.py",        # Main entry point for the agent
#     ],
#     "src": [
#         "agent.py",           # LLM controller
#         "intent_detector.py", # Decide if tool needed
#         "interpreter.py",     # Tool output → human-readable
        
#     ],
    
#     "src/tools": [
#         "calculator.py",     # Calculator tool implementation
#     ],
#     "prompts": [
#         "intent_prompt.txt",      # Prompt for intent detection
#         "interpret_prompt.txt",   # Prompt for interpreting tool output
#     ],
#     "tests": [
#         "test_calculator.py",     # Tests for the calculator tool
#         "test_intent.py",         # Tests for the intent detection
#         "test_agent.py",          # Tests for the agent
#     ],
# }


PROJECT_NAME = "8_azerbaijani_ner_lora"

# STRUCTURE TEMPLATE
STRUCTURE = {
    "": [
        "README.md",             # Project overview and documentation
        "requirements.txt",      # Required libraries
    ],
    "configs": [
        "model_config.yaml",     # Base model + LoRA params
        "training_config.yaml",  # Batch size, optimizer, epochs
        "inference_config.yaml", # Inference params
    ],
    "data/raw": [
        "az_ner_dataset.csv",    # Raw CSV dataset
    ],
    "data/processed": [],
    "data/splits": [],
    "notebooks": [
        "eda.ipynb",             # Exploratory data analysis
    ],
    "src": [
        "__init__.py",
    ],
    "src/data": [
        "__init__.py",
        "dataset_loader.py",     # HuggingFace dataset loader
        "preprocessing.py",      # Tokenization & label alignment
    ],
    "src/model": [
        "__init__.py",
        "model_builder.py",      # Base model + LoRA adapter
        "trainer.py",            # Training loop + evaluation
    ],
    "src/evaluation": [
        "__init__.py",
        "metrics.py",            # Precision, recall, F1
        "error_analysis.py",     # FP/FN/boundary/type errors
    ],
    "src/inference": [
        "__init__.py",
        "predict.py",            # Inference + reconstruction
    ],
    "src/utils": [
        "__init__.py",
        "logging.py",
        "helpers.py",
    ],
    "experiments/run_001/checkpoints": [],
    "experiments/run_001/logs": [],
    "scripts": [
        "train.py",
        "eval.py",
        "predict.py",
    ],
    "deployment/api": [
        "app.py",
        "requirements.txt",
    ],
    "deployment/docker": [
        "Dockerfile",
    ],
}

def create_structure(base_path: str):
    for folder, files in STRUCTURE.items():
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)

        for item in files:
            item_path = os.path.join(folder_path, item)

            if item.endswith("/"):
                os.makedirs(item_path, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(item_path), exist_ok=True)

                if not os.path.exists(item_path):
                    with open(item_path, "w", encoding="utf-8"):
                        pass


if __name__ == "__main__":
    base_dir = os.path.join(os.getcwd(), PROJECT_NAME)
    os.makedirs(base_dir, exist_ok=True)
    create_structure(base_dir)
    print(f"Project structure created at: {base_dir}")