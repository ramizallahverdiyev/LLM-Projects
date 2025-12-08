# Mini RAG

A minimal end-to-end Retrieval-Augmented Generation (RAG) example. It reads a document (PDF/DOCX/TXT), chunks it, creates sentence-transformer embeddings, stores them in a FAISS index, retrieves the most relevant chunks for a query, and streams an answer from a local LLM endpoint.

## Project Layout
- `main.py` orchestrates the full pipeline on `data/raw/pdf-sample.pdf`.
- `src/reader.py` loads PDF, DOCX, or TXT files.
- `src/chunker.py` splits text into overlapping chunks (defaults: 500 words, 100 overlap).
- `src/embedder.py` wraps a `sentence-transformers` model (MiniLM by default).
- `src/vectordb.py` manages FAISS index + metadata files under `embeddings/`.
- `src/retriever.py` performs k-NN search to build the context.
- `src/answer.py` calls a local LLM endpoint (default: Ollama on `http://localhost:11434` with `llama3.2`).

## Prerequisites
- Python 3.9+ (tested with modern Python versions).
- Local LLM server reachable at `http://localhost:11434/api/generate` (e.g., `ollama run llama3.2` to download and serve the model).

## Setup
```bash
# from the repo root
python -m venv .venv
.venv\Scripts\activate       # PowerShell on Windows
pip install -r requirements.txt
```

## Usage
```bash
cd "2. Mini RAG"
python main.py
```
- The script reads `data/raw/pdf-sample.pdf`, builds/updates `embeddings/faiss.index` and `embeddings/metadata.pkl`, retrieves top chunks for the hardcoded query, and prints the final answer.
- To try your own document, place it in `data/raw/` and update `doc_path` in `main.py`.
- To ask a different question, change the `query` string in `main.py`.

## Customization Tips
- Change chunk sizing/overlap in `src/chunker.py` for different retrieval granularity.
- Swap the embedding model by passing a different `model_name` to `Embedder` in `main.py`.
- Adjust number of retrieved chunks via the `k` argument in `src/retriever.py` (or when calling it).
- Point to another local/remote LLM by editing the URL or `model` in `src/answer.py`.
