# Query Rewriting + Reranking Project

This project demonstrates a **retrieval-augmented pipeline** for query rewriting and document reranking. It leverages HyDE for synthetic query generation, FAISS for efficient vector search, and a reranker model for fine-grained ranking.

## Features

* **HyDE-based Query Rewriting**: Expand short user queries into richer synthetic queries.
* **FAISS Vector Indexing**: Build an IVF index for efficient document retrieval.
* **Reranking**: Rank retrieved documents with semantic similarity scoring.
* **Context Selection**: Select final contexts for downstream applications.

## Folder Structure

```
6_query_rewriting_reranking/
│   index_ivf.faiss
│   main.py
│   README.md
│
├── data/
│   ├── documents.json
│   └── queries.json
│
├── src/
│   ├── context_selection/
│   │   ├── select_context.py
│   │   └── __init__.py
│   │
│   ├── embeddings/
│   │   ├── build_index.py
│   │   ├── index_files/
│   │   └── __init__.py
│   │
│   ├── query_rewriting/
│   │   ├── hyde_rewriter.py
│   │   └── __init__.py
│   │
│   ├── reranking/
│   │   ├── bge_reranker.py
│   │   └── __init__.py
│   │
│   ├── utils/
│   │   ├── metrics.py
│   │   ├── preprocessing.py
│   │   └── __init__.py
│   │
│   └── __init__.py
│
└── tests/
    ├── test_pipeline.py
    └── __pycache__/
```

## Example Output

**User Query:** `Top universities in Azerbaijan`

**Rewritten Query:** `The top universities in Azerbaijan offer a variety of programs in science, engineering, and business.`

**Top-k Retrieved Documents (Before Reranking):**

* Azerbaijan Medical University
* Azerbaijan Technical University (AzTU)
* Baku State University (BDU)
* Azerbaijan State University of Economics (UNEC)
* Khazar University

**Top-k Documents After Reranking:**

* Baku State University (BDU) (score: 6.5311)
* Azerbaijan Medical University (score: 4.7849)
* Azerbaijan State University of Economics (UNEC) (score: 4.1333)
* Azerbaijan Technical University (AzTU) (score: 3.9776)
* Khazar University (score: 3.0490)

**Final Selected Contexts:**

* **Baku State University (BDU):** Baku State University (BDU) is the oldest and most internationally recognized university in Azerbaijan.
* **Azerbaijan Medical University:** Azerbaijan Medical University is the leading medical institution in Azerbaijan.
* **Azerbaijan State University of Economics (UNEC):** Azerbaijan State University of Economics (UNEC) specializes in economics, finance, and business education.

## Usage

1. Activate virtual environment:

```bash
.venv\Scripts\activate
```

2. Run the pipeline:

```bash
python -m main
```

3. Enter your query when prompted.

## Requirements

* Python >= 3.10
* torch
* transformers
* sentence-transformers
* faiss

This README is structured to reflect the project workflow, folder organization, and example outputs.
