# Embedding Similarity Search

This project demonstrates a simple, standalone implementation of
semantic similarity search using text embeddings and cosine similarity.

The goal is to understand how textual meaning can be represented as
vectors and compared mathematically, without using vector databases,
LLMs, or retrieval frameworks.

---

## Project Scope

This project focuses only on:

* Converting text into vector embeddings
* Measuring semantic similarity between texts
* Ranking texts based on similarity to a query

Out of scope (intentionally):

* PDF processing
* Text chunking
* Large Language Models (LLMs)
* Vector databases (FAISS, Pinecone, etc.)
* Retrieval-augmented generation (RAG)

---

## How It Works

1. A fixed set of reference texts is defined.
2. Each reference text is converted into an embedding vector.
3. A query text is embedded using the same model.
4. Cosine similarity is computed between the query embedding and all stored embeddings.
5. Results are ranked and the top-k most similar texts are returned.

---

## Project Structure

```
embedding_similarity_search/
│
├── data/
│   ├── raw_texts.py          # Reference texts
│   └── queries.py            # Query examples
│
├── embeddings/
│   ├── embedder.py           # Embedding generation
│   └── embedding_store.py   # In-memory embedding storage
│
├── similarity/
│   ├── metrics.py            # Cosine similarity implementation
│   └── search.py             # Semantic search logic
│
├── evaluation/
│   └── manual_checks.md      # Qualitative evaluation notes
│
├── config/
│   └── settings.py           # Central configuration
│
├── main.py                   # Orchestration script
├── requirements.txt
└── README.md
```

---

## Installation

Create and activate a virtual environment (recommended), then install dependencies:

```bash
pip install -r requirements.txt
```

Required packages:

* numpy
* sentence-transformers

---

## Running the Project

Make sure you are in the project root directory (where `main.py` is located), then run:

```bash
python main.py
```

---

## Sample Output

Below is an example output produced by running the project:

```
Loaded 25 reference texts.

============================================================
QUERY: How do machines learn patterns from data?

1. [0.7509] Machine learning algorithms learn patterns from historical data.
2. [0.5473] Unsupervised learning discovers hidden structures in data.
3. [0.4672] Supervised learning requires labeled training data.

============================================================
QUERY: How do banks evaluate credit risk?

1. [0.7787] Banks use credit scoring models to assess customer risk.
2. [0.4716] Loan approval decisions depend on income, credit history, and debt.
3. [0.4679] Risk management is a critical function in financial institutions.

============================================================
QUERY: What does cosine similarity measure?

1. [0.7767] Cosine similarity computes similarity based on vector orientation.
2. [0.5065] Semantic similarity measures meaning rather than exact wording.
3. [0.1936] Information retrieval systems search relevant content.
```

Similarity scores may vary slightly depending on runtime environment.

---

## Notes

* All embeddings are generated in-memory.
* No persistence layer is used.
* Cosine similarity is implemented manually for clarity.
* The project is designed for learning and experimentation, not production use.

---

## Evaluation

Qualitative observations and behavior analysis can be found in:

```
evaluation/manual_checks.md
```

---

## License

This project is provided for educational purposes.
