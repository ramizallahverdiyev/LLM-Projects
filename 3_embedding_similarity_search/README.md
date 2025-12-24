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
============================================================
QUERY: How do machines learn patterns from data?

1. [0.7509] Machine learning algorithms learn patterns from historical data.
2. [0.5473] Unsupervised learning discovers hidden structures in data.
3. [0.4672] Supervised learning requires labeled training data.

============================================================
QUERY: What are vector embeddings used for?

1. [0.6616] High-dimensional vectors are common in embedding spaces.
2. [0.5994] Vector embeddings capture the semantic meaning of text.
3. [0.3106] Cosine similarity computes similarity based on vector orientation.

============================================================
QUERY: Difference between supervised and unsupervised learning

1. [0.6039] Unsupervised learning discovers hidden structures in data.
2. [0.5595] Supervised learning requires labeled training data.
3. [0.3247] Machine learning algorithms learn patterns from historical data.

============================================================
QUERY: How do language models understand text meaning?

1. [0.5357] Natural language processing focuses on text and language understanding.
2. [0.5323] Large language models are trained on massive text corpora.
3. [0.4999] Vector embeddings capture the semantic meaning of text.

============================================================
QUERY: How do banks evaluate credit risk?

1. [0.7787] Banks use credit scoring models to assess customer risk.
2. [0.4716] Loan approval decisions depend on income, credit history, and debt.
3. [0.4679] Risk management is a critical function in financial institutions.

============================================================
QUERY: What methods are used for fraud detection?

1. [0.6441] Fraud detection systems monitor abnormal transaction behavior.
2. [0.3444] Banks use credit scoring models to assess customer risk.
3. [0.2915] Machine learning algorithms learn patterns from historical data.

============================================================
QUERY: How is loan default risk predicted?

1. [0.6747] Financial models help predict default probability.
2. [0.5291] Banks use credit scoring models to assess customer risk.
3. [0.5013] Loan approval decisions depend on income, credit history, and debt.

============================================================
QUERY: How is AI applied in medical diagnosis?

1. [0.7891] Artificial intelligence is widely used in medical diagnosis.
2. [0.5381] Medical imaging techniques assist doctors in disease detection.
3. [0.4009] Machine learning models can predict patient health risks.

============================================================
QUERY: Can machine learning predict patient risks?

1. [0.8655] Machine learning models can predict patient health risks.
2. [0.4276] Artificial intelligence is widely used in medical diagnosis.
3. [0.3993] Banks use credit scoring models to assess customer risk.

============================================================
QUERY: What does cosine similarity measure?

1. [0.7767] Cosine similarity computes similarity based on vector orientation.

1. [0.7767] Cosine similarity computes similarity based on vector orientation.
2. [0.5065] Semantic similarity measures meaning rather than exact wording.
3. [0.1936] Information retrieval systems search relevant content.

============================================================
QUERY: Why is Python popular in data science?

1. [0.8466] Python is a popular programming language for data science.
2. [0.3203] Data pipelines automate data processing workflows.
3. [0.2974] Artificial intelligence is widely used in medical diagnosis.


1. [0.7767] Cosine similarity computes similarity based on vector orientation.
2. [0.5065] Semantic similarity measures meaning rather than exact wording.
3. [0.1936] Information retrieval systems search relevant content.

============================================================
QUERY: Why is Python popular in data science?

1. [0.8466] Python is a popular programming language for data science.
2. [0.3203] Data pipelines automate data processing workflows.

1. [0.7767] Cosine similarity computes similarity based on vector orientation.
2. [0.5065] Semantic similarity measures meaning rather than exact wording.
3. [0.1936] Information retrieval systems search relevant content.

============================================================
QUERY: Why is Python popular in data science?


1. [0.7767] Cosine similarity computes similarity based on vector orientation.
2. [0.5065] Semantic similarity measures meaning rather than exact wording.
3. [0.1936] Information retrieval systems search relevant content.

============================================================

1. [0.7767] Cosine similarity computes similarity based on vector orientation.
2. [0.5065] Semantic similarity measures meaning rather than exact wording.
3. [0.1936] Information retrieval systems search relevant content.

1. [0.7767] Cosine similarity computes similarity based on vector orientation.
2. [0.5065] Semantic similarity measures meaning rather than exact wording.
3. [0.1936] Information retrieval systems search relevant content.
1. [0.7767] Cosine similarity computes similarity based on vector orientation.
2. [0.5065] Semantic similarity measures meaning rather than exact wording.
3. [0.1936] Information retrieval systems search relevant content.

============================================================
2. [0.5065] Semantic similarity measures meaning rather than exact wording.
3. [0.1936] Information retrieval systems search relevant content.

============================================================
3. [0.1936] Information retrieval systems search relevant content.

============================================================
QUERY: Why is Python popular in data science?

============================================================
QUERY: Why is Python popular in data science?
QUERY: Why is Python popular in data science?


1. [0.8466] Python is a popular programming language for data science.
2. [0.3203] Data pipelines automate data processing workflows.
3. [0.2974] Artificial intelligence is widely used in medical diagnosis.

============================================================
QUERY: How do recommendation systems work?

1. [0.6762] Recommendation systems personalize user experiences.
3. [0.2974] Artificial intelligence is widely used in medical diagnosis.

============================================================
QUERY: How do recommendation systems work?

1. [0.6762] Recommendation systems personalize user experiences.

============================================================
QUERY: How do recommendation systems work?

1. [0.6762] Recommendation systems personalize user experiences.
============================================================
QUERY: How do recommendation systems work?

1. [0.6762] Recommendation systems personalize user experiences.
QUERY: How do recommendation systems work?

1. [0.6762] Recommendation systems personalize user experiences.
1. [0.6762] Recommendation systems personalize user experiences.
2. [0.3254] Information retrieval systems search relevant content.
3. [0.2803] APIs allow software systems to communicate with each other.
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
