# FAISS Vector Search with Approximate Indexing (IVF)

This project demonstrates how to build an **approximate vector search system** using **FAISS** with **IVF (Inverted File Index)**.
The focus is not on exact similarity search, but on understanding **how index selection, training, and search parameters affect performance and retrieval quality** in real systems.

---

## Project Motivation

Vector embeddings enable semantic similarity search, but **brute-force comparison does not scale**. In production systems, engineers must carefully design:

* Vector indexes
* Speed vs accuracy trade-offs
* Index training strategies
* Query-time tuning

This project is designed as a **learning-oriented, engineering-focused implementation** of FAISS IVF indexing.

---

## Key Concepts Covered

* Text → vector embeddings
* Approximate nearest neighbor (ANN) search
* FAISS IVF index design
* Index training vs query-time search
* `nlist` and `nprobe` trade-offs
* Distance interpretation
* Manual (brute-force) similarity baseline

---

## Project Structure

```
faiss-vector-search/
│
├── data/
│   ├── raw/
│   │   └── sentences.txt
│   └── processed/
│       └── embeddings.npy
│
├── src/
│   ├── data_loader.py
│   ├── embedding.py
│   ├── indexer.py
│   ├── search.py
│   ├── utils.py
│
├── experiments/
│   └── manual_similarity_check.py
│
├── notes/
│   └── observations.md
│
├── main.py
└── README.md
```

Each module follows **single-responsibility design**, mirroring real ML engineering workflows.

---

## Dataset

* Plain text file (`sentences.txt`)
* One sentence per line
* 235 sentences used in experiments
* Designed for **semantic diversity**, not linguistic complexity

Each sentence represents **one retrieval unit → one embedding vector**.

---

## Embedding Model

* Model: `sentence-transformers/all-MiniLM-L6-v2`
* Embedding dimension: **384**
* Vectors are **L2-normalized**
* Stored as `float32` for FAISS compatibility

Embeddings are cached to disk (`embeddings.npy`) to avoid recomputation.

---

## FAISS Index Design

### Index Type

* **IVF + Flat** (`IndexIVFFlat`)
* Approximate nearest neighbor search

### Why IVF?

* Demonstrates real indexing behavior
* Avoids trivial exact search
* Introduces cluster-based pruning
* Enables tunable recall vs latency trade-offs

### Parameters

* `nlist = 32` (number of clusters)
* `nprobe = 4` (clusters searched per query)

Index training is performed explicitly before vectors are added.

---

## Running the Project

```bash
python main.py
```

Pipeline steps:

1. Load sentences
2. Generate or load embeddings
3. Build IVF index
4. Train index (cluster centroids)
5. Add embeddings to index
6. Run KNN search queries
7. Print formatted results

---

## Example Search Outputs

### Query

```
vector similarity search with embeddings
```

Results:

```
01. Distance=0.2111 | Vector search retrieves similar embeddings.
02. Distance=0.6835 | Vector databases store embeddings.
03. Distance=0.7757 | Brute force search compares all vectors.
04. Distance=0.8545 | FAISS enables efficient similarity search.
05. Distance=0.8678 | Domain specific embeddings improve results.
```

---

### Query

```
machine learning models for data analysis
```

Results:

```
01. Distance=0.8555 | Machine learning models learn patterns from historical data.
02. Distance=1.1067 | Exploratory data analysis reveals data patterns.
03. Distance=1.1529 | Reproducibility is essential in data science.
04. Distance=1.1876 | Evaluation requires benchmark datasets.
05. Distance=1.1930 | Gradient boosting builds models sequentially.
```

---

### Query

```
how indexing improves search performance
```

Results:

```
01. Distance=0.4661 | Index tuning improves results.
02. Distance=0.4846 | Approximate search improves performance.
03. Distance=0.6122 | Indexes accelerate nearest neighbor queries.
04. Distance=0.6974 | Similarity search scales poorly without indexing.
05. Distance=0.7544 | IndexFlat performs exact search.
```

---

## Observations

* Approximate search may return slightly different neighbors across runs
* Increasing `nprobe` improves recall but increases latency
* Embedding quality dominates retrieval quality
* FAISS optimizes **search**, not **semantic meaning**
* Index parameters must scale with dataset size

Detailed notes are available in `notes/observations.md`.

---

## Manual Similarity Baseline

A brute-force cosine similarity implementation is included in:

```
experiments/manual_similarity_check.py
```

This baseline helps validate FAISS behavior and highlights why indexing is required for scalability.

---

## Engineering Takeaways

* Vector search is primarily a **systems problem**
* Index choice is a critical design decision
* Training and querying are separate phases
* Approximation is intentional and beneficial
* Clean separation of concerns improves maintainability

---

## Possible Extensions

* Compare different `nlist` values
* Benchmark `nprobe` vs latency
* Add metadata mapping
* Integrate with a RAG pipeline
* Compare IVF with HNSW

---

## Summary

This project implements a **realistic approximate vector search system** using FAISS IVF indexing, focusing on **how and why indexes are selected and tuned**, not just how to use an API.

It provides a strong foundation for larger retrieval systems and production-grade RAG architectures.
