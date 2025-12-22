# FAISS Vector Search – Observations

## 1. Exact vs Approximate Search
Using IVF indexing introduces approximation.
Results may change slightly between runs.
This is expected and is a trade-off for speed.

## 2. Effect of nprobe
Increasing nprobe improves recall but increases latency.
Low nprobe values are fast but risk missing good neighbors.

## 3. Embedding Quality Matters
Poor embeddings produce poor retrieval, regardless of index.
FAISS optimizes search, not semantic meaning.

## 4. Manual vs FAISS Search
Manual brute-force similarity gives stable but slow results.
FAISS produces similar results with much better scalability.

## 5. Engineering Insight
Vector search is a systems problem, not just an ML problem.
Index choice is a critical design decision.
