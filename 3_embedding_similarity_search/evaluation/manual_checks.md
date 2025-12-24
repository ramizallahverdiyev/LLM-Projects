# Manual Evaluation – Embedding Similarity Search

This document records qualitative observations from running the
semantic similarity search system.

The goal is to understand system behavior, not to measure accuracy
numerically.

---

## 1. Setup

- Embedding model: all-MiniLM-L6-v2
- Similarity metric: cosine similarity
- Normalization: enabled
- Top-k results: 3
- Reference texts: ~30 short sentences
- Queries: mixed across AI, finance, healthcare, and general topics

---

## 2. Observations

### Query: "How do banks evaluate credit risk?"

Top results were strongly related to:
- credit scoring models
- loan approval decisions
- financial risk assessment

This indicates that semantic similarity captures conceptual meaning
even when wording differs.

---

### Query: "What does cosine similarity measure?"

Results focused on:
- vector orientation
- similarity metrics
- embedding space properties

The system correctly ignored unrelated texts.

---

### Query: "How do machines learn patterns from data?"

Top matches included:
- machine learning algorithms
- supervised and unsupervised learning concepts

The query was matched semantically rather than by keyword overlap.

---

### Query: "How is AI applied in medical diagnosis?"

Returned texts mentioned:
- medical imaging
- clinical decision support systems
- healthcare AI usage

This shows cross-domain semantic understanding.

---

## 3. Expected vs Actual Behavior

- Queries did not require exact keyword matches
- Conceptually similar texts ranked higher
- Some results had close similarity scores, indicating overlapping meanings

This behavior is expected for dense embedding spaces.

---

## 4. Limitations Observed

- Very short texts can be ambiguous
- Closely related topics may compete in ranking
- No thresholding: all results are returned regardless of relevance

These are limitations of simple embedding-based retrieval without
additional filtering or re-ranking.

---

## 5. Conclusions

- Embedding-based similarity search works well for semantic matching
- Cosine similarity is effective for normalized embeddings
- System behavior aligns with theoretical expectations

This project successfully demonstrates the core mechanics of
semantic similarity search using embeddings.
