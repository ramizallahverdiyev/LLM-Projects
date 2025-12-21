"""
Semantic search module.

Responsibilities:
- Embed query text
- Compare query embedding with stored embeddings
- Rank results by similarity score
- Return top-k matches
"""

from typing import List, Tuple
import numpy as np

from embeddings.embedder import TextEmbedder
from embeddings.embedding_store import EmbeddingStore
from similarity.metrics import cosine_similarity
from config.settings import TOP_K_RESULTS


def semantic_search(
    query: str,
    embedder: TextEmbedder,
    store: EmbeddingStore,
    top_k: int = TOP_K_RESULTS
) -> List[Tuple[str, float]]:
    """
    Perform semantic similarity search.

    Returns:
        List of (text, similarity_score) sorted by score descending
    """

    if len(store) == 0:
        raise ValueError("Embedding store is empty.")

    # 1. Embed the query
    query_embedding = embedder.embed_text(query)

    # 2. Retrieve stored texts and embeddings
    texts, embeddings = store.get_all()

    # 3. Compute similarity scores
    scores = []
    for text, emb in zip(texts, embeddings):
        score = cosine_similarity(query_embedding, emb)
        scores.append((text, score))

    # 4. Sort by similarity score (descending)
    scores.sort(key=lambda x: x[1], reverse=True)

    # 5. Return top-k results
    return scores[:top_k]
