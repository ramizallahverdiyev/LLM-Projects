import numpy as np

from src.data_loader import load_sentences
from src.embedding import (
    EmbeddingConfig,
    get_or_create_embeddings,
)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    return float(np.dot(a, b))


def manual_knn_search(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    k: int = 5,
):
    """
    Brute-force KNN using cosine similarity.
    """
    similarities = []

    for idx, emb in enumerate(embeddings):
        score = cosine_similarity(query_embedding, emb)
        similarities.append((idx, score))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]


if __name__ == "__main__":
    sentences = load_sentences("data/raw/sentences.txt")

    config = EmbeddingConfig()
    embeddings = get_or_create_embeddings(
        sentences,
        config,
        cache_path="data/processed/embeddings.npy",
    )

    query = "vector similarity search using embeddings"

    query_embedding = get_or_create_embeddings(
        [query],
        config,
        force_recompute=True,
    )[0]

    results = manual_knn_search(query_embedding, embeddings, k=5)

    print("\nManual similarity results:")
    for idx, score in results:
        print(f"Score={score:.4f} | {sentences[idx]}")
