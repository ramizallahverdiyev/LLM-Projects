"""
Similarity metrics module.

Responsibilities:
- Compute similarity score between vectors
"""

import numpy as np


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Returns:
        float: similarity score in range [-1, 1]
    """
    if vec_a.ndim != 1 or vec_b.ndim != 1:
        raise ValueError("Input vectors must be 1-dimensional.")

    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("Zero-vector encountered in cosine similarity.")

    return dot_product / (norm_a * norm_b)
