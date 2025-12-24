"""
Embedding generation module.

Responsibilities:
- Load embedding model
- Convert text to vector representation
- Support single-text and batch embedding
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import (
    EMBEDDING_MODEL_NAME,
    NORMALIZE_EMBEDDINGS
)


class TextEmbedder:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string into a vector.
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=NORMALIZE_EMBEDDINGS
        )
        return embedding

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts into a 2D array of vectors.
        Shape: (num_texts, embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=NORMALIZE_EMBEDDINGS
        )
        return embeddings
