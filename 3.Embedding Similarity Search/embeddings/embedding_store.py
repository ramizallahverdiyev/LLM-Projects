"""
Embedding storage module.

Responsibilities:
- Store text and its corresponding embedding
- Provide access to stored embeddings and texts
"""

from typing import List, Tuple
import numpy as np


class EmbeddingStore:
    def __init__(self):
        self._texts: List[str] = []
        self._embeddings: List[np.ndarray] = []

    def add(self, text: str, embedding: np.ndarray) -> None:
        """
        Store a single text and its embedding.
        """
        self._texts.append(text)
        self._embeddings.append(embedding)

    def add_batch(self, texts: List[str], embeddings: np.ndarray) -> None:
        """
        Store multiple texts and their embeddings.
        """
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts and embeddings must match.")

        for text, embedding in zip(texts, embeddings):
            self.add(text, embedding)

    def get_all(self) -> Tuple[List[str], np.ndarray]:
        """
        Retrieve all stored texts and embeddings.
        """
        return self._texts, np.vstack(self._embeddings)

    def __len__(self) -> int:
        return len(self._texts)
