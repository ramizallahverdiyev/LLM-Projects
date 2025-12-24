from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class FaissIndexer:
    """
    FAISS IVF index for approximate nearest neighbor search.

    This index:
    - partitions vector space into clusters (nlist)
    - searches only a subset of clusters at query time
    """

    def __init__(self, embedding_dim: int, nlist: int = 32):
        """
        Parameters:
        - embedding_dim: vector dimension (D)
        - nlist: number of Voronoi cells (clusters)
        """
        self.embedding_dim = embedding_dim
        self.nlist = nlist
        self.index = None

    def build_index(self) -> None:
        """
        Build an IVF Flat index (approximate search).
        """
        try:
            import faiss
        except ImportError as e:
            raise ImportError(
                "Missing dependency: faiss. "
            ) from e

        # Quantizer = coarse clustering (uses exact L2 internally)
        quantizer = faiss.IndexFlatL2(self.embedding_dim)

        # IVF index
        self.index = faiss.IndexIVFFlat(
            quantizer,
            self.embedding_dim,
            self.nlist,
            faiss.METRIC_L2,
        )

    def train(self, embeddings: np.ndarray) -> None:
        """
        Train the IVF index (learn cluster centroids).
        """
        if self.index is None:
            raise RuntimeError("Index not built yet.")

        if not self.index.is_trained:
            self.index.train(embeddings)

    def add_embeddings(self, embeddings: np.ndarray) -> None:
        """
        Add vectors to the trained index.
        """
        if self.index is None:
            raise RuntimeError("Index not built.")

        if not self.index.is_trained:
            raise RuntimeError("Index must be trained before adding vectors.")

        self.index.add(embeddings)

    def save_index(self, output_path: str) -> None:
        if self.index is None:
            raise RuntimeError("No index to save.")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        import faiss
        faiss.write_index(self.index, str(path))

    def load_index(self, input_path: str) -> None:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(input_path)

        import faiss
        self.index = faiss.read_index(str(path))

    @property
    def size(self) -> int:
        if self.index is None:
            return 0
        return self.index.ntotal
