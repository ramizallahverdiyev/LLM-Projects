from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.embedding import EmbeddingConfig, generate_embeddings
from src.indexer import FaissIndexer


class FaissSearcher:
    """
    Handles KNN search over a trained FAISS IVF index.
    """

    def __init__(
        self,
        indexer: FaissIndexer,
        embedding_config: EmbeddingConfig,
        nprobe: int = 4,
    ):
        """
        Parameters:
        - indexer: trained FaissIndexer instance
        - embedding_config: config used for query embedding
        - nprobe: number of clusters to probe during search
        """
        if indexer.index is None:
            raise RuntimeError("Indexer does not contain a built index.")

        self.indexer = indexer
        self.embedding_config = embedding_config
        self.nprobe = nprobe

        # Set IVF search parameter
        self.indexer.index.nprobe = nprobe

    def search(
        self,
        query: str,
        k: int = 5,
        return_distances: bool = True,
    ) -> Tuple[List[int], List[float]]:
        """
        Perform KNN search for a single query string.

        Returns:
        - indices of nearest neighbors
        - distances to nearest neighbors
        """
        if not query.strip():
            raise ValueError("Query string is empty.")

        # Convert query to embedding
        query_embedding = generate_embeddings(
            [query],
            self.embedding_config,
        )

        # FAISS expects shape (1, D)
        distances, indices = self.indexer.index.search(query_embedding, k)

        indices = indices[0].tolist()
        distances = distances[0].tolist()

        if return_distances:
            return indices, distances

        return indices, []
