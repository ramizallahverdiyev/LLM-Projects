from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class EmbeddingConfig:
    """
    Configuration for embedding generation.
    """
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    normalize: bool = True  # normalize vectors for cosine-like behavior


def generate_embeddings(
    sentences: List[str],
    config: EmbeddingConfig,
) -> np.ndarray:
    """
    Convert a list of sentences into a 2D numpy array of embeddings (N x D).
    Notes:
    - This function is intentionally independent from FAISS.
    - Normalization is useful for cosine similarity behavior.
    """
    if not sentences:
        raise ValueError("No sentences provided for embedding generation.")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "Missing dependency: sentence-transformers. "
        ) from e

    model = SentenceTransformer(config.model_name)

    embeddings = model.encode(
        sentences,
        batch_size=config.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=config.normalize,
    )

    # dtype is float32
    embeddings = embeddings.astype(np.float32)

    return embeddings


def save_embeddings(embeddings: np.ndarray, output_path: str) -> None:
    """
    Save embeddings to disk as a .npy file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D (N x D). Got shape: {embeddings.shape}")

    np.save(path, embeddings)


def load_embeddings(input_path: str) -> np.ndarray:
    """
    Load embeddings from a .npy file.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {input_path}")

    embeddings = np.load(path)

    #float32
    embeddings = embeddings.astype(np.float32)

    if embeddings.ndim != 2:
        raise ValueError(f"Loaded embeddings must be 2D (N x D). Got shape: {embeddings.shape}")

    return embeddings


def get_or_create_embeddings(
    sentences: List[str],
    config: EmbeddingConfig,
    cache_path: Optional[str] = None,
    force_recompute: bool = False,
) -> np.ndarray:
    """
    If cache_path is provided and exists, load embeddings from disk.
    Otherwise, compute embeddings and optionally save them.
    """
    if cache_path and (not force_recompute):
        path = Path(cache_path)
        if path.exists():
            return load_embeddings(cache_path)

    embeddings = generate_embeddings(sentences, config)

    if cache_path:
        save_embeddings(embeddings, cache_path)

    return embeddings
