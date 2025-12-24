"""
Central configuration for the Embedding Similarity Search project.

This file contains only configuration values.
"""
# Embedding configuration

# Embedding model identifier
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Expected embedding dimension
EMBEDDING_DIMENSION = 384

# Similarity metric
SIMILARITY_METRIC = "cosine"

# Number of top results to return
TOP_K_RESULTS = 3

# Whether to normalize embeddings before similarity computation
NORMALIZE_EMBEDDINGS = True
