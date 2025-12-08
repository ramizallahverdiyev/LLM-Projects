# src/vectordb.py

import faiss
import numpy as np
import pickle
import os

class VectorDB:
    def __init__(self, dim=384, index_path="embeddings/faiss.index", meta_path="embeddings/metadata.pkl"):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path

        self.index = None
        self.metadata = []

        # Try to load existing index/metadata; fall back to a fresh index if files are missing or corrupted
        if os.path.exists(index_path) and os.path.getsize(index_path) > 0:
            try:
                self.index = faiss.read_index(index_path)
                if os.path.exists(meta_path) and os.path.getsize(meta_path) > 0:
                    self.metadata = pickle.load(open(meta_path, "rb"))
            except Exception as e:
                print(f"Warning: failed to load existing FAISS index, rebuilding. Reason: {e}")

        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
            self.metadata = []

    def add(self, embeddings, metadata_list):
        self.index.add(embeddings)
        self.metadata.extend(metadata_list)

    def save(self):
        faiss.write_index(self.index, self.index_path)
        pickle.dump(self.metadata, open(self.meta_path, "wb"))

    def search(self, query_vector, k=3):
        distances, indices = self.index.search(query_vector, k)
        results = [self.metadata[i] for i in indices[0]]
        return results
