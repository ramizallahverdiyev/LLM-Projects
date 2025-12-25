import json
import torch
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os

class FAISSRetriever:
    def __init__(self, model_name="all-MiniLM-L6-v2", index_path="index_ivf.faiss", nlist=100, device=None):
        """
        FAISS IVF + Flat retriever.
        nlist: number of clusters for IVF
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.index_path = index_path
        self.nlist = nlist
        self.documents = []

    def load_documents(self, path="data/documents.json"):
        """Load documents from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)
        print(f"Loaded {len(self.documents)} documents.")

    def build_index(self):
        """Build IVF + Flat FAISS index."""
        texts = [doc["text"] for doc in self.documents]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        dim = embeddings.shape[1]

        # IVF index creation
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, self.nlist, faiss.METRIC_L2)

        # Train and add embeddings
        index.train(embeddings)
        index.add(embeddings)

        # Save index
        faiss.write_index(index, self.index_path)
        print(f"FAISS IVF index saved to {self.index_path}.")
        return index

    def search(self, query_embedding, top_k=5):
        """
        Search top-k documents given a query embedding.
        Note: query is already embedded outside for flexibility.
        """
        index = faiss.read_index(self.index_path)
        index.nprobe = min(10, self.nlist)  # number of clusters to search
        distances, indices = index.search(query_embedding, top_k)
        results = [self.documents[idx] for idx in indices[0]]
        return results
