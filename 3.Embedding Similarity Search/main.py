"""
Main orchestration script for the Embedding Similarity Search project.

This file:
- loads data
- creates embeddings
- stores embeddings
- runs semantic search for each query
- prints results
"""

from data.raw_texts import RAW_TEXTS
from data.queries import QUERIES

from embeddings.embedder import TextEmbedder
from embeddings.embedding_store import EmbeddingStore

from similarity.search import semantic_search
from config.settings import TOP_K_RESULTS


def main():
    embedder = TextEmbedder()
    store = EmbeddingStore()

    text_embeddings = embedder.embed_batch(RAW_TEXTS)
    store.add_batch(RAW_TEXTS, text_embeddings)

    print(f"Loaded {len(store)} reference texts.\n")


    for query in QUERIES:
        print("=" * 60)
        print(f"QUERY: {query}\n")

        results = semantic_search(
            query=query,
            embedder=embedder,
            store=store,
            top_k=TOP_K_RESULTS
        )

        for rank, (text, score) in enumerate(results, start=1):
            print(f"{rank}. [{score:.4f}] {text}")

        print()

    print("Semantic search completed.")


if __name__ == "__main__":
    main()
