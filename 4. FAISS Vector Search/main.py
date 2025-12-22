from pathlib import Path
from src.data_loader import load_sentences
from src.embedding import (
    EmbeddingConfig,
    get_or_create_embeddings,
)
from src.indexer import FaissIndexer
from src.search import FaissSearcher
from src.utils import print_search_results

BASE_DIR = Path(__file__).resolve().parent

def main():
    # 1. Load raw sentences
    sentences = load_sentences(BASE_DIR / "data" / "raw" / "sentences.txt")
    print(f"Loaded {len(sentences)} sentences")

    # 2. Generate or load embeddings
    embedding_config = EmbeddingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=64,
        normalize=True,
    )

    embeddings = get_or_create_embeddings(
        sentences=sentences,
        config=embedding_config,
        cache_path=BASE_DIR / "data" / "processed" / "embeddings.npy",
        force_recompute=False,
    )

    print(f"Embeddings shape: {embeddings.shape}")

    # 3. Build and train FAISS IVF index
    embedding_dim = embeddings.shape[1]

    indexer = FaissIndexer(
        embedding_dim=embedding_dim,
        nlist=32,
    )

    indexer.build_index()
    indexer.train(embeddings)
    indexer.add_embeddings(embeddings)

    print(f"FAISS index size: {indexer.size}")

    # 4. Create searcher with nprobe tuning
    searcher = FaissSearcher(
        indexer=indexer,
        embedding_config=embedding_config,
        nprobe=4,
    )

    # 5. Test queries
    queries = [
        "vector similarity search with embeddings",
        "machine learning models for data analysis",
        "how indexing improves search performance",
    ]

    k = 5

    for query in queries:
        indices, distances = searcher.search(query, k=k)

        results = [sentences[i] for i in indices]

        print_search_results(
            query=query,
            results=results,
            distances=distances,
        )


if __name__ == "__main__":
    main()
