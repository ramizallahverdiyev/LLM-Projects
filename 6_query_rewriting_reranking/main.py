from src.query_rewriting.hyde_rewriter import HyDERewriter
from src.embeddings.build_index import FAISSRetriever
from src.reranking.bge_reranker import BGEReranker
from src.context_selection.select_context import ContextSelector
from sentence_transformers import SentenceTransformer

def main():
    # 1. User query input
    user_query = input("Enter your query: ")

    # 2. Query rewrite (HyDE)
    rewriter = HyDERewriter()
    synthetic_query = rewriter.rewrite(user_query)
    print(f"Original Query: {user_query}")
    print(f"Rewritten Query: {synthetic_query}\n")

    # 3. FAISS index build + retrieval
    retriever = FAISSRetriever(nlist=10)
    retriever.load_documents("data/documents.json")
    retriever.build_index()

    # Query embedding
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = embed_model.encode([synthetic_query], convert_to_numpy=True)

    top_k_docs = retriever.search(query_embedding, top_k=5)

    print("Top-k retrieved documents (before reranking):")
    for doc in top_k_docs:
        print("-", doc["title"])

    # 4. Reranking
    reranker = BGEReranker()
    reranked_docs = reranker.rerank(synthetic_query, top_k_docs)

    print("\nTop-k documents after reranking:")
    for doc in reranked_docs:
        print(f"{doc['title']} (score: {doc['score']:.4f})")

    # 5. Context selection
    selector = ContextSelector(max_context_tokens=200)
    final_contexts = selector.select(reranked_docs)

    print("\nFinal selected context(s):")
    for ctx in final_contexts:
        print(f"{ctx['title']}: {ctx['text'][:100]}...")  # preview first 50 characters

if __name__ == "__main__":
    main()
