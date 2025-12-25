# src/reranking/bge_reranker.py

from sentence_transformers import CrossEncoder

class BGEReranker:
    """
    BGE / cross-encoder based reranker.
    Takes top-k candidate documents and a query,
    returns documents sorted by relevance score.
    """
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        model_name: HuggingFace cross-encoder model
        """
        self.model = CrossEncoder(model_name)

    def rerank(self, query, candidate_docs):
        """
        Args:
            query (str): original or rewritten query
            candidate_docs (list[dict]): top-k documents from FAISS
        
        Returns:
            list[dict]: documents sorted by relevance score (highest first)
        """
        # Prepare pairs (query, doc_text)
        pairs = [(query, doc["text"]) for doc in candidate_docs]
        
        # Compute relevance scores
        scores = self.model.predict(pairs)
        
        # Attach scores and sort
        for doc, score in zip(candidate_docs, scores):
            doc["score"] = float(score)
        sorted_docs = sorted(candidate_docs, key=lambda x: x["score"], reverse=True)
        
        return sorted_docs
