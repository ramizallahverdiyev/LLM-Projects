# src/utils/metrics.py

def precision_at_k(retrieved, relevant, k=5):
    """
    Compute Precision@k
    retrieved: list of retrieved document IDs
    relevant: set of relevant document IDs
    """
    retrieved_k = retrieved[:k]
    relevant_count = sum([1 for doc_id in retrieved_k if doc_id in relevant])
    return relevant_count / k

def recall_at_k(retrieved, relevant, k=5):
    """
    Compute Recall@k
    """
    retrieved_k = retrieved[:k]
    relevant_count = sum([1 for doc_id in retrieved_k if doc_id in relevant])
    return relevant_count / len(relevant) if relevant else 0.0
