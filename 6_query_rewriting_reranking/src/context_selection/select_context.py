from typing import List, Dict

class ContextSelector:
    """
    Selects the most relevant passages from top-k retrieved documents.
    """

    def __init__(self, max_context_tokens: int = 512):
        """
        max_context_tokens: maximum total tokens to return as context
        """
        self.max_context_tokens = max_context_tokens

    def select(self, candidate_docs: List[Dict]) -> List[Dict]:
        """
        Simple heuristic: 
        - sort candidate docs by 'score' if available
        - truncate text to fit max_context_tokens
        """
        # Sort by score if exists
        if "score" in candidate_docs[0]:
            candidate_docs = sorted(candidate_docs, key=lambda x: x["score"], reverse=True)

        selected_contexts = []
        total_tokens = 0

        for doc in candidate_docs:
            text = doc["text"]
            token_count = len(text.split())  # simple token approximation
            if total_tokens + token_count > self.max_context_tokens:
                # truncate text to remaining tokens
                remaining = self.max_context_tokens - total_tokens
                truncated_text = " ".join(text.split()[:remaining])
                selected_contexts.append({"title": doc["title"], "text": truncated_text})
                break
            else:
                selected_contexts.append({"title": doc["title"], "text": text})
                total_tokens += token_count

        return selected_contexts
