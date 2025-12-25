from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class HyDERewriter:
    """
    HyDE-based query rewriting module using instruction-tuned FLAN-T5.
    Converts short user queries into 1-2 sentence expanded queries for retrieval.
    """
    def __init__(self, model_name="google/flan-t5-small", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def rewrite(self, query, max_length=60, temperature=0.7):
        """
        Rewrites a short query into 1-2 concise sentences without hallucination.
        """
        prompt = (
            f"Rephrase the following user query in 1–2 concise sentences "
            f"for retrieval purposes. Do not invent facts. Query: {query}"
        )
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id
            )
        rewritten = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return rewritten

# Example
if __name__ == "__main__":
    rewriter = HyDERewriter()
    query = "TOP universities in Azerbaijan"
    rewritten = rewriter.rewrite(query)
    print("Original Query:", query)
    print("Rewritten Query:", rewritten)
