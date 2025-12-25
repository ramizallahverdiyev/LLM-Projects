from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class HyDERewriter:
    """
    HyDE-based query rewriting module.
    Converts short user queries into expanded synthetic queries.
    """
    def __init__(self, model_name="gpt2-medium", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

    def rewrite(self, query, max_length=100):
        """
        Generates a synthetic query/hypothetical document from the input query.
        """
        input_ids = self.tokenizer.encode(query, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7
            )
        rewritten_query = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return rewritten_query


if __name__ == "__main__":
    rewriter = HyDERewriter()
    query = "Top AI universities in Azerbaijan"
    rewritten = rewriter.rewrite(query)
    print("Original Query:", query)
    print("Rewritten Query:", rewritten)