from pathlib import Path
from ollama import chat  # Local Ollama API

class Interpreter:
    def __init__(self, prompt_path: str, model_name: str = "llama3.1:latest"):
        self.model_name = model_name
        self.prompt_path = Path(prompt_path)
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        self.prompt_template = self.prompt_path.read_text()

    def interpret(self, query: str, tool_result=None, free_form: bool = False) -> str:
        """
        Uses the LLM to generate a human-readable explanation.

        Args:
            query (str): The user query.
            tool_result (optional): Result from a tool like calculator.
            free_form (bool): If True, generate a normal LLM response without requiring tool output.

        Returns:
            str: Human-readable response from the model.
        """

        if free_form:
            response = chat(
                model=self.model_name,
                messages=[{"role": "user", "content": query}]
            )
        else:
            # Tool-based explanation mode
            prompt = self.prompt_template.replace("{user_query}", query)
            if tool_result is not None:
                prompt += f"\nTool output: {tool_result}"

            response = chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )

        # Extract only the content
        if hasattr(response, "message") and hasattr(response.message, "content"):
            return response.message.content
        elif isinstance(response, dict) and "content" in response:
            return response["content"]

        # fallback
        return str(response)

