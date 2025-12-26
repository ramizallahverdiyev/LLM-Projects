from pathlib import Path
from ollama import chat  # Local Ollama API

class Interpreter:
    def __init__(self, prompt_path: str, model_name: str = "llama3.1:latest"):
        """
        Initialize the interpreter with an interpret prompt file and local Ollama model.
        
        Args:
            prompt_path (str): Path to interpret_prompt.txt
            model_name (str): Ollama model name
        """
        self.prompt_path = Path(prompt_path)
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        self.prompt_template = self.prompt_path.read_text()
        self.model_name = model_name

    def interpret(self, user_query: str, calculator_result) -> str:
        """
        Generate a human-readable explanation of the calculator result.
        
        Args:
            user_query (str): Original user query
            calculator_result: Result from calculator tool
        
        Returns:
            str: Explanatory text
        """
        # Fill prompt with user query and calculator result
        prompt = self.prompt_template.replace("{user_query}", user_query)\
                                     .replace("{calculator_result}", str(calculator_result))

        # Call local Ollama
        response = chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract response text
        response_text = response["content"] if isinstance(response, dict) and "content" in response else str(response)

        return response_text
