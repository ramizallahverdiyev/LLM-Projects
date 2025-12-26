# src/intent_detector.py

import json
import re
from pathlib import Path
from ollama import chat  # Yeni Ollama API

class IntentDetector:
    def __init__(self, prompt_path: str, model_name: str = "llama3.1:latest"):
        """
        Initialize the intent detector with a prompt file and local Ollama model.
        
        Args:
            prompt_path (str): Path to intent_prompt.txt
            model_name (str): Ollama model name
        """
        self.prompt_path = Path(prompt_path)
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        self.prompt_template = self.prompt_path.read_text()
        self.model_name = model_name

    def detect_intent(self, user_query: str) -> dict:
        """
        Detects if the user query requires a tool, and returns arguments.

        Args:
            user_query (str): The input query from the user

        Returns:
            dict: {"tool_name": "calculator" or None, "arguments": {...}}
        """
        prompt = self.prompt_template.replace("{user_query}", user_query)

        # Call local Ollama
        response = chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract response text
        response_text = response["content"] if isinstance(response, dict) and "content" in response else str(response)

        # Attempt to parse JSON from response
        result = self._parse_json_from_text(response_text)

        # Fallback heuristic if JSON parsing fails or tool_name is None
        if not result.get("tool_name"):
            result = self._fallback_heuristic(user_query)

        return result

    def _parse_json_from_text(self, text: str) -> dict:
        """
        Safely extract JSON object from LLM response text.
        """
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return {"tool_name": None, "arguments": {}}
        return {"tool_name": None, "arguments": {}}

    def _fallback_heuristic(self, query: str) -> dict:
        """
        Simple heuristic: detect calculator queries if arithmetic operators or math functions exist.
        """
        if re.search(r'[\d\+\-\*/\^%]|sqrt|sin|cos', query):
            return {"tool_name": "calculator", "arguments": {"expression": query}}
        return {"tool_name": None, "arguments": {}}