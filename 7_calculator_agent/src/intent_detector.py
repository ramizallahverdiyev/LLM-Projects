import json
import re
from pathlib import Path
from ollama import chat  # Ollama local model

class IntentDetector:
    def __init__(self, prompt_path: str, model_name: str = "llama3.1:latest"):
        self.prompt_path = Path(prompt_path)
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        self.prompt_template = self.prompt_path.read_text()
        self.model_name = model_name

    def detect_intent(self, user_query: str) -> dict:
        prompt = self.prompt_template.replace("{user_query}", user_query)
        response = chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
        response_text = response.get("content", str(response)) if isinstance(response, dict) else str(response)
        result = self._parse_json_from_text(response_text)
        
        # Extract only content
        if hasattr(response, "message") and hasattr(response.message, "content"):
            response_text = response.message.content
        elif isinstance(response, dict) and "content" in response:
            response_text = response["content"]
        else:
            response_text = str(response)
        
        # Attempt to parse JSON from response
        result = self._parse_json_from_text(response_text)

        # Fallback heuristic
        if not result.get("tool_name"):
            result = self._fallback_heuristic(user_query)

        return result

    def _parse_json_from_text(self, text: str) -> dict:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return {"tool_name": None, "arguments": {}}
        return {"tool_name": None, "arguments": {}}

    def _fallback_heuristic(self, query: str) -> dict:
        if re.search(r'[\d\+\-\*/\^%]|sqrt|sin|cos', query):
            return {"tool_name": "calculator", "arguments": {"expression": query}}
        return {"tool_name": None, "arguments": {}}