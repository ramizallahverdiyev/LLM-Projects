import json
import re
from src.intent_detector import IntentDetector
from src.interpreter import Interpreter
from src.tools.calculator import evaluate_expression

class Agent:
    def __init__(self, intent_prompt_path: str, interpret_prompt_path: str):
        self.detector = IntentDetector(intent_prompt_path)
        self.interpreter = Interpreter(interpret_prompt_path)
    
    def handle_query(self, query: str) -> str:
        # Detect intent
        intent = self.detector.detect_intent(query)

        tool_result = None

        # Calculator tool
        if intent["tool_name"] == "calculator":
            expression = intent["arguments"]["expression"]
            expression = re.sub(r'what is\s*', '', expression, flags=re.I).strip().rstrip('?')

            calc_result = evaluate_expression(expression)
            tool_result = calc_result

        # Determine free_form mode
        free_form = intent["tool_name"] is None

        # Generate explanation
        explanation = self.interpreter.interpret(query, tool_result=tool_result, free_form=free_form)
        return explanation

