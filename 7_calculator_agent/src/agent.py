from src.intent_detector import IntentDetector
from src.interpreter import Interpreter
from src.tools.calculator import evaluate_expression

class Agent:
    """
    Mini LLM Agent:
    1. Detects the intent of the query
    2. Calls the calculator tool if intent is 'calculator'
    3. Uses interpreter to explain the result
    """

    def __init__(self, intent_prompt_path: str, interpret_prompt_path: str):
        self.detector = IntentDetector(intent_prompt_path)
        self.interpreter = Interpreter(interpret_prompt_path)

    def handle_query(self, query: str) -> str:
        # Detect intent
        intent = self.detector.detect_intent(query)

        # Calculator tool
        if intent["tool_name"] == "calculator":
            expression = intent["arguments"]["expression"]

            # Optional cleaning: remove phrases like "What is " and "?"
            expression = expression.replace("What is ", "").replace("?", "").strip()

            calc_result = evaluate_expression(expression)
            if "error" in calc_result:
                return f"Calculator error: {calc_result['error']}"
            result_value = calc_result["result"]
        else:
            result_value = None

        # Use interpreter
        explanation = self.interpreter.interpret(query, result_value)
        return explanation
