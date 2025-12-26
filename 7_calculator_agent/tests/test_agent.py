# tests/test_agent.py

from src.intent_detector import IntentDetector
from src.interpreter import Interpreter
from src.tools.calculator import evaluate_expression

detector = IntentDetector("prompts/intent_prompt.txt")
interpreter = Interpreter("prompts/interpret_prompt.txt")

def test_full_calculator_workflow():
    query = "What is 12 + 7?"

    intent = detector.detect_intent(query)
    assert intent["tool_name"] == "calculator"
    assert "expression" in intent["arguments"]

    expression = intent["arguments"]["expression"]
    expression = expression.replace("What is ", "").replace("?", "").strip()
    calc_result = evaluate_expression(expression)
    
    assert "result" in calc_result
    result_value = calc_result["result"]
    assert result_value == 19

    explanation = interpreter.interpret(query, result_value)
    assert str(result_value) in explanation

def test_non_calculator_workflow():
    query = "Explain gravity"

    intent = detector.detect_intent(query)
    assert intent["tool_name"] is None
    assert intent["arguments"] == {}
