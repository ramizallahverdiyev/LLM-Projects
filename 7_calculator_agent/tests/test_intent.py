from src.intent_detector import IntentDetector

detector = IntentDetector("prompts/intent_prompt.txt")

def test_calculator_intent():
    result = detector.detect_intent("12 + 7")
    assert result["tool_name"] == "calculator"
    assert "expression" in result["arguments"]

def test_non_calculator_intent():
    result = detector.detect_intent("Explain gravity")
    assert result["tool_name"] is None
    assert result["arguments"] == {}

def test_complex_calculator_intent():
    result = detector.detect_intent("Calculate sqrt(64) + 3 ** 2")
    assert result["tool_name"] == "calculator"
    assert "expression" in result["arguments"]