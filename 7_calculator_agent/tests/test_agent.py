# tests/test_agent.py

from src.agent import Agent

agent = Agent("prompts/intent_prompt.txt", "prompts/interpret_prompt.txt")

def test_full_calculator_workflow():
    query = "What is 12 + 7?"
    explanation = agent.handle_query(query)
    assert "19" in explanation

def test_non_calculator_workflow():
    query = "Explain gravity"
    explanation = agent.handle_query(query)
    assert "Explain gravity" in explanation
