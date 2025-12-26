from src.agent import Agent

agent = Agent("prompts/intent_prompt.txt", "prompts/interpret_prompt.txt")

print("Mini LLM Calculator Agent")
print("Type 'exit' to quit.\n")

while True:
    query = input("Enter your query: ").strip()
    if query.lower() in ["exit", "quit"]:
        break
    response = agent.handle_query(query)
    print(f"Agent: {response}\n")