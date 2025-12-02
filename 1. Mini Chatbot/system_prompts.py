import requests

# ================================
# SYSTEM PROMPT TEMPLATES
# ================================
#
# 1) Default assistant (normal chatbot)
# SYSTEM_PROMPT = (
#     "You are a helpful and concise assistant. "
#     "Always answer clearly and accurately."
# )
#
# 2) JSON-only output (RAG, API, production)
# SYSTEM_PROMPT = (
#     "You are a JSON-only assistant. ALWAYS respond only with valid JSON.\n"
#     "Format:\n"
#     "{\n"
#     "   \"answer\": \"...\",\n"
#     "   \"reason\": \"...\",\n"
#     "   \"confidence\": 0.0-1.0\n"
#     "}\n"
#     "Never output anything outside JSON."
# )
#
# 3) Code generator (only return Python code)
# SYSTEM_PROMPT = (
#     "You are a code generator. ALWAYS respond with Python code only.\n"
#     "Do not add explanations. Do not add comments.\n"
#     "Only pure code."
# )
#
# 4) Step-by-step reasoning (chain-of-thought hidden)
# SYSTEM_PROMPT = (
#     "Use internal reasoning to solve the problem, "
#     "but DO NOT reveal your reasoning. "
#     "Only output the final correct answer."
# )
#
# 5) Teacher mode (step-by-step explanation)
# SYSTEM_PROMPT = (
#     "You are a calm teacher. Always explain things step-by-step "
#     "in a simple and intuitive way."
# )
#
# 6) Strict structure response (fixed format)
# SYSTEM_PROMPT = (
#     "For EVERY answer, always follow this structure:\n"
#     "1) Short answer\n"
#     "2) Explanation\n"
#     "3) Example\n"
# )
#
# 7) Role-based behavior (dynamic role switching)
# SYSTEM_PROMPT = (
#     "Adjust your behavior depending on the user's request:\n"
#     "- If the user asks for code → output only code.\n"
#     "- If the question is mathematical → answer directly.\n"
#     "- If explanation is needed → explain step-by-step.\n"
#     "- If the question is conceptual → provide analogies.\n"
# )
#
# ================================
# CURRENTLY USED PROMPT
# ================================
SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Always answer clearly. "
    "Follow this structure:\n"
    "1) Short answer\n"
    "2) Explanation\n"
    "3) Example\n"
)

# Chat history (context)
history = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
]


def ask_llama(history):
    url = "http://localhost:11434/api/chat"

    data = {
        "model": "llama3.1",
        "messages": history,
        "stream": False,

        # ==================================
        # GENERATION PARAMETERS
        # ==================================
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 250,
            # "repeat_penalty": 1.1,
            # "top_k": 40
        }
    }

    response = requests.post(url, json=data)
    return response.json()["message"]["content"]


if __name__ == "__main__":
    print("Chatbot with Prompt Templates Ready!")

    while True:
        user_msg = input("\nYou: ")

        # Add user message to history
        history.append({"role": "user", "content": user_msg})

        # Get the answer from the model
        answer = ask_llama(history)
        print("\nAssistant:", answer)

        # Add model answer to history
        history.append({"role": "assistant", "content": answer})
