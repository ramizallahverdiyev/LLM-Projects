import requests
import json

# ======================================
# SYSTEM PROMPT TEMPLATES
# ======================================
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
#     "  \"answer\": \"...\",\n"
#     "  \"reason\": \"...\",\n"
#     "  \"confidence\": 0.0-1.0\n"
#     "}\n"
#     "Never output anything outside JSON."
# )
#
# 3) Code generator (only return Python code)
# SYSTEM_PROMPT = (
#     "You are a code generator. ALWAYS respond with Python code only.\n"
#     "Do not add explanations, do not add comments.\n"
#     "Only pure code."
# )
#
# 4) Hidden chain-of-thought (internal reasoning, do not show steps)
# SYSTEM_PROMPT = (
#     "Use internal reasoning to solve problems but DO NOT reveal your reasoning. "
#     "Only output the final answer."
# )
#
# 5) Teacher mode (step-by-step explanation)
# SYSTEM_PROMPT = (
#     "You are a calm teacher. Always explain things step-by-step "
#     "in a simple and intuitive way."
# )
#
# 6) Structured answers (structured format)
# SYSTEM_PROMPT = (
#     "For EVERY answer, follow this structure:\n"
#     "1) Short answer\n"
#     "2) Explanation\n"
#     "3) Example\n"
# )
#
# 7) Role-based behavior (change role depending on question type)
# SYSTEM_PROMPT = (
#     "Adjust your behavior depending on the user's request:\n"
#     "- If the user asks for code → output only code.\n"
#     "- If the question is mathematical → answer directly.\n"
#     "- If explanation is needed → explain step-by-step.\n"
#     "- If the question is conceptual → provide analogies.\n"
# )
#
# ======================================
# FINAL SYSTEM PROMPT USED
# ======================================
SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Always answer clearly and concisely.\n"
    "For every answer, use this structure:\n"
    "1) Short answer\n"
    "2) Explanation\n"
    "3) Example\n"
)

# ======================================
# CHAT HISTORY & MEMORY CONFIG
# ======================================

history = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

# 1) Message count-based trimming (simple)
MAX_HISTORY = 10  # maximum number of user+assistant messages to keep


def trim_history_message_count(history):
    """
    Simple trimming: keep the system message, and only the last MAX_HISTORY messages from others.
    """
    if len(history) > MAX_HISTORY + 1:
        history[:] = [history[0]] + history[-MAX_HISTORY:]


# 2) Token-based trimming (enterprise style) – skeleton
# MAX_TOKENS = 2048
#
# def count_tokens(text: str) -> int:
#     """
#     Token count using Ollama's tokenize endpoint.
#     """
#     resp = requests.post(
#         "http://localhost:11434/api/tokenize",
#         json={"model": "llama3.1", "prompt": text}
#     )
#     resp.raise_for_status()
#     return len(resp.json().get("tokens", []))
#
# def trim_history_by_tokens(history):
#     """
#     To ensure the context window limit MAX_TOKENS is not exceeded,
#     keep the system message and messages from the end backwards.
#     """
#     total = 0
#     trimmed = [history[0]]  # system prompt kept
#
#     for msg in reversed(history[1:]):
#         t = count_tokens(msg["content"])
#         if total + t > MAX_TOKENS:
#             break
#         trimmed.insert(1, msg)
#         total += t
#
#     history[:] = trimmed


# 3) Semantic memory skeleton (initial idea for RAG)
# semantic_memory = []
#
# def store_important_message(msg: str):
#     """
#     Here you actually embed(msg) and store it in FAISS etc.
#     Simple skeleton – will be expanded later.
#     """
#     if "my name is" in msg.lower() or "i live in" in msg.lower():
#         semantic_memory.append(msg)
#
# def recall_relevant_context(query: str):
#     """
#     Semantic search will be done here – currently just a skeleton.
#     """
#     return []  # in the future: return top-k relevant messages


# ======================================
# LLaMA CHAT – STREAMING
# ======================================

def stream_llama_chat(history):
    """
    Ollama streaming /api/chat endpoint.
    history: list of system + user/assistant messages.
    """
    url = "http://localhost:11434/api/chat"

    data = {
        "model": "llama3.1",
        "messages": history,
        "stream": True,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 300,
            # "repeat_penalty": 1.1,
            # "top_k": 40,
        },
    }

    with requests.post(url, json=data, stream=True) as r:
        r.raise_for_status()
        full_answer = ""

        for line in r.iter_lines():
            if not line:
                continue

            chunk = json.loads(line.decode("utf-8"))

            # /api/chat streaming format: {"message": {"role": "...", "content": "..."}, "done": bool, ...}
            token = chunk.get("message", {}).get("content", "")
            full_answer += token

            # Real-time printing to the screen
            print(token, end="", flush=True)

            if chunk.get("done", False):
                break

        return full_answer


# ======================================
# MAIN LOOP
# ======================================

if __name__ == "__main__":
    print("Local LLaMA 3.1 Chatbot")

    while True:
        user_input = input("\n\nYou: ")

        # (optional) skeleton for storing important information in semantic memory
        # store_important_message(user_input)

        # Add user message to history
        history.append({"role": "user", "content": user_input})

        # Trimming – currently simple message count-based trimming is active
        trim_history_message_count(history)
        # To activate token-based trimming:
        # trim_history_by_tokens(history)

        print("\nAssistant: ", end="", flush=True)
        answer = stream_llama_chat(history)

        # Add assistant's answer to history
        history.append({"role": "assistant", "content": answer})

        # Trimming again (after the answer)
        trim_history_message_count(history)
        # or:
        # trim_history_by_tokens(history)
