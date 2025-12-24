import requests
import json

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer clearly and concisely."
)

# Chat history – system message remains
history = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

MAX_HISTORY = 10   # maximum number of user+assistant messages to keep


def stream_llama(history):
    url = "http://localhost:11434/api/chat"

    data = {
        "model": "llama3.1",
        "messages": history,
        "stream": True,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 300
        }
    }

    with requests.post(url, json=data, stream=True) as r:
        full_answer = ""

        for line in r.iter_lines():
            if not line:
                continue
            
            chunk = json.loads(line.decode('utf-8'))
            token = chunk.get("message", {}).get("content", "")
            full_answer += token
            print(token, end="", flush=True)

            if chunk.get("done", False):
                break

        return full_answer


def trim_history(history):
    # system message: history[0]
    # keep only the last MAX_HISTORY messages from others
    if len(history) > MAX_HISTORY + 1:
        history[:] = [history[0]] + history[-MAX_HISTORY:]


if __name__ == "__main__":
    print("Chatbot with Memory Trimming Ready!")

    while True:
        user_input = input("\n\nYou: ")

        history.append({"role": "user", "content": user_input})

        # Trimming applied
        trim_history(history)

        print("\nAssistant: ", end="", flush=True)
        answer = stream_llama(history)

        history.append({"role": "assistant", "content": answer})

        # Trimming applied
        trim_history(history)
