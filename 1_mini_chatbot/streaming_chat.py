import requests
import json

# System prompt
SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Always answer clearly and concisely."
)

# Chat history
history = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
]

def stream_llama(history):
    url = "http://localhost:11434/api/chat"

    data = {
        "model": "llama3.1",
        "messages": history,
        "stream": True,      # STREAMING enabled
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 300
        }
    }

    # Streaming request
    with requests.post(url, json=data, stream=True) as r:
        full_answer = ""

        # Iterate over streaming response tokens
        for line in r.iter_lines():
            if not line:
                continue

            chunk = json.loads(line.decode('utf-8'))

            token = chunk.get("message", {}).get("content", "")
            full_answer += token

            # Print token in real-time
            print(token, end="", flush=True)

            if chunk.get("done", False):
                break

        return full_answer


if __name__ == "__main__":
    print("Streaming Chatbot Ready!")

    while True:
        user_input = input("\n\nYou: ")

        # Add user message to history
        history.append({"role": "user", "content": user_input})

        print("\nAssistant: ", end="", flush=True)

        # Stream response token-by-token
        answer = stream_llama(history)

        # Add model answer to history
        history.append({"role": "assistant", "content": answer})
