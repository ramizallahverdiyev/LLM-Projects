import requests

# Chat history
history = [
    {
        "role": "system",
        "content": "You are a helpful and concise assistant."
    }
]

def ask_llama(history):
    url = "http://localhost:11434/api/chat"

    data = {
        "model": "llama3.1",
        "messages": history,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 200,
        }
    }

    response = requests.post(url, json=data)
    
    # Extract the answer from the JSON response
    return response.json()["message"]["content"]

if __name__ == "__main__":
    print("Chatbot ready!")

    while True:
        user_msg = input("\nYou: ")

        # Add user message to history
        history.append({"role": "user", "content": user_msg})

        # Get the answer from the model
        answer = ask_llama(history)
        print("\nAssistant:", answer)

        # Add model answer to history
        history.append({"role": "assistant", "content": answer})