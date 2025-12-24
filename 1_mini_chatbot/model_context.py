import requests

history = [
    {
        "role": "system",
        "content": "You are a helpful and concise assistant. Answer clearly and accurately."
    }
]

def ask_llama(history):
    url = "http://localhost:11434/api/chat"

    data = {
        "model": "llama3.1",
        "messages": history,
        "stream": False
    }

    response = requests.post(url, json=data)
    return response.json()["message"]["content"]

if __name__ == "__main__":
    print("Chatbot ready!")

    while True:
        user_msg = input("\nYou: ")
        
        # Add user message
        history.append({"role": "user", "content": user_msg})

        # Ask the model
        answer = ask_llama(history)
        print("\nAssistant:", answer)

        # Add model reply to history
        history.append({"role": "assistant", "content": answer})
