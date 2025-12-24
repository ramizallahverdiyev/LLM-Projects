import requests

def ask_llama(prompt):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "llama3.1",
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, json=data)
    result = response.json()
    return result["response"]

if __name__ == "__main__":
    question = input("Ask your question: ")
    answer = ask_llama(question)
    print("\nModel answer:")
    print(answer)
