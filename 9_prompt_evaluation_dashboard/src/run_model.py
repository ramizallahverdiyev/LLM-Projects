import requests
import json

def run_llama_model(prompt):
    """
    Runs the local Llama model with the given prompt by calling a local API.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.2:latest",
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)

        if response.status_code != 200:
            return f"Error: Model API returned status code {response.status_code}. Response: {response.text}"

        response.raise_for_status()

        data = response.json()
        model_output = data.get("response", "No response from model.")
        
        return model_output

    except requests.exceptions.RequestException as e:
        print(f"Error calling the model API: {e}")
        return f"Error: Could not connect to the model API at {url}. Is the model server running?"

if __name__ == '__main__':
    prompt_text = "What is the capital of France?"
    output = run_llama_model(prompt_text)
    print(f"Prompt: {prompt_text}")
    print(f"Model Output: {output}")
