import requests
import json

def ask_llm(question, context, model="llama3.2"):
    prompt = f"""
Use the context below to answer the question.
If the answer is not in the context, say: 'The document does not contain this information.'

CONTEXT:
{context}

QUESTION:
{question}
"""

    res = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "format": "json"
        },
        stream=True
    )

    answer = ""

    for line in res.iter_lines():
        if not line:
            continue

        data = json.loads(line.decode("utf-8"))

        # stream content
        if "response" in data:
            answer += data["response"]

        if data.get("done", False):
            break

    return answer.strip()