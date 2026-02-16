import requests

OLLAMA_URL = "http://ollama:11434/api/generate"
MODEL = "mistral"


def generate_answer(context: str, question: str) -> str:

    if context.strip():
        prompt = f"""
You are a helpful assistant.

Answer ONLY from the context.

Context:
{context}

Question:
{question}

If not found, say:
"I could not find this in the document."
"""
    else:
        prompt = f"""
Answer this question clearly:

Question:
{question}
"""


    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    return response.json()["response"].strip()
