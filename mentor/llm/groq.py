import requests
from .client import LLMClient

class GroqClient(LLMClient):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def chat(self, *, messages, model, temperature, max_tokens):
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}

        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        r = requests.post(url, json=data, headers=headers)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
