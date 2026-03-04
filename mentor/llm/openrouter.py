import os
import requests
from typing import List, Dict, Optional

OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"

class OpenRouterClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        http_referer: Optional[str] = None,
        x_title: Optional[str] = None,
        timeout: int = 60
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.model = model or os.getenv("OPENROUTER_DEFAULT_MODEL", "qwen/qwen-3-instruct")
        self.http_referer = http_referer or os.getenv("OPENROUTER_HTTP_REFERER", "")
        self.x_title = x_title or os.getenv("OPENROUTER_X_TITLE", "EUCapML Case Tutor")
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        top_p: float = 0.9
    ) -> str:
        if not self.is_configured:
            raise RuntimeError("OpenRouter is not configured (missing OPENROUTER_API_KEY).")

        model_id = model or self.model
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        # Optional attribution headers (recommended by OpenRouter)
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.x_title:
            headers["X-Title"] = self.x_title

        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p
        }
        resp = requests.post(OPENROUTER_BASE, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
