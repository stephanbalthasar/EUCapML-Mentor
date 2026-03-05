# mentor/llm/groq.py
import time
import random
import requests
from typing import List, Dict, Any, Optional
from requests.exceptions import HTTPError, RequestException

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

class GroqClient:
    """
    Minimal OpenAI-compatible client for Groq with robust retry/backoff.
    """

    def __init__(
        self,
        api_key: str,
        timeout: int = 30,
        max_retries: int = 4,
        backoff_base: float = 1.8,
        max_backoff: float = 12.0,
    ):
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.max_backoff = max_backoff

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> str:
        url = f"{GROQ_BASE_URL}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
        }
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)

        # Simple exponential backoff with jitter on 429/5xx
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(url, json=payload, headers=self._headers(), timeout=self.timeout)
                if r.status_code == 429:
                    # Respect Retry-After if present, else compute a backoff with jitter
                    retry_after_hdr = r.headers.get("retry-after")
                    if retry_after_hdr:
                        try:
                            wait = float(retry_after_hdr)
                        except ValueError:
                            wait = 0.0
                    else:
                        wait = min(self.max_backoff, (self.backoff_base ** attempt) + random.uniform(0, 0.5))
                    if attempt < self.max_retries:
                        time.sleep(wait)
                        continue
                    # Out of retries → raise a friendly error
                    raise HTTPError(f"Rate limited by provider (429). Please retry in ~{int(max(wait,3))}s.", response=r)

                # For other statuses, raise_for_status catches 4xx/5xx
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"]

            except (HTTPError, RequestException) as e:
                last_error = e
                status = getattr(getattr(e, "response", None), "status_code", None)
                # Retry on transient server errors
                if status and 500 <= status < 600 and attempt < self.max_retries:
                    wait = min(self.max_backoff, (self.backoff_base ** attempt) + random.uniform(0, 0.5))
                    time.sleep(wait)
                    continue
                # Network hiccups / timeouts: limited retries
                if not status and attempt < self.max_retries:
                    wait = min(self.max_backoff, (self.backoff_base ** attempt) + random.uniform(0, 0.5))
                    time.sleep(wait)
                    continue
                # Non-retryable or out of retries
                break

        # Bubble a gentle message upward (caller can show it in UI)
        raise RuntimeError(
            "Temporarily rate‑limited or service busy. Please wait a few seconds and try again."
            if isinstance(last_error, HTTPError) else
            f"Temporary network issue: {last_error}"
        )
