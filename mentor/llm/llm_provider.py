from typing import List, Dict, Optional, Tuple
from providers.openrouter import OpenRouterClient
from providers.groq_client import GroqClient
from llm_registry import get_default_model, get_model_registry, ModelInfo

class LLMProvider:
    def __init__(self):
        self._openrouter = OpenRouterClient()
        self._groq = GroqClient()
        self._default = get_default_model()

    def list_models(self) -> Tuple[ModelInfo, dict]:
        """Return (default_model, all_models_grouped)."""
        return self._default, get_model_registry()

    def _client_for(self, provider: str):
        if provider == "openrouter":
            return self._openrouter
        if provider == "groq":
            return self._groq
        raise ValueError(f"Unknown provider: {provider}")

    def is_available(self, provider: str) -> bool:
        return self._client_for(provider).is_configured

    def complete(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        top_p: float = 0.9,
        allow_fallback: bool = True
    ) -> str:
        """
        Try the requested provider/model. If it fails and allow_fallback=True,
        fall back to default (Qwen on OpenRouter).
        """
        chosen_provider = provider or self._default.provider
        chosen_model = model or self._default.model_id

        # Try primary
        primary_client = self._client_for(chosen_provider)
        try:
            if not primary_client.is_configured:
                raise RuntimeError(f"{chosen_provider} not configured.")
            return primary_client.complete(
                messages,
                model=chosen_model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            )
        except Exception as e:
            if not allow_fallback:
                raise

            # Fallback to default (Qwen via OpenRouter)
            fallback_client = self._client_for(self._default.provider)
            if not fallback_client.is_configured:
                # If fallback also not configured, raise original error
                raise e
            return fallback_client.complete(
                messages,
                model=self._default.model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            )
