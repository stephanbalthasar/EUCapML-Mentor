# mentor/llm/llm_registry.py
from dataclasses import dataclass
from typing import Dict, List

@dataclass(frozen=True)
class ModelInfo:
    provider: str      # "openrouter" | "groq"
    model_id: str      # e.g., "qwen/qwen3-30b-a3b:free"
    label: str         # UI label
    notes: str = ""    # Optional UI help

def get_model_registry() -> Dict[str, List[ModelInfo]]:
    """
    Provider -> [models]. Keep these slugs valid for 2026.
    """
    return {
        "openrouter": [
            # Zero-cost, rotates across available free endpoints
            ModelInfo("openrouter", "openrouter/free", "OpenRouter Free (auto)", "Zero-cost; rotates across free models"),
            # Qwen 3 free variants (valid slugs)
            ModelInfo("openrouter", "qwen/qwen3-30b-a3b:free", "Qwen3‑30B‑A3B (free)", "Reliable free Qwen3 variant"),
            ModelInfo("openrouter", "qwen/qwen3-235b-a22b:free", "Qwen3‑235B‑A22B (free)", "Stronger; higher latency"),
        ],
        "groq": [
            # Your existing Groq models
            ModelInfo("groq", "llama-3.3-70b-versatile", "Llama 3.3 70B (Groq)", "Fast; older cutoff"),
            ModelInfo("groq", "llama-3.1-8b-instant", "Llama 3.1 8B Instant (Groq)", "Ultra-fast; testing"),
        ],
    }

def get_default_model() -> ModelInfo:
    """
    Default to the OpenRouter Free router so the app works on a €0 budget.
    You can override via OPENROUTER_DEFAULT_MODEL in secrets.
    """
    return ModelInfo("openrouter", "openrouter/free", "OpenRouter Free (default)")
