from dataclasses import dataclass
from typing import Dict, List

@dataclass(frozen=True)
class ModelInfo:
    provider: str         # "openrouter" or "groq"
    model_id: str         # e.g., "qwen/qwen-3-instruct" or "llama-3.3-70b-versatile"
    label: str            # UI label
    notes: str = ""       # Shown in UI (optional)

def get_model_registry() -> Dict[str, List[ModelInfo]]:
    """Registry grouped by provider."""
    return {
        "openrouter": [
            ModelInfo("openrouter", "qwen/qwen-3-instruct", "Qwen 3 Instruct (default)", "Low hallucination; good for legal tutoring"),
            ModelInfo("openrouter", "qwen/qwen-2.5-instruct", "Qwen 2.5 Instruct", "Reliable and fast"),
            # You can add more OpenRouter models here
        ],
        "groq": [
            ModelInfo("groq", "llama-3.3-70b-versatile", "Llama 3.3 70B (Groq)", "Very fast; older knowledge cutoffs"),
            ModelInfo("groq", "llama-3.1-8b-instant", "Llama 3.1 8B Instant (Groq)", "Ultra-fast; use for testing only"),
        ]
    }

def get_default_model() -> ModelInfo:
    # Keep Qwen 3 Instruct as default
    return ModelInfo("openrouter", "qwen/qwen-3-instruct", "Qwen 3 Instruct (default)")
