from typing import List, Dict

class LLMClient:
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Return assistant content for the given chat messages."""
        raise NotImplementedError
