# mentor/engines/router.py

from typing import Literal, Dict, Set, List
from mentor.rag.booklet_retriever import extract_signals

Route = Literal["legal", "chit_chat"]

class SimpleRouter:
    """
    Plain-vanilla routing:
      - Extract signals using the retriever's gazetteers
      - Count how many gazetteer-based signals (concept, case_name)
      - If count >= 2 → legal
      - Else → chit_chat
    """

    def __init__(self, gazetteers, alias_bi: Dict[str, Set[str]], min_hits: int = 2):
        self.gaz = gazetteers
        self.alias_bi = alias_bi
        self.min_hits = min_hits

    def route(self, query: str) -> Route:
        sigs: List[Dict] = extract_signals(query, self.gaz, self.alias_bi)
        # gazetteer types come directly from the TXT files
        gaz_hits = sum(1 for s in sigs if s["type"] in {"concept", "case_name"})
        return "legal" if gaz_hits >= self.min_hits else "chit_chat"
