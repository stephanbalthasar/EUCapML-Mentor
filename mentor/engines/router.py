# app/router.py
# -----------------------------------------------------------------------------
# FIX #1 — CANONICAL-ONLY ROUTER (final)
# Detect concepts using ONLY canonical case names + canonical concepts.
# Aliases are ignored. One hit per canonical. Safe fuzzy typo-handling.
# -----------------------------------------------------------------------------

from __future__ import annotations
import os
import unicodedata
import difflib
from typing import List, Dict


# -----------------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------------
_HYPHEN_MAP = dict.fromkeys(map(ord, "‑–—−—"), ord("-"))

def _norm(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_HYPHEN_MAP)
    s = s.lower()
    return " ".join(s.split())


_STOPWORDS = {
    "can","you","tell","me","about","what","summarize","summarise",
    "please","in","the","a","an","on","of","decision","explain","describe",
    "give","provide","case"
}


# -----------------------------------------------------------------------------
# Load canonical terms (concepts + cases)
# -----------------------------------------------------------------------------
def _read_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

def _dedup_nested_canonicals(terms: List[str]) -> List[str]:
    """
    Minimal fix:
    - Normalize all terms
    - Remove exact duplicates ONLY
    - Preserve short canonicals (e.g., MAR, WpHG)
    - Preserve multi‑word concepts (e.g., inside information)
    """
    seen = set()
    deduped = []

    for t in terms:
        n = _norm(t)
        if n and n not in seen:
            seen.add(n)
            deduped.append(n)

    return deduped

def _load_canonicals() -> List[str]:
    base = os.path.dirname(os.path.abspath(__file__))
    concepts  = _read_file(os.path.join(base, "..", "mentor", "rag", "gazetteer_concepts.txt"))
    cases     = _read_file(os.path.join(base, "..", "mentor", "rag", "gazetteer_cases.txt"))
    return _dedup_nested_canonicals(concepts + cases)


_CANONICALS = _load_canonicals()


# -----------------------------------------------------------------------------
# Concept matching
# -----------------------------------------------------------------------------
def _canonical_matches(canon: str, q: str, q_words: List[str]) -> bool:
    """
    Match user query against a canonical concept:
      - EXACT substring match → match
      - SUBSTRING token match (token is part of canonical) → match
      - FUZZY typo match (single-word tokens) → match
    """

    # Exact match
    if canon in q:
        return True

    # Token-subset match (user token inside canonical)
    for w in q_words:
        if len(w) >= 5 and w not in _STOPWORDS:
            if w in canon:
                return True

    # Fuzzy (single-word canonicals only)
    if " " not in canon and len(canon) >= 5:
        for w in q_words:
            if len(w) < 5 or w in _STOPWORDS:
                continue
            if difflib.SequenceMatcher(None, w, canon).ratio() >= 0.88:
                return True

    return False


# -----------------------------------------------------------------------------
# Public router
# -----------------------------------------------------------------------------
def route(user_query: str, *, threshold: int = 2) -> Dict[str, int]:
    q = _norm(user_query)
    if not q:
        return {"mode": "chat", "count": 0}

    q_words = q.split()

    hits = 0
    for canon in _CANONICALS:
        if _canonical_matches(canon, q, q_words):
            hits += 1

    mode = "rag" if hits >= threshold else "chat"
    return {"mode": mode, "count": hits}
