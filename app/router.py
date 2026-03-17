# app/router.py
# -----------------------------------------------------------------------------
# FIX #1 — CANONICAL-ONLY ROUTER (stable concept counting)
#
# Purpose:
#   Detect whether a query contains >= 2 canonical legal concepts
#   using only:
#       - gazetteer_concepts.txt
#       - gazetteer_cases.txt
#
#   Aliases are EXCLUDED entirely from routing logic.
#   This eliminates alias-inflation (e.g., C-628/13, “Lafonta v AMF” etc.)
#
#   Stable, deterministic behaviour:
#       “Summarize the ECJ decision in Lafonta” → 2 concepts
#       “Summarize the ECJ decision in Lafonat” → 2 concepts (fuzzy Lafonat→Lafonta)
#       “Summarize Lafonta” → 1 concept
#       “ECJ Spector Lafonta” → 3 concepts
#
# -----------------------------------------------------------------------------

from __future__ import annotations
import os
import unicodedata
import difflib
from typing import Dict, Set, List


# -----------------------------------------------------------------------------
# Normalization utilities
# -----------------------------------------------------------------------------
_HYPHEN_MAP = dict.fromkeys(map(ord, "‑–—−—"), ord("-"))

def _norm(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = t.translate(_HYPHEN_MAP)
    t = t.lower().strip()
    return " ".join(t.split())


# Stopwords ignored in fuzzy matching
_STOPWORDS = {
    "can", "you", "tell", "me", "about", "what", "summarize", "summarise",
    "please", "in", "the", "a", "an", "on", "of", "decision",
    "explain", "describe", "give", "provide", "case", "ec",
}


# -----------------------------------------------------------------------------
# Load canonical gazetteer ONLY (concepts + cases)
# -----------------------------------------------------------------------------
def _read_file(path: str) -> List[str]:
    if not os.path.exists(path):
        raise RuntimeError(f"Gazetteer file missing: {path}")
    out = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def _dedup_nested_canonicals(terms: List[str]) -> List[str]:
    """
    Remove shorter canonical entries that are substrings of longer ones.
    E.g. ["spector", "spector photo"] → ["spector photo"]
    """
    normed = sorted({_norm(t) for t in terms}, key=len, reverse=True)
    deduped = []
    for t in normed:
        if not any(t in longer for longer in deduped):
            deduped.append(t)
    return deduped


def _load_canonical_terms() -> List[str]:
    base_dir = os.path.dirname(os.path.abspath(__file__))

    concepts_path = os.path.join(base_dir, "..", "mentor", "rag", "gazetteer_concepts.txt")
    cases_path    = os.path.join(base_dir, "..", "mentor", "rag", "gazetteer_cases.txt")

    concepts = _read_file(concepts_path)
    cases    = _read_file(cases_path)

    canonicals = concepts + cases
    return _dedup_nested_canonicals(canonicals)


_CANONICAL_TERMS: List[str] = _load_canonical_terms()


# -----------------------------------------------------------------------------
# Matching logic: EXACT + SAFE FUZZY on CANONICALS only
# -----------------------------------------------------------------------------
def _canonical_match(canonical: str, q: str, q_words: List[str]) -> bool:
    """
    Return True if this canonical concept appears in the query:
      - EXACT substring → True
      - FUZZY match:
            * canonical is single-word
            * token length >= 5
            * ratio >= 0.88
            * skip stopwords
    """

    # 1) Exact match first (multi-word allowed)
    if canonical in q:
        return True

    # 2) Fuzzy match only for single-word canonicals
    if " " in canonical:
        return False

    if len(canonical) < 5:
        return False

    for w in q_words:
        if len(w) < 5:
            continue
        if w in _STOPWORDS:
            continue

        ratio = difflib.SequenceMatcher(None, w, canonical).ratio()
        if ratio >= 0.88:
            return True

    return False


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def route(user_query: str, *, threshold: int = 2) -> Dict[str, int]:
    """
    Count canonical gazetteer hits (exact + safe fuzzy).
    >= threshold → RAG
    <  threshold → Chat
    """
    q = _norm(user_query)
    if not q:
        return {"mode": "chat", "count": 0}

    q_words = q.split()
    hits = 0

    for canon in _CANONICAL_TERMS:
        if _canonical_match(canon, q, q_words):
            hits += 1

    mode = "rag" if hits >= threshold else "chat"
    return {"mode": mode, "count": hits}
