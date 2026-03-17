# app/router2.py
from __future__ import annotations
from typing import Dict, Any, List

# Import the retriever and its signal extractor
from mentor.rag.booklet_retriever import extract_signals, ParagraphRetriever

# One global retriever instance: loads gazetteers, booklet corpus, and auto-aliases once
_retriever = ParagraphRetriever()               # loads gazetteers + corpus
_gaz = _retriever.gaz
_auto_alias = _retriever.alias_bi               # merged alias map (gazetteer + auto-alias)

# -----------------------------
# Internal helpers
# -----------------------------
def _summarize_signals(signals: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Build type-aware counts. We intentionally exclude 'other' from 'effective' decisions.
    Also dedupe per (type, canonical) to avoid double-counting.
    """
    counts = {
        "concepts": 0,
        "cases": 0,          # case_name
        "case_numbers": 0,   # case_no
        "articles": 0,
        "sections": 0,
        "other": 0,
    }
    seen_keys = set()

    for s in signals:
        typ = (s.get("type") or "").lower()
        can = (s.get("canonical") or "").lower()
        key = (typ, can)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        if typ == "concept":
            counts["concepts"] += 1
        elif typ == "case_name":
            counts["cases"] += 1
        elif typ == "case_no":
            counts["case_numbers"] += 1
        elif typ == "article":
            counts["articles"] += 1
        elif typ == "section":
            counts["sections"] += 1
        else:
            counts["other"] += 1

    counts["effective"] = (
        counts["concepts"]
        + counts["cases"]
        + counts["case_numbers"]
        + counts["articles"]
        + counts["sections"]
    )
    return counts

# -----------------------------
# Public API
# -----------------------------
def route(user_query: str) -> Dict[str, Any]:
    """
    Signal-based router (robust counting):
      - Extract signals with extract_signals().
      - Compute type-aware counts and 'effective' (excludes 'other').
      - RAG if effective > 2
      - RAG if effective == 2 AND at least one is a case_name/case_no
      - Else Chat
    """
    if not user_query or not user_query.strip():
        return {
            "mode": "chat",
            "count": 0,
            "counts": {
                "concepts": 0, "cases": 0, "case_numbers": 0, "articles": 0, "sections": 0, "other": 0, "effective": 0
            }
        }

    # 1) Extract signals (uses gazetteers + alias maps from the retriever)
    signals: List[Dict[str, Any]] = extract_signals(
        user_query,
        gaz=_gaz,
        corpus_auto_alias=_auto_alias,
    )
    # (Extractor behavior & dedup come from your current booklet_retriever (3).py.)  # noqa

    # 2) Summarize counts
    counts = _summarize_signals(signals)
    n_eff = counts["effective"]
    has_case = (counts["cases"] > 0) or (counts["case_numbers"] > 0)

    # 3) Routing rules
    if n_eff > 2:
        mode = "rag"
    elif n_eff == 2 and has_case:
        mode = "rag"
    else:
        mode = "chat"

    # 4) Return detailed counts for UI
    return {
        "mode": mode,
        "count": n_eff,   # effective count (excludes 'other')
        "counts": counts,
    }
