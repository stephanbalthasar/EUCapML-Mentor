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
    Simplified confidence-based router (Option A):
      - Sum confidences over all signals returned by extract_signals().
      - If total_conf > 2.5 -> RAG
      - Else if total_conf > 1.5 AND there is a case_name or case_no -> RAG
      - Else -> assistant (chat)
    """
    if not user_query or not user_query.strip():
        return {
            "mode": "chat",
            "count": 0,
            "counts": {
                "concepts": 0, "cases": 0, "case_numbers": 0,
                "articles": 0, "sections": 0, "other": 0, "effective": 0
            },
            "total_conf": 0.0,
        }

    signals: List[Dict[str, Any]] = extract_signals(
        user_query,
        gaz=_gaz,
        corpus_auto_alias=_auto_alias,
    )

    # Sum confidences and compute lightweight counts (for UI only)
    total_conf = 0.0
    counts = {
        "concepts": 0, "cases": 0, "case_numbers": 0,
        "articles": 0, "sections": 0, "other": 0, "effective": 0
    }
    has_case = False

    for s in signals:
        total_conf += float(s.get("confidence", 0.0) or 0.0)
        t = (s.get("type") or "").lower()
        if t == "concept":
            counts["concepts"] += 1
        elif t == "case_name":
            counts["cases"] += 1
            has_case = True
        elif t == "case_no":
            counts["case_numbers"] += 1
            has_case = True
        elif t == "article":
            counts["articles"] += 1
        elif t == "section":
            counts["sections"] += 1
        else:
            counts["other"] += 1

    # effective (for display only; not used for routing anymore)
    counts["effective"] = (
        counts["concepts"]
        + counts["cases"]
        + counts["case_numbers"]
        + counts["articles"]
        + counts["sections"]
    )
    print("signals_for_debug=", [((s.get("type") or "").lower(),
                              (s.get("canonical") or ""),
                              float(s.get("confidence", 0.0) or 0.0))
                             for s in signals])
    print("sum_conf=", round(total_conf, 3), "has_case=", has_case)
    # Routing by your simplified rules
    if total_conf > 2.5:
        mode = "rag"
    elif total_conf > 1.5 and has_case:
        mode = "rag"
    else:
        mode = "chat"

    return {
        "mode": mode,
        "count": counts["effective"],  # UI display only
        "counts": counts,
        "total_conf": round(total_conf, 3),
    }
