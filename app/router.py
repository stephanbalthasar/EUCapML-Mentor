# router.py
from __future__ import annotations
from typing import Any, Dict, List

# Resilient import: works whether you run as a package or locally
try:
    from mentor.rag.booklet_retriever import extract_signals, ParagraphRetriever  # package layout
except Exception:
    from booklet_retriever import extract_signals, ParagraphRetriever  # local layout

# One global retriever instance: loads gazetteers, booklet corpus, and auto-aliases once
_retriever = ParagraphRetriever()
_gaz = _retriever.gaz
_auto_alias = _retriever.alias_bi  # merged alias map (gazetteer + auto-alias)

# Case-like types we consider for the “has case” condition
_CASE_TYPES = {"case_name", "case_no", "case_number", "case"}


def _summarize_for_ui(signals: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Build lightweight counts purely for display/debug (NOT used for routing).
    We exclude 'other' from 'effective' just like before.
    """
    counts = {
        "concepts": 0,
        "cases": 0,          # case_name
        "case_numbers": 0,   # case_no
        "articles": 0,
        "sections": 0,
        "other": 0,
    }
    for s in signals:
        t = (s.get("type") or "").strip().lower()
        if t == "concept":
            counts["concepts"] += 1
        elif t == "case_name":
            counts["cases"] += 1
        elif t == "case_no":
            counts["case_numbers"] += 1
        elif t == "article":
            counts["articles"] += 1
        elif t == "section":
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


def _ui_mode_label(mode: str, total_conf: float) -> str:
    """
    Returns a ready-to-display label like:
        "Mode: RAG (Confidence=1.99)"  or  "Mode: Chat (Confidence=1.40)"
    """
    return f"Mode: {'RAG' if mode == 'rag' else 'Chat'} (Confidence={total_conf:.2f})"


def route(user_query: str) -> Dict[str, Any]:
    """
    Confidence-based router (Option A, finalized):
      - Let signals = extract_signals(query)
      - total_conf = sum(confidence for each signal)
      - has_case = any(type in {'case_name','case_no','case_number','case'})
      - Routing:
          * if total_conf > 2.5 -> 'rag'
          * elif total_conf > 1.5 and has_case -> 'rag'
          * else -> 'chat'
    Always returns a UI label showing the mode and the exact confidence sum used.
    """
    # Empty/whitespace query -> assistant chat with zero diagnostics
    if not user_query or not user_query.strip():
        total_conf = 0.0
        mode = "chat"
        return {
            "mode": mode,
            "counts": {
                "concepts": 0, "cases": 0, "case_numbers": 0,
                "articles": 0, "sections": 0, "other": 0, "effective": 0
            },
            "count": 0,  # effective count (display only)
            "total_conf": total_conf,
            "ui_label": _ui_mode_label(mode, total_conf),
            "router_version": "conf-sum-A-UI-2026-03-17",
        }

    # 1) Extract signals
    signals: List[Dict[str, Any]] = extract_signals(
        user_query,
        gaz=_gaz,
        corpus_auto_alias=_auto_alias,
    )

    # 2) Sum confidences and compute has_case
    total_conf = 0.0
    has_case = False

    for s in signals:
        # confidence defaults safely to 0.0 if missing
        total_conf += float(s.get("confidence", 0.0) or 0.0)
        t = (s.get("type") or "").strip().lower()
        if t in _CASE_TYPES:
            has_case = True

    # 3) Decide mode (ONLY by confidence and has_case per your spec)
    if total_conf >= 2.0:
        mode = "rag"
    elif total_conf > 1.5 and has_case:
        mode = "rag"
    else:
        mode = "chat"

    # 4) UI counts (display/debug only; not part of the decision)
    counts = _summarize_for_ui(signals)

    # 5) Return with a ready-to-display label
    return {
        "mode": mode,
        "count": counts["effective"],    # for display only
        "counts": counts,
        "total_conf": round(total_conf, 3),
        "ui_label": _ui_mode_label(mode, total_conf),  # <- always indicates mode + confidence
        "router_version": "conf-sum-A-UI-2026-03-17",
    }
