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

def accumulate_signals(messages, gaz, alias_map, window=3):
    """
    Returns (accumulated_signals: List[dict], total_conf: float, has_case: bool).

    messages: list of last N user message strings.
    gaz, alias_map: passed into extract_signals().
    window: number of turns (already sliced by caller).
    """

    # Extract all signals
    collected = []
    for msg in messages:
        sigs = extract_signals(msg, gaz=gaz, corpus_auto_alias=alias_map) or []
        collected.extend(sigs)

    # Deduplicate by canonical → keep highest confidence per canonical
    best = {}
    for s in collected:
        can = s.get("canonical")
        if not can:
            continue
        conf = float(s.get("confidence", 0.0) or 0.0)
        if can not in best or conf > best[can].get("confidence", 0.0):
            best[can] = s

    unique_signals = list(best.values())

    # Accumulated confidence
    total_conf = sum(float(s.get("confidence", 0.0) or 0.0) for s in unique_signals)

    # Case flag
    has_case = any((s.get("type") or "").lower().strip() in _CASE_TYPES for s in unique_signals)

    return unique_signals, total_conf, has_case

def route(user_query: str, *, recent_user_messages=None) -> Dict[str, Any]:
    """
    NEW SIGNATURE:
        - recent_user_messages: list of last N user turns (strings)
          including the current message's predecessors (NOT including user_query).
          If None, fallback to old behavior (single-turn routing).

    OUTPUT unchanged.
    """

    # Fallback for empty query
    if not user_query or not user_query.strip():
        total_conf = 0.0
        mode = "chat"
        return {
            "mode": mode,
            "counts": {
                "concepts": 0, "cases": 0, "case_numbers": 0,
                "articles": 0, "sections": 0, "other": 0, "effective": 0
            },
            "count": 0,
            "total_conf": total_conf,
            "ui_label": _ui_mode_label(mode, total_conf),
            "router_version": "accumulated-2026-03-18",
        }

    # -------------------------------
    # 1) ACCUMULATED SIGNALS (NEW)
    # -------------------------------
    WINDOW = 3
    if not recent_user_messages:
        # Old behavior fallback: only route last query
        sigs = extract_signals(user_query, gaz=_gaz, corpus_auto_alias=_auto_alias) or []
        # Use existing summarizer for UI
        counts = _summarize_for_ui(sigs)
        total_conf = sum(float(s.get("confidence", 0.0) or 0.0) for s in sigs)
        has_case = any((s.get("type") or "").lower().strip() in _CASE_TYPES for s in sigs)
    else:
        # Slice last WINDOW - 1 previous messages (not including user_query)
        prev = recent_user_messages[-(WINDOW - 1):]
        msgs = prev + [user_query]

        unique_sigs, total_conf, has_case = accumulate_signals(
            msgs, gaz=_gaz, alias_map=_auto_alias, window=WINDOW
        )
        counts = _summarize_for_ui(unique_sigs)

    # -------------------------------
    # 2) DECIDE MODE (unchanged thresholds)
    # -------------------------------
    if total_conf >= 2.0:
        mode = "rag"
    elif total_conf > 1.5 and has_case:
        mode = "rag"
    else:
        mode = "chat"

    return {
        "mode": mode,
        "count": counts["effective"],
        "counts": counts,
        "total_conf": round(total_conf, 3),
        "ui_label": _ui_mode_label(mode, total_conf),
        "router_version": "accumulated-2026-03-18",
    }
