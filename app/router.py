# app/router.py
# -----------------------------------------------------------------------------
# Heuristic Router
# Uses existing booklet_retriever machinery to detect legal concepts and decide:
#   - <2 concepts  → Chat mode  (assist)
#   - ≥2 concepts  → RAG mode   (answer)
# No LLM calls. No duplication of gazetteer logic.
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import Dict, Any, List, Set

# We reuse the SAME functions your retriever uses:
from mentor.rag.booklet_retriever import extract_signals, ParagraphRetriever


# -----------------------------------------------------------------------------
# INTERNAL: cluster signals into concepts (avoid double-counting aliases)
# -----------------------------------------------------------------------------
def _cluster_signals(signals: List[Dict]) -> List[Set[str]]:
    """
    Given a list of signals (each having types and expanded alias sets),
    build clusters of signals whose expanded sets overlap.

    Returns a list of clusters, each cluster is a set of canonical labels
    (purely for debug/inspection).
    """
    clusters: List[Set[str]] = []

    for s in signals:
        # Only count legal concepts, not "other"
        if s["type"] not in {"concept", "case_name", "case_no", "article", "section"}:
            continue

        expanded = set(x.lower() for x in s["expanded"])
        canonical = s["canonical"].lower()

        placed = False
        for cl in clusters:
            # If overlap → same concept cluster
            if cl & expanded:
                cl.add(canonical)
                cl.update(expanded)
                placed = True
                break
        if not placed:
            clusters.append(set([canonical, *expanded]))

    return clusters


# -----------------------------------------------------------------------------
# PUBLIC API: analyze query → return concept count & routing decision
# -----------------------------------------------------------------------------
def route(
    user_query: str,
    pr: ParagraphRetriever,
    *,
    threshold: int = 2
) -> Dict[str, Any]:
    """
    Analyze query with existing gazetteer/signal machinery and return:
      {
        "mode": "rag" | "chat",
        "count": int
      }
    """
    if not (user_query or "").strip():
        return {"mode": "chat", "count": 0}

    # Use your existing extractor:
    signals = extract_signals(user_query, pr.gaz, pr.alias_bi)

    # Cluster signals to avoid double-counting aliases
    clusters = _cluster_signals(signals)
    count = len(clusters)

    mode = "rag" if count >= threshold else "chat"

    return {"mode": mode, "count": count}
