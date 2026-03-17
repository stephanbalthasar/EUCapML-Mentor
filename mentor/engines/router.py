# app/router2.py

from __future__ import annotations
from typing import Dict, Any, List

# Import the retriever and its signal extractor
from mentor.rag.booklet_retriever import extract_signals, Gazetteers, build_corpus_auto_alias
from mentor.rag.booklet_retriever import ParagraphRetriever  # only used to load gazetteers & corpus

# You only need ONE global retriever instance to access gazetteers + alias maps
# This avoids reloading files for every query.
_retriever = ParagraphRetriever()   # loads gazetteers, booklet corpus, auto-aliases
_gaz = _retriever.gaz
_auto_alias = _retriever.alias_bi   # merged alias map (gazetteers + auto-aliases)


def route(user_query: str) -> Dict[str, Any]:
    """
    Signal-based router:
    - Use extract_signals() to detect legal signals in the query.
    - If >2 signals → RAG
    - If exactly 2 signals AND at least one is a case_name/case_no → RAG
    - Else → Chat mode.
    """

    if not user_query or not user_query.strip():
        return {"mode": "chat", "count": 0}

    # 1. Extract signals using the robust RAG extractor
    signals: List[Dict[str, Any]] = extract_signals(
        user_query,
        gaz=_gaz,
        corpus_auto_alias=_auto_alias,
    )

    n = len(signals)
    has_case = any(s["type"] in ("case_name", "case_no") for s in signals)

    # 2. Routing rules
    if n > 2:
        return {"mode": "rag", "count": n}

    if n == 2 and has_case:
        return {"mode": "rag", "count": n}

    return {"mode": "chat", "count": n}
