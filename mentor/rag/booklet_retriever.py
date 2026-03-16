# -*- coding: utf-8 -*-
"""
Minimal legal retriever (Option B):
1) Extract "strong tokens" from the user query (e.g., Lafonta, MAR, Art. 7(1), C-628/13).
2) Exact substring filter (case-insensitive). If any hits -> return top-K by (#matches, shorter text).
3) Otherwise fall back to BM25 over the booklet JSONL.
4) If BM25 was used, optionally re-rank top candidates with a Sentence-Transformers model (CPU).
5) No hard similarity floor; never silently return zero if there are literal matches.

Source of truth: ONLY the remote JSONL configured via BOOKLET_REPO/REF/PATH.
No local fallback. If the JSONL cannot be fetched, we fail fast with a clear error.

Expected to be constructed as ParagraphRetriever(...), and used via .search(query, top_k,...)
to stay compatible with your ChatEngine (it prefers .search if present).

Environment / Streamlit secrets:
  BOOKLET_REPO  -> e.g. "stephanbalthasar/EUCapML-Mentor-Content"
  BOOKLET_REF   -> e.g. "main"
  BOOKLET_PATH  -> e.g. "artifacts/booklet_index.jsonl"
  (optional token priority: GITHUB_TOKEN -> REPO_XPAT -> BOOKLET_TOKEN)

Optional knobs (env/secrets):
  BOOKLET_DEVICE = "cpu" (default)  # for SentenceTransformer
"""

from __future__ import annotations

import io
import os
import re
import json
from typing import List, Dict, Optional, Tuple

import requests

# Optional libraries; degrade gracefully.
try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except Exception:
    _HAS_BM25 = False

try:
    from sentence_transformers import SentenceTransformer
    import numpy as _np
    _HAS_ST = True
except Exception:
    _HAS_ST = False
    _np = None  # type: ignore

# Streamlit secrets are convenient, but we also accept pure env.
try:
    import streamlit as st  # noqa
except Exception:
    st = None


# ------------------------------- small helpers -------------------------------

# Keep letters, digits, and legal punctuation used in citations (§, (), /, -).
# This tokenizer is used for BM25; it's intentionally simple and robust.
_TOKEN_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+(?:[()\-/§\d]*[A-Za-zÄÖÜäöüß\d]*)*|\d+[()\-/\d]*", re.UNICODE)

# Very lightweight stopwords to remove conversational fluff; legal abbreviations remain.
_STOP_EN = {"the", "a", "an", "and", "or", "but", "of", "in", "on", "for", "to", "is", "are", "was", "were", "be", "been", "can", "could", "would", "should", "with", "as", "by", "at", "from", "that", "this", "it", "about", "there", "anything", "tell", "me"}
_STOP_DE = {"der", "die", "das", "und", "oder", "aber", "mit", "ohne", "auf", "aus", "bei", "durch", "gegen", "unter", "vom", "zur", "zum", "gemäß", "ist", "sind", "war", "waren"}

def _secret_or_env(key: str) -> Optional[str]:
    if st is not None:
        try:
            val = st.secrets.get(key)
            if val:
                return str(val)
        except Exception:
            pass
    return os.getenv(key)

def _build_raw_url_from_secrets() -> Tuple[str, Optional[str]]:
    repo = _secret_or_env("BOOKLET_REPO")
    ref  = _secret_or_env("BOOKLET_REF")
    path = _secret_or_env("BOOKLET_PATH")
    if not (repo and ref and path):
        raise RuntimeError("Booklet retriever misconfigured: set BOOKLET_REPO, BOOKLET_REF, BOOKLET_PATH.")
    raw_url = f"https://raw.githubusercontent.com/{repo}/{ref}/{path}"
    token = (_secret_or_env("GITHUB_TOKEN")
             or _secret_or_env("REPO_XPAT")
             or _secret_or_env("BOOKLET_TOKEN"))
    return raw_url, token

def _fetch_jsonl(raw_url: str, token: Optional[str]) -> str:
    headers = {"Authorization": f"token {token}"} if token else {}
    r = requests.get(raw_url, headers=headers, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to fetch booklet JSONL (HTTP {r.status_code}).")
    return r.text

def _tokenize_legal(text: str, lang_hint: Optional[str] = None) -> List[str]:
    s = (text or "").lower()
    toks = [t for t in _TOKEN_RE.findall(s) if t]
    if lang_hint == "de":
        return [t for t in toks if t not in _STOP_DE]
    if lang_hint == "en":
        return [t for t in toks if t not in _STOP_EN]
    # No reliable lang detection needed; queries are short. Apply both lists lightly.
    return [t for t in toks if t not in _STOP_EN and t not in _STOP_DE]

def _strong_tokens(query: str) -> List[str]:
    """
    Extract "strong tokens":
      - capitalized tokens not at sentence start (simple heuristic),
      - ALL-CAPS abbreviations (e.g., MAR, ECJ),
      - tokens containing digits or legal punctuation: (), /, -, § (e.g., 7(1), C-628/13, §15),
    We keep them as raw substrings and use case-insensitive substring search.
    """
    q = query.strip()
    if not q:
        return []
    # Split on whitespace, keep punctuation within tokens (for citations).
    rough = re.findall(r"[^\s]+", q)
    strong: List[str] = []

    # Heuristics
    for i, tok in enumerate(rough):
        # Normalize fancy hyphens
        t = tok.replace("‑", "-").replace("–", "-").strip(".,;:!?()[]{}\"'“”‘’")
        if not t:
            continue
        # ALL-CAPS legal abbreviations (length >= 2)
        if re.fullmatch(r"[A-ZÄÖÜ]{2,}", t):
            strong.append(t)
            continue
        # Contains digits or legal punctuation -> likely a citation
        if re.search(r"[0-9§()/\-]", t):
            strong.append(t)
            continue
        # Capitalized non-initial proper noun (very light heuristic)
        if i > 0 and re.fullmatch(r"[A-ZÄÖÜ][a-zäöüß]+", t):
            strong.append(t)
            continue

    # Deduplicate, preserve order
    seen = set()
    out = []
    for t in strong:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


# ----------------------------- main retriever class -----------------------------

class ParagraphRetriever:
    """
    Minimal, deterministic retriever for your booklet JSONL.

    Usage:
        r = ParagraphRetriever(paragraphs_ignored)
        hits = r.search("ECJ decision in Lafonta", top_k=8)

    Returns: list[dict] with keys: text, score, rank, node_id, doc_id, type, anchor, breadcrumb, lang
    """

    def __init__(self, _paragraphs_ignored=None):
        # Load JSONL from remote only (explicit). No local fallback.
        raw_url, token = _build_raw_url_from_secrets()
        text = _fetch_jsonl(raw_url, token)

        self.nodes: List[Dict] = []
        self._texts_lower: List[str] = []

        for line in io.StringIO(text):
            s = line.strip()
            if not s:
                continue
            try:
                d = json.loads(s)
            except Exception:
                continue
            t = (d.get("text") or "").strip()
            if not t:
                continue
            # Accept all types that carry content; you can narrow if needed.
            dtype = d.get("type", "paragraph")
            if dtype not in {"paragraph", "case_note", "footnote", "heading"}:
                continue
            self.nodes.append(d)
            self._texts_lower.append(t.lower())

        if not self.nodes:
            raise RuntimeError("Booklet JSONL loaded but contained no usable nodes.")

        # Build BM25 index if library is available.
        self._bm25 = None
        if _HAS_BM25:
            corpus_tokens = [_tokenize_legal(n.get("text", "")) for n in self.nodes]
            self._bm25 = BM25Okapi(corpus_tokens)

        # Optional Sentence-Transformers model for semantic re-ranking
        self._st = None
        if _HAS_ST:
            device = _secret_or_env("BOOKLET_DEVICE") or "cpu"
            try:
                self._st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
            except Exception:
                self._st = None  # run lexical-only if it fails

    # ----------------------------- public API -----------------------------

    def search(self, query: str, top_k: int = 8, **_kwargs) -> List[Dict]:
        """
        Exact-match first; BM25 fallback; optional semantic re-rank on fallback.
        Never applies a hard similarity floor. Returns up to top_k results.
        """
        q = (query or "").strip()
        if not q:
            return []

        strong = _strong_tokens(q)
        # STEP 1: Exact substring filter (case-insensitive) if strong tokens exist.
        if strong:
            lower_tokens = [t.lower() for t in strong]
            exact_hits: List[Tuple[int, int]] = []  # (node_idx, matched_count)

            for i, txt_low in enumerate(self._texts_lower):
                m = sum(1 for tok in lower_tokens if tok in txt_low)
                if m > 0:
                    exact_hits.append((i, m))

            if exact_hits:
                # Sort: more token matches first, then shorter text
                exact_hits.sort(key=lambda x: (-x[1], len(self.nodes[x[0]].get("text", ""))))
                return self._package_hits([idx for idx, _m in exact_hits[:top_k]], query=q, scores=None)

        # STEP 2: BM25 fallback
        if self._bm25 is None:
            # No BM25 installed; do a trivial keyword scan as last resort.
            toks = _tokenize_legal(q)
            if not toks:
                return []
            scored: List[Tuple[int, int]] = []  # (node_idx, matches)
            for i, txt_low in enumerate(self._texts_lower):
                m = sum(1 for tok in toks if tok in txt_low)
                if m > 0:
                    scored.append((i, m))
            if not scored:
                return []
            scored.sort(key=lambda x: (-x[1], len(self.nodes[x[0]].get("text", ""))))
            return self._package_hits([i for i, _m in scored[:top_k]], query=q, scores=None)

        # Proper BM25
        q_tokens = _tokenize_legal(q)
        bm25_scores = self._bm25.get_scores(q_tokens)
        # Pick top-N candidates (cap to keep semantic re-rank fast)
        import numpy as _np_local  # lightweight local import
        arr = _np_local.asarray(bm25_scores)
        if arr.size == 0:
            return []
        # Select top 50 indices (or fewer)
        N = min(50, max(top_k * 4, 20))
        top_idx = _np_local.argsort(-arr)[:N]
        cand_idx = [int(i) for i in top_idx if arr[int(i)] > 0]
        if not cand_idx:
            return []

        # Optional semantic re-ranking on candidates
        if self._st is not None and len(cand_idx) > 1:
            # Normalize BM25 to 0..1
            bm = arr[cand_idx]
            bmin, bmax = float(bm.min()), float(bm.max())
            bm_norm = (bm - bmin) / (bmax - bmin + 1e-12) if bmax > bmin else _np_local.zeros_like(bm)

            # Compute dense similarity for candidates
            q_vec = self._st.encode([q], normalize_embeddings=True, show_progress_bar=False)[0]
            cand_texts = [self.nodes[i].get("text", "") for i in cand_idx]
            C = self._st.encode(cand_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
            dense = (C @ q_vec)  # cosine in [-1,1]
            dense = (dense + 1.0) / 2.0  # -> [0,1]

            # Blend with a simple fixed weight (lexical-first)
            final = 0.7 * bm_norm + 0.3 * dense
            order = _np_local.argsort(-final)
            ranked = [cand_idx[int(i)] for i in order[:top_k]]
            # Pass blended scores back (optional)
            scores = [float(final[int(i)]) for i in order[:top_k]]
            return self._package_hits(ranked, query=q, scores=scores)

        # No semantic model -> return BM25 top-K
        # Note: sort BM25 top indices by score (descending) and limit to top_k
        top_for_k = _np_local.argsort(-arr)[:top_k]
        ranked = [int(i) for i in top_for_k if arr[int(i)] > 0]
        return self._package_hits(ranked, query=q, scores=None)

    # ----------------------------- packaging -----------------------------

    def _package_hits(self, indices: List[int], query: str, scores: Optional[List[float]]) -> List[Dict]:
        out: List[Dict] = []
        for rank, i in enumerate(indices, start=1):
            n = self.nodes[i]
            item = {
                "text": n.get("text", ""),
                "score": (scores[rank - 1] if scores and rank - 1 < len(scores) else None),
                "rank": rank,
                "node_id": n.get("node_id"),
                "doc_id": n.get("doc_id"),
                "type": n.get("type"),
                "anchor": n.get("anchor"),
                "breadcrumb": n.get("breadcrumb"),
                "lang": n.get("lang", None),
                "links": n.get("links", {}),
            }
            out.append(item)
        return out
