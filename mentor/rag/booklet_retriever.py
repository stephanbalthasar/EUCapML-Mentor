# -*- coding: utf-8 -*-
"""
Back-compat retrievers with optional JSONL-over-GitHub contents loading.

Path secrets (read from Streamlit secrets first, then environment):
  BOOKLET_REPO  -> e.g. "stephanbalthasar/EUCapML-Mentor-Content"
  BOOKLET_REF   -> e.g. "main"
  BOOKLET_PATH  -> e.g. "artifacts/booklet_index.jsonl"

Token priority (first present wins):
  1) GITHUB_TOKEN   (st.secrets / env)
  2) REPO_XPAT      (st.secrets / env)
  3) BOOKLET_TOKEN  (st.secrets / env)  # optional fallback

Optional tuning:
  BOOKLET_MIN_SIM -> similarity floor in [0,1], default 0.38

Behavior:
- If BOOKLET_REPO/REF/PATH are present, ParagraphRetriever loads the JSONL from:
      https://raw.githubusercontent.com/{BOOKLET_REPO}/{BOOKLET_REF}/{BOOKLET_PATH}
  using the token above if present.
- Otherwise, it falls back to legacy lexical retrieval over the passed-in
  paragraphs list (as used by your existing app).

Threshold is enforced INSIDE the retriever (no second gate in the chat layer).
"""

from __future__ import annotations
import os
import re
import io
import json
import math
import sys
from typing import List, Dict, Optional, Tuple

# optional deps (graceful fallback)
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except Exception:
    _HAS_BM25 = False

try:
    import numpy as np
except Exception as e:
    raise RuntimeError("This retriever requires numpy. Please `pip install numpy`.") from e

try:
    import requests
except Exception as e:
    raise RuntimeError("This retriever fetches JSONL via HTTP; please `pip install requests`.") from e

# Streamlit secrets (optional); we fall back to env if missing
try:
    import streamlit as st  # noqa
except Exception:
    st = None


# ----------------- simple text utils -----------------
_TOKEN_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+", re.UNICODE)
_EN_STOP = {
    "the","a","an","and","or","but","of","in","to","is","are","was","were","on","for",
    "with","as","by","at","from","that","this","it","be","been","will","would","can",
    "could","should","under","per","such"
}
_DE_STOP = {
    "der","die","das","und","oder","aber","nicht","mit","auf","aus","bei","durch","gegen",
    "ohne","unter","vom","zur","zum","gemäß","auch","sowie","daher","soweit","darüber",
    "hierzu","hiervon","hierfür","ist","sind","war","waren","einer","einem","einen","eines",
    "denn","doch","noch","schon"
}

def _tokenize(text: str, lang_hint: Optional[str] = None) -> List[str]:
    t = (text or "").lower()
    toks = _TOKEN_RE.findall(t)
    stop = _DE_STOP if lang_hint == "de" else _EN_STOP
    return [w for w in toks if w not in stop]

def _detect_lang_quick(s: str) -> str:
    s_low = (s or "").lower()
    if any(c in s_low for c in "äöüß"):
        return "de"
    toks = _TOKEN_RE.findall(s_low)
    if not toks:
        return "en"
    de = sum(1 for w in toks if w in _DE_STOP)
    return "de" if de / max(1, len(toks)) > 0.08 else "en"

def _normalize(v: np.ndarray) -> np.ndarray:
    if v.size == 0:
        return v
    vmin, vmax = float(v.min()), float(v.max())
    if math.isclose(vmax, vmin):
        return np.zeros_like(v)
    return (v - vmin) / (vmax - vmin)

def _l2norm_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


# ----------------- JSONL-backed hybrid retriever -----------------
class _BookletRetriever:
    """
    Internal: JSONL-backed hybrid retriever.
    Exposes .search(query, top_k, min_sim) and returns list[dict] with:
        {"text", "node_id", "anchor", "type", "score", "dense", "lexical", "lang", "links"}
    """

    def __init__(
        self,
        jsonl_url: str,                # raw.githubusercontent.com URL
        http_token: Optional[str] = None,
        include_types: Tuple[str, ...] = ("paragraph", "case_note", "footnote"),
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dense_weight: float = 0.55,
        lexical_weight: float = 0.45,
    ):
        self.jsonl_url = jsonl_url
        self.http_token = http_token
        self.include_types = include_types
        self.model_name = model_name
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight

        self.nodes: List[Dict] = []
        self.texts: List[str] = []
        self.langs: List[str] = []

        # dense
        self._st_model = None
        self._emb_matrix = None

        # lexical
        self._bm25 = None

        self._load_nodes()
        self._build_lexical()
        self._build_dense()

    # ---- loading ----
    def _load_nodes(self):
        headers = {"Authorization": f"token {self.http_token}"} if self.http_token else {}
        r = requests.get(self.jsonl_url, headers=headers, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Failed to fetch JSONL {r.status_code}: {self.jsonl_url}")

        stream = io.StringIO(r.text)
        for raw in stream:
            line = raw.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("type") not in self.include_types:
                continue
            text = (d.get("text") or "").strip()
            if not text:
                continue
            self.nodes.append(d)

        self.texts = [n["text"] for n in self.nodes]
        self.langs = [n.get("lang", _detect_lang_quick(n["text"])) for n in self.nodes]

    def _build_lexical(self):
        if not _HAS_BM25:
            sys.stderr.write("[retriever] rank-bm25 not installed → lexical disabled\n")
            self._bm25 = None
            return
        tokenized = [_tokenize(t, lang_hint=lang) for t, lang in zip(self.texts, self.langs)]
        self._bm25 = BM25Okapi(tokenized)

    def _build_dense(self):
        if not _HAS_ST:
            sys.stderr.write("[retriever] sentence-transformers not installed → dense disabled\n")
            self._st_model = None
            self._emb_matrix = None
            return
        self._st_model = SentenceTransformer(self.model_name)
        embs = self._st_model.encode(self.texts, batch_size=64, normalize_embeddings=False, show_progress_bar=False)
        embs = np.asarray(embs, dtype=np.float32)
        self._emb_matrix = _l2norm_rows(embs)

    # ---- search ----
    def search(self, query: str, top_k: int = 6, min_sim: float = 0.38) -> List[Dict]:
        if not query or not query.strip():
            return []

        dense_scores = np.zeros(len(self.texts), dtype=np.float32)
        if self._st_model is not None and self._emb_matrix is not None:
            q_vec = self._st_model.encode([query], normalize_embeddings=False)[0].astype(np.float32)
            q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-12)
            dense_scores = (self._emb_matrix @ q_vec + 1.0) / 2.0  # [-1,1] -> [0,1]

        lexical_scores = np.zeros(len(self.texts), dtype=np.float32)
        if self._bm25 is not None:
            toks = _tokenize(query, lang_hint=_detect_lang_quick(query))
            raw = np.array(self._bm25.get_scores(toks), dtype=np.float32)
            lexical_scores = _normalize(raw)

        alpha, beta = float(self.dense_weight), float(self.lexical_weight)
        if math.isclose(alpha + beta, 0.0):
            alpha, beta = 0.0, 1.0
        combined = alpha * dense_scores + beta * lexical_scores

        order = np.argsort(-combined)
        hits = []
        for i in order[: max(200, top_k * 5)]:
            score = float(combined[i])
            if score < float(min_sim):
                continue
            n = self.nodes[i]
            hits.append({
                "text": n["text"],
                "node_id": n.get("node_id",""),
                "anchor": n.get("anchor",""),
                "type": n.get("type","paragraph"),
                "score": score,
                "dense": float(dense_scores[i]),
                "lexical": float(lexical_scores[i]),
                "lang": n.get("lang","en"),
                "links": n.get("links",{}),
            })
            if len(hits) >= top_k:
                break
        return hits


# ----------------- Back-compat public classes -----------------
def _secret_or_env(key: str) -> Optional[str]:
    if st is not None:
        try:
            val = st.secrets.get(key)
            if val:
                return str(val)
        except Exception:
            pass
    return os.getenv(key)

def _build_raw_url_from_secrets() -> Optional[tuple[str, Optional[str]]]:
    """
    Returns (raw_url, token) if BOOKLET_REPO/REF/PATH are set in secrets or env,
    else None. Token priority: GITHUB_TOKEN -> REPO_XPAT -> BOOKLET_TOKEN.
    """
    repo = _secret_or_env("BOOKLET_REPO")
    ref  = _secret_or_env("BOOKLET_REF")
    path = _secret_or_env("BOOKLET_PATH")
    if not (repo and ref and path):
        return None
    raw_url = f"https://raw.githubusercontent.com/{repo}/{ref}/{path}"
    token   = (
        _secret_or_env("GITHUB_TOKEN")
        or _secret_or_env("REPO_XPAT")
        or _secret_or_env("BOOKLET_TOKEN")
    )
    return (raw_url, token)

class ParagraphRetriever:
    """
    Your existing app constructs this with INDEX["paragraphs"].
    With BOOKLET_REPO/REF/PATH set, this class loads the JSONL from your private repo
    (hybrid retrieval). Otherwise it falls back to BM25 over the provided list.
    """
    def __init__(self, paragraphs: List[Dict]):
        self._min_sim = float(_secret_or_env("BOOKLET_MIN_SIM") or "0.38")
        ru = _build_raw_url_from_secrets()

        self._hybrid: Optional[_BookletRetriever] = None
        self._bm25_only = None
        self._bm25_docs: List[str] = []

        if ru:
            raw_url, token = ru
            self._hybrid = _BookletRetriever(
                jsonl_url=raw_url,
                http_token=token,
                include_types=("paragraph","case_note","footnote"),
            )
        else:
            # Legacy lexical-only path over the provided paragraphs
            if not _HAS_BM25:
                sys.stderr.write("[retriever] rank-bm25 not installed; legacy lexical disabled\n")
            texts = []
            for p in (paragraphs or []):
                t = (p.get("text") if isinstance(p, dict) else str(p)).strip()
                if t:
                    texts.append(t)
            self._bm25_docs = texts
            if _HAS_BM25 and texts:
                self._bm25_only = BM25Okapi([_tokenize(t) for t in texts])

    def retrieve(self, query: str, top_k: int = 15) -> List[Dict]:
        """
        Returns list[dict] with {"text": "..."} to keep your prompt builder unchanged.
        """
        if self._hybrid:
            hits = self._hybrid.search(query=query, top_k=top_k, min_sim=self._min_sim)
            return [{"text": h["text"]} for h in hits]

        # Legacy lexical fallback
        if not query or not query.strip() or not self._bm25_only:
            return []
        toks = _tokenize(query)
        raw = np.array(self._bm25_only.get_scores(toks), dtype=np.float32)
        order = np.argsort(-raw)
        out = []
        for i in order[:top_k]:
            out.append({"text": self._bm25_docs[int(i)]})
        return out


class ChapterRetriever:
    """
    Kept for back-compat with imports. Not used when JSONL-backed retriever is active.
    """
    def __init__(self, chapters: List[Dict]):
        self._chapters = chapters or []

    def retrieve_best(self, query: str) -> Optional[Dict]:
        if not self._chapters or not query:
            return None
        toks = set(_tokenize(query))
        for ch in self._chapters:
            text = (ch.get("text") or "").lower()
            if any(tok in text for tok in toks):
                return {"chapter_num": ch.get("chapter_num"), "text": ch.get("text","")}
        return None
