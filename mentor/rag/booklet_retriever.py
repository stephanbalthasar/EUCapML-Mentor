# -*- coding: utf-8 -*-
"""
Plain-vanilla booklet retriever:
- Loads nodes from booklet_index.jsonl
- Builds a hybrid retriever: lexical (BM25) + dense (embeddings)
- Ranks chunks and returns top-k above a similarity threshold.

Usage:
    retr = BookletRetriever(jsonl_path="artifacts/booklet_index.jsonl")
    results = retr.search(query="What are issuer obligations under MAR Art. 17?", top_k=6, min_sim=0.38)

Returned item per hit:
{
  "node_id": str,
  "type": "paragraph|case_note|footnote|section",
  "anchor": str,
  "text": str,
  "score": float,      # combined score in [0,1]
  "dense": float,      # cosine normalized to [0,1]
  "lexical": float,    # BM25 normalized to [0,1]
  "lang": "en|de",
  "links": dict
}
"""

from __future__ import annotations
import json, os, re, hashlib, math, sys
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np

# --- Optional embedding backend (graceful fallback) ---
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False

# --- Lexical retrieval (BM25) ---
try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except Exception:
    _HAS_BM25 = False


# ------------------------- Helpers -------------------------

_EN_STOP = {
    "the","a","an","and","or","but","of","in","to","is","are","was","were",
    "on","for","with","as","by","at","from","that","this","it","be","been",
    "will","would","can","could","should","under","per","such"
}
_DE_STOP = {
    "der","die","das","und","oder","aber","nicht","mit","auf","aus","bei",
    "durch","gegen","ohne","unter","vom","zur","zum","gemäß","auch","sowie",
    "daher","soweit","darüber","hierzu","hiervon","hierfür","ist","sind",
    "war","waren","einer","einem","einen","eines","denn","doch","noch","schon"
}

TOKEN_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+", re.UNICODE)

def _lang_stop(lang: str):
    return _DE_STOP if lang == "de" else _EN_STOP

def _simple_tokenize(text: str, lang_hint: Optional[str] = None) -> List[str]:
    t = text.lower()
    toks = TOKEN_RE.findall(t)
    stop = _lang_stop(lang_hint or "en")
    return [w for w in toks if w not in stop]

def _detect_lang_quick(s: str) -> str:
    s_low = s.lower()
    if any(c in s_low for c in "äöüß"):
        return "de"
    # quick heuristic by token overlap
    toks = TOKEN_RE.findall(s_low)
    if not toks:
        return "en"
    de = sum(1 for w in toks if w in _DE_STOP)
    return "de" if de / max(1, len(toks)) > 0.08 else "en"

def _normalize(v: np.ndarray) -> np.ndarray:
    # min-max normalize to [0,1] across a vector; avoid division by zero
    if v.size == 0:
        return v
    vmin, vmax = float(v.min()), float(v.max())
    if math.isclose(vmax, vmin):
        return np.zeros_like(v)
    return (v - vmin) / (vmax - vmin)

def _cosine_sim_matrix(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    # expects both query_vec and doc_matrix already L2-normalized
    return np.clip(doc_matrix @ query_vec, -1.0, 1.0)

def _stable_hash(items: List[str]) -> str:
    h = hashlib.md5()
    for s in items:
        h.update(s.encode("utf-8"))
    return h.hexdigest()


# ------------------------- Data model -------------------------

@dataclass
class Node:
    node_id: str
    type: str
    anchor: str
    text: str
    links: Dict
    lang: str

    @staticmethod
    def from_json(d: Dict) -> "Node":
        return Node(
            node_id=d["node_id"],
            type=d["type"],
            anchor=d.get("anchor",""),
            text=d.get("text",""),
            links=d.get("links",{}),
            lang=d.get("lang","en"),
        )


# ------------------------- Retriever -------------------------

class BookletRetriever:
    def __init__(
        self,
        jsonl_path: str,
        include_types: Tuple[str, ...] = ("paragraph", "case_note", "footnote"),  # sections usually not helpful
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embed_cache_dir: Optional[str] = None,
        dense_weight: float = 0.55,
        lexical_weight: float = 0.45,
    ):
        """
        :param jsonl_path: path to booklet_index.jsonl
        :param include_types: which node types to index for retrieval
        :param model_name: sentence-transformers model (384d, fast, free)
        :param embed_cache_dir: where to cache embeddings; default next to jsonl
        :param dense_weight: weight of dense cosine component in [0,1]
        :param lexical_weight: weight of BM25 component in [0,1]
        """
        self.jsonl_path = jsonl_path
        self.include_types = tuple(include_types)
        self.model_name = model_name
        self.embed_cache_dir = embed_cache_dir or os.path.dirname(os.path.abspath(jsonl_path)) or "."
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight

        self.nodes: List[Node] = []
        self.corpus_texts: List[str] = []
        self.corpus_langs: List[str] = []
        self.ids: List[str] = []

        self._bm25 = None
        self._emb_model: Optional[SentenceTransformer] = None
        self._emb_matrix: Optional[np.ndarray] = None  # shape: (N, D), L2-normalized

        self._load_nodes()
        self._build_lexical()
        self._build_dense()

    # ---------- loading & indexing ----------

    def _load_nodes(self):
        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(f"JSONL not found: {self.jsonl_path}")

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                t = d.get("type","")
                if t not in self.include_types:
                    continue
                n = Node.from_json(d)
                if not n.text.strip():
                    continue
                self.nodes.append(n)

        # Keep a simple corpus view
        self.corpus_texts = [n.text for n in self.nodes]
        self.corpus_langs = [n.lang for n in self.nodes]
        self.ids = [n.node_id for n in self.nodes]

        if not self.nodes:
            raise RuntimeError("No nodes loaded for the requested include_types. Check your JSONL path and types.")

    def _build_lexical(self):
        if not _HAS_BM25:
            sys.stderr.write("[booklet_retriever] rank-bm25 not installed → lexical disabled (dense only if available).\n")
            self._bm25 = None
            return

        tokenized_docs = [
            _simple_tokenize(txt, lang_hint=lang) for txt, lang in zip(self.corpus_texts, self.corpus_langs)
        ]
        self._bm25 = BM25Okapi(tokenized_docs)

    def _build_dense(self):
        if not _HAS_ST:
            sys.stderr.write("[booklet_retriever] sentence-transformers not installed → dense disabled (lexical only).\n")
            self._emb_model = None
            self._emb_matrix = None
            return

        # model lazy-load on demand (faster init)
        self._emb_model = SentenceTransformer(self.model_name)

        # cache path based on jsonl + model hash
        sig = _stable_hash(self.ids + [self.model_name])
        cache_path = os.path.join(self.embed_cache_dir, f"embeddings_{sig}.npy")

        if os.path.exists(cache_path):
            embs = np.load(cache_path)
            self._emb_matrix = self._l2_normalize_rows(embs)
            return

        # compute and cache
        embs = self._emb_model.encode(
            self.corpus_texts, batch_size=64, normalize_embeddings=False, show_progress_bar=False
        )
        embs = np.asarray(embs, dtype=np.float32)
        self._emb_matrix = self._l2_normalize_rows(embs)

        try:
            np.save(cache_path, embs)
        except Exception as e:
            sys.stderr.write(f"[booklet_retriever] Warning: failed to save embeddings cache: {e}\n")

    @staticmethod
    def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / norms

    # ---------- public API ----------

    def search(
        self,
        query: str,
        top_k: int = 6,
        min_sim: float = 0.38,
        return_sections: bool = False,
    ) -> List[Dict]:
        """
        Hybrid retrieval: BM25 + cosine. Returns up to top_k items with combined score >= min_sim.
        :param query: user query (you can concatenate prior chat turns before passing it here)
        :param top_k: max number of items to return
        :param min_sim: threshold on combined score in [0,1]
        :param return_sections: include 'section' nodes if they were indexed (default: we exclude them at init)
        """
        if not query or not query.strip():
            return []

        # Dense
        dense_scores = np.zeros(len(self.corpus_texts), dtype=np.float32)
        if self._emb_model is not None and self._emb_matrix is not None:
            q_vec = self._emb_model.encode([query], normalize_embeddings=False)[0].astype(np.float32)
            q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-12)
            dense_raw = _cosine_sim_matrix(q_vec, self._emb_matrix)  # [-1,1]
            # normalize to [0,1] for fusion
            dense_scores = (dense_raw + 1.0) / 2.0
        else:
            dense_scores = np.zeros(len(self.corpus_texts), dtype=np.float32)

        # Lexical
        lexical_scores = np.zeros(len(self.corpus_texts), dtype=np.float32)
        if self._bm25 is not None:
            q_lang = _detect_lang_quick(query)
            q_tokens = _simple_tokenize(query, lang_hint=q_lang)
            raw = np.array(self._bm25.get_scores(q_tokens), dtype=np.float32)
            lexical_scores = _normalize(raw)

        # Combine
        alpha = float(self.dense_weight)
        beta = float(self.lexical_weight)
        if math.isclose(alpha + beta, 0.0):
            alpha = 0.0; beta = 1.0
        combined = alpha * dense_scores + beta * lexical_scores

        # Rank and filter
        idx = np.argsort(-combined)  # descending
        hits = []
        for i in idx[: max(200, top_k * 5)]:  # small overfetch → better thresholding
            score = float(combined[i])
            if score < min_sim:
                continue
            n = self.nodes[i]
            if (not return_sections) and n.type == "section":
                continue
            hits.append({
                "node_id": n.node_id,
                "type": n.type,
                "anchor": n.anchor,
                "text": n.text,
                "score": score,
                "dense": float(dense_scores[i]),
                "lexical": float(lexical_scores[i]),
                "lang": n.lang,
                "links": n.links,
            })
            if len(hits) >= top_k:
                break
        return hits

    # Convenience: allows swapping to lexical-only or dense-only easily
    def set_weights(self, dense_weight: float = 0.55, lexical_weight: float = 0.45):
        self.dense_weight = float(dense_weight)
        self.lexical_weight = float(lexical_weight)
