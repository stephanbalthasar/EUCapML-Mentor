# mentor/booklet/retriever.py
# Lightweight retrievers for paragraphs and chapters.
# Uses keyword overlap by default; can switch to embeddings if you pass an encoder with .encode()

from __future__ import annotations

import numpy as np
import string


# --- tiny helper to keep acronyms like MAR, WpHG, ESMA, MiCA ---
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _tokenize_keep_acronyms(text: str) -> set[str]:
    """
    Minimal tokenizer for legal text:
    - strips punctuation
    - keeps normal words with length >= 4 (legacy behavior)
    - ALSO keeps acronyms: tokens with >= 2 uppercase letters (e.g., MAR, WpHG, ESMA), length 2..8
    """
    if not text:
        return set()
    toks = text.translate(_PUNCT_TABLE).split()
    out: set[str] = set()
    for tok in toks:
        # keep acronyms (e.g., MAR, WpHG, ESMA, MiCA)
        upper_count = sum(1 for ch in tok if ch.isupper())
        if 2 <= len(tok) <= 8 and upper_count >= 2:
            out.add(tok.lower())
            continue
        # normal words (legacy threshold: >3 chars)
        tl = tok.lower()
        if len(tl) > 3:
            out.add(tl)
    return out


class ParagraphRetriever:
    def __init__(self, paragraphs: list[dict], embedder=None):
        """
        paragraphs: [{'para_num', 'text', 'chapter_num', 'chapter_title'}, ...]
        embedder: optional, must implement .encode(list[str]) -> array
        """
        self.paragraphs = paragraphs
        self.embedder = embedder
        self._emb = None
        if embedder:
            texts = [p["text"] for p in paragraphs]
            self._emb = embedder.encode(texts, normalize_embeddings=True)

    def retrieve(self, query: str, top_k: int = 15, **kwargs) -> list[dict]:
        if not self.paragraphs:
            return []

        # Keyword-overlap fallback (now preserves acronyms like MAR / WpHG / ESMA / MiCA)
        if self._emb is None:
            q_words = _tokenize_keep_acronyms(query)
            scored: list[tuple[int, dict]] = []
            for p in self.paragraphs:
                p_words = _tokenize_keep_acronyms(p["text"])
                overlap = len(q_words & p_words)
                scored.append((overlap, p))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [p for score, p in scored[:top_k]]

        # Embedding similarity (unchanged)
        qv = self.embedder.encode([query], normalize_embeddings=True)[0]
        sims = np.dot(self._emb, qv)
        idx = np.argsort(sims)[::-1][:top_k]
        return [self.paragraphs[i] for i in idx]


class ChapterRetriever:
    def __init__(self, chapters: list[dict], embedder=None):
        """
        chapters: [{'chapter_num','title','text'}, ...]
        """
        self.chapters = chapters
        self.embedder = embedder
        self._emb = None
        if embedder:
            texts = [c["text"] for c in chapters]
            self._emb = embedder.encode(texts, normalize_embeddings=True)

    def retrieve_best(self, query: str):
        if not self.chapters:
            return None
        if self._emb is None:
            # (kept as-is; paragraph-level fix already addresses the core issue)
            q_words = set(w.lower() for w in query.split() if len(w) > 3)
            best, best_score = None, -1
            for c in self.chapters:
                c_words = set(c["text"].lower().split())
                sc = len(q_words & c_words)
                if sc > best_score:
                    best_score = sc
                    best = c
            return best
        qv = self.embedder.encode([query], normalize_embeddings=True)[0]
        sims = np.dot(self._emb, qv)
        best_idx = int(np.argmax(sims))
        return self.chapters[best_idx]
