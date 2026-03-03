# mentor/booklet/retriever.py
# Lightweight retrievers for paragraphs and chapters.
# Uses keyword overlap by default; can switch to embeddings if you pass an encoder with .encode()

import numpy as np

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

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.paragraphs:
            return []

        if self._emb is None:
            # keyword overlap fallback
            q_words = set(w.lower() for w in query.split() if len(w) > 3)
            scored = []
            for p in self.paragraphs:
                p_words = set(p["text"].lower().split())
                overlap = len(q_words & p_words)
                scored.append((overlap, p))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [p for score, p in scored[:top_k]]

        # embedding similarity
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

    def retrieve_best(self, query: str) -> dict | None:
        if not self.chapters:
            return None

        if self._emb is None:
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
