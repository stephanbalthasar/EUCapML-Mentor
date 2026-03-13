# mentor/booklet/retriever.py
# Lightweight retrievers for paragraphs and chapters.
# Default: acronym-aware lexical scoring; switches to embeddings if an encoder with .encode() is provided.
# NEW: Built‑in relevance gate so callers receive only meaningful hits (or [] if nothing is relevant).

from __future__ import annotations

from typing import List, Dict, Tuple, Optional
import numpy as np
import string


import re  # NEW

# --- Normalization helpers ----------------------------------------------------
def _normalize_text_basic(s: str) -> str:
    """Lowercase and normalize hyphens/dashes to spaces for simple phrase checks."""
    return re.sub(r"[\u2010\u2011\u2012\u2013\u2014\-]+", " ", (s or "").lower())

def _extract_query_units(q: str) -> tuple[set[str], list[str]]:
    """
    Return:
      - Q_tokens: set of normalized unigrams (len >= 3)
      - Q_phrases: list of normalized bigrams and trigrams (order preserved)
    """
    qn = _normalize_text_basic(q)
    toks = [t for t in qn.split() if len(t) >= 3]
    tokens = set(toks)
    phrases: list[str] = []
    for n in (2, 3):
        for i in range(len(toks) - n + 1):
            phrases.append(" ".join(toks[i : i + n]))
    return tokens, phrases

# --- Citation / docket pattern detectors -------------------------------------
_DOCKET_RE = re.compile(r"\b[CTF]\s*[-–]?\s*\d{1,4}\s*/\s*\d{2}\b", re.I)
_ART_NUM_RE = re.compile(r"(?:art(?:ikel)?|article)\s*\.?\s*(\d+[a-z]?)", re.I)
_PARA_SIGN_RE = re.compile(r"§\s*(\d+[a-z]?)", re.I)

def _extract_citation_patterns(q: str) -> list[re.Pattern]:
    """Build concrete regexes only for patterns present in the query."""
    patterns: list[re.Pattern] = []
    if _DOCKET_RE.search(q or ""):
        patterns.append(_DOCKET_RE)
    for num in _ART_NUM_RE.findall(q or ""):
        patterns.append(re.compile(rf"\b(?:art(?:ikel)?|article)\s*\.?\s*{re.escape(num)}\b", re.I))
    for num in _PARA_SIGN_RE.findall(q or ""):
        patterns.append(re.compile(rf"§\s*{re.escape(num)}\b", re.I))
    return patterns

# --- Exact‑match checks used by the Q<4 gate ----------------------------------
def _contains_any_phrase(text: str, phrases: list[str]) -> bool:
    if not phrases:
        return False
    t = _normalize_text_basic(text)
    return any(ph and ph in t for ph in phrases)

def _contains_all_tokens(text: str, tokens: set[str]) -> bool:
    if not tokens:
        return False
    # Reuse your acronym‑aware tokenizer so MAR/WpHG/MiCA survive.
    t_words = _tokenize_keep_acronyms((text or "").lower())
    return tokens.issubset(t_words)

def _matches_any_pattern(text: str, patterns: list[re.Pattern]) -> bool:
    if not patterns:
        return False
    return any(p.search(text or "") for p in patterns)

# --- Sentence-Transformers adapter (optional embedder) -----------------------
# Enables embedding mode for ParagraphRetriever by wrapping any ST model.
# Model examples:
#   - 'paraphrase-multilingual-mpnet-base-v2' (multilingual, 768-d)  [recommended for DE/EN]
#   - 'all-MiniLM-L6-v2' (English, 384-d, very fast)
# See docs / model cards for details.  (Hugging Face / SBERT)
try:
    from sentence_transformers import SentenceTransformer  # pip install sentence-transformers
except Exception:
    SentenceTransformer = None

class STEmbedder:
    """
    Thin adapter so our retriever can use Sentence-Transformers models.

    Parameters
    ----------
    model_name : str
        Any Sentence-Transformers checkpoint, e.g.:
        'paraphrase-multilingual-mpnet-base-v2' (multilingual, 768-d)
        'all-MiniLM-L6-v2' (English, 384-d)

    device : str
        'cpu' (default) or a CUDA device string like 'cuda' / 'cuda:0' if available.

    normalize_by_model : bool
        If True, pass normalize_embeddings=True into model.encode().
        Regardless of this flag, we still L2-normalize defensively if requested
        by the caller via normalize_embeddings=True.

    Notes
    -----
    - The adapter exposes .encode(list[str], normalize_embeddings=True|False) -> np.ndarray
      which is exactly what ParagraphRetriever expects.
    """
    def __init__(self, model_name: str, device: str = "cpu", normalize_by_model: bool = True):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install -U sentence-transformers"
            )
        self._model = SentenceTransformer(model_name, device=device)
        self._normalize_by_model = normalize_by_model

    def encode(self, texts, normalize_embeddings: bool = True):
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        # Some ST versions support normalize_embeddings directly; we pass it through.
        try:
            vecs = self._model.encode(
                texts,
                batch_size=64,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=(normalize_embeddings and self._normalize_by_model),
            )
        except TypeError:
            # Older ST versions may not expose normalize_embeddings; fall back to raw + manual L2
            vecs = self._model.encode(
                texts,
                batch_size=64,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        vecs = vecs.astype(np.float32, copy=False)
        if normalize_embeddings:
            # Defensive L2-normalization (idempotent if already normalized)
            denom = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
            vecs = vecs / denom
        return vecs
# -----------------------------------------------------------------------------


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
        embedder: optional, must implement .encode(list[str], ...) -> np.ndarray
                  If provided, we build normalized paragraph embeddings for cosine similarity.
        """
        self.paragraphs = paragraphs
        self.embedder = embedder
        self._emb = None
        if embedder:
            texts = [p.get("text", "") for p in paragraphs]
            # Expectation: the encoder supports normalize_embeddings=True.
            # If not, we will normalize below defensively.
            self._emb = embedder.encode(texts, normalize_embeddings=True)

    def retrieve(
        self,
        query: str,
        top_k: int = 15,
        *,
        # relevance gate (defaults mirror the supporting_sources_selector thresholds)
        gate: bool = True,
        min_abs: Optional[float] = None,
        min_gap: Optional[float] = None,
        floor: Optional[float] = None,
        require_anchor: bool = True,
        **kwargs,
    ) -> list[dict]:
        """
        Return up to top_k paragraph dicts ranked by relevance to `query`.

        Scoring:
          - If an embedder is available: cosine similarity (paragraphs pre-encoded).
          - Otherwise: acronym-aware lexical score (binary-cosine on tokens).

        Gate (when gate=True, default):
          - Determine mode from scoring (embed vs. lex).
          - Apply absolute + relative thresholds to the TOP score; if weak, return [].
          - Keep only items above a per-item floor.
          - require_anchor=True: each kept paragraph must share at least 1 token with the query.

        Back-compat:
          - Set gate=False to get legacy "raw top_k" behavior with no filtering.

        Parameters
        ----------
        query : str
        top_k : int
        gate : bool
        min_abs, min_gap, floor : Optional[float]
            Override thresholds if you need custom tuning per-call.
        require_anchor : bool
            If True (default), require at least 1 shared token with the query.

        Returns
        -------
        list[dict]
        """
        if not self.paragraphs:
            return []
        if not (query or "").strip():
            # Empty/whitespace query yields no meaningful retrieval
            return []
        # ------------------------------
        # A0) Query informativeness gate
        # ------------------------------
        Q_tokens, Q_phrases = _extract_query_units(query)
        Q_long = {t for t in Q_tokens if len(t) >= 4}  # ignore tiny tokens in rule (2)
        citation_patterns = _extract_citation_patterns(query)

        if len(Q_tokens) < 4:
            # Under-informative query: allow ONLY exact-match candidates (Rules 1–3)
            cand_idxs: list[int] = []
            for i, p in enumerate(self.paragraphs):
                text = p.get("text", "")
                rule1 = _contains_any_phrase(text, Q_phrases)                   # (1) phrase match
                rule2 = (len(Q_long) > 0) and _contains_all_tokens(text, Q_long) # (2) all long tokens present
                rule3 = (len(citation_patterns) > 0) and _matches_any_pattern(text, citation_patterns)  # (3) citation/docket
                if rule1 or rule2 or rule3:
                    cand_idxs.append(i)

            if not cand_idxs:
                return []  # nothing literal -> no booklet chunks

            # Rank ONLY the candidate set (embed or lexical), then run the usual thresholds.
            if mode == "embed":
                # Build normalized query vector once
                qv = self.embedder.encode([query], normalize_embeddings=True)[0]
                qv = qv / (np.linalg.norm(qv) + 1e-12)
                P = self._emb / (np.linalg.norm(self._emb, axis=1, keepdims=True) + 1e-12)
                sims = np.dot(P[cand_idxs], qv)  # cosine similarities for candidates
                order = np.argsort(sims)[::-1]
                ranked = [(float(sims[j]), self.paragraphs[cand_idxs[j]]) for j in order]
            else:
                # Lexical score limited to candidates
                scored: list[tuple[float, dict]] = []
                for i in cand_idxs:
                    p = self.paragraphs[i]
                    p_words = _tokenize_keep_acronyms(p.get("text", ""))
                    denom = max(1.0, (len(q_words) * len(p_words)) ** 0.5)
                    score = len(q_words & p_words) / denom
                    scored.append((score, p))
                ranked = sorted(scored, key=lambda x: x[0], reverse=True)

            # ---- apply your existing thresholds on the ranked list ----
            scores = [s for s, _ in ranked] or [0.0]
            top = float(scores[0])
            med = float(np.median(scores))
            if mode == "embed":
                _min_abs = 0.28 if min_abs is None else float(min_abs)
                _min_gap = 0.08 if min_gap is None else float(min_gap)
                _floor   = 0.20 if floor   is None else float(floor)
            else:
                _min_abs = 0.14 if min_abs is None else float(min_abs)
                _min_gap = 0.05 if min_gap is None else float(min_gap)
                _floor   = 0.10 if floor   is None else float(floor)

            if (top < _min_abs) or ((top - med) < _min_gap):
                return []

            filtered: list[dict] = []
            for s, p in ranked:
                if s < _floor:
                    continue
                filtered.append(p)
                if len(filtered) == min(5, top_k):   # <= TOP-5 CAP FOR Q<4
                    break
            return filtered
        
        # ------------------------------
        # A) Scoring (embed -> lexical)
        # ------------------------------
        mode = "embed" if self._emb is not None else "lex"
        q_words = _tokenize_keep_acronyms(query)

        if mode == "embed":
            # Embedding cosine similarity
            qv = self.embedder.encode([query], normalize_embeddings=True)[0]
            # Some encoders may ignore normalize_embeddings; normalize defensively.
            q_norm = np.linalg.norm(qv) + 1e-12
            qv = qv / q_norm

            P = self._emb
            if P is None:
                # Fallback if embeddings were not built for some reason
                mode = "lex"
            else:
                # Ensure paragraph vectors are normalized (just in case).
                P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
                sims = np.dot(P, qv)  # cosine similarities
                order = np.argsort(sims)[::-1]
                ranked: list[tuple[float, dict]] = [(float(sims[i]), self.paragraphs[i]) for i in order]
        if mode == "lex":
            # Lexical fallback: acronym-aware binary-cosine
            scored: list[tuple[float, dict]] = []
            for p in self.paragraphs:
                p_words = _tokenize_keep_acronyms(p.get("text", ""))
                denom = max(1.0, (len(q_words) * len(p_words)) ** 0.5)
                score = len(q_words & p_words) / denom
                scored.append((score, p))
            scored.sort(key=lambda x: x[0], reverse=True)
            ranked = scored

        # ----------------------------------------
        # B) Optional relevance gate (embed/lex)
        # ----------------------------------------
        if not gate:
            # Legacy behavior: return top_k as-is
            return [p for s, p in ranked[:top_k]]

        if not ranked:
            return []

        scores = [s for s, _ in ranked]
        top = float(scores[0])
        med = float(np.median(scores))

        # Defaults mirror the selector thresholds (kept conservative)
        if mode == "embed":
            # If caller didn't pass overrides, use these:
            if min_abs is None:
                min_abs = 0.28
            if min_gap is None:
                min_gap = 0.08
            if floor is None:
                floor = 0.20
        else:
            if min_abs is None:
                min_abs = 0.14
            if min_gap is None:
                min_gap = 0.05
            if floor is None:
                floor = 0.10

        # Absolute + relative gates: if the best match is weak, return no booklet context
        if (top < float(min_abs)) or ((top - med) < float(min_gap)):
            return []

        # Distinctive-token anchor: require at least 1 shared token with query
        def _passes_anchor(p_text: str) -> bool:
            if not require_anchor:
                return True
            if not q_words:
                return True
            return len(q_words & _tokenize_keep_acronyms(p_text)) >= 1

        # Keep only items above the per-item floor, honoring anchor, then clamp to top_k
        filtered: list[dict] = []
        for s, p in ranked:
            if s < float(floor):
                continue
            if not _passes_anchor(p.get("text", "")):
                continue
            filtered.append(p)
            if len(filtered) == top_k:
                break

        return filtered


class ChapterRetriever:
    def __init__(self, chapters: list[dict], embedder=None):
        """
        chapters: [{'chapter_num','title','text'}, ...]
        """
        self.chapters = chapters
        self.embedder = embedder
        self._emb = None
        if embedder:
            texts = [c.get("text", "") for c in chapters]
            self._emb = embedder.encode(texts, normalize_embeddings=True)

    def retrieve_best(self, query: str):
        if not self.chapters:
            return None
        if self._emb is None:
            # (kept as-is; paragraph-level gate already addresses the core issue)
            q_words = set(w.lower() for w in query.split() if len(w) > 3)
            best, best_score = None, -1
            for c in self.chapters:
                c_words = set((c.get("text", "")).lower().split())
                sc = len(q_words & c_words)
                if sc > best_score:
                    best_score = sc
                    best = c
            return best
        qv = self.embedder.encode([query], normalize_embeddings=True)[0]
        qv = qv / (np.linalg.norm(qv) + 1e-12)
        P = self._emb / (np.linalg.norm(self._emb, axis=1, keepdims=True) + 1e-12)
        sims = np.dot(P, qv)
        best_idx = int(np.argmax(sims))
        return self.chapters[best_idx]


def fetch_booklet_chunks_for_prompt(
    retriever,
    query: str,
    *,
    top_k: int = 15,
    truncate_chars: Optional[int] = None,
) -> Tuple[List[Dict], List[str]]:
    """
    Helper to:
      - call retriever.retrieve(query, top_k=..., gate=True)
      - normalize to a list[str] for the prompt
      - optionally truncate long paragraphs

    Returns:
      hits           : original list[dict] items from the retriever
      booklet_chunks : list[str] derived from 'text' fields (optionally truncated)
    """
    hits: List[Dict] = []
    try:
        if hasattr(retriever, "retrieve"):
            try:
                # Use gated retrieval by default to avoid irrelevant noise in prompts
                hits = retriever.retrieve(query, top_k=top_k, gate=True) or []
            except TypeError:
                # supports retrievers that use named arguments
                hits = retriever.retrieve(query=query, top_k=top_k, gate=True) or []
    except Exception:
        hits = []

    chunks: List[str] = []
    for h in hits:
        t = h.get("text") if isinstance(h, dict) else str(h)
        if not t:
            continue
        if truncate_chars and len(t) > truncate_chars:
            t = t[:truncate_chars] + "…"
        chunks.append(t)

    return hits, chunks
