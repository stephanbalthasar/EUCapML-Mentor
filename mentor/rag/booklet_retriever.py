# -*- coding: utf-8 -*-
"""
Gazetteers local, booklet remote (unchanged behavior for JSONL) + robust token discovery.

Local (same folder):
- gazetteer_concepts.txt
- gazetteer_cases.txt
- gazetteer_aliases.txt

Remote booklet (same as before):
- raw.githubusercontent.com first
- GitHub Contents API fallback with token

Token discovery (in order):
1) ParagraphRetriever(token=...)  [optional override]
2) env REPO_XPAT
3) env GITHUB_TOKEN
4) env REPO_XPAT_FILE -> read file content
5) local files next to this module: REPO_XPAT.txt, .repo_xpat, .xpat

Diagnostics:
- set RAG_DEBUG=1 to log which source provided the token (not the token itself).
"""

from __future__ import annotations
import json
import os
import re
import time
import difflib
from typing import Dict, List, Optional, Tuple, Set, Iterable
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# -----------------------------------------------------------------------------
# Config (same defaults as before for the booklet)
# -----------------------------------------------------------------------------
_DEFAULT_REPO = os.getenv("BOOKLET_REPO", "stephanbalthasar/b2-eucapml-content")
_DEFAULT_REF  = os.getenv("BOOKLET_REF",  "main")
_BOOKLET_PATH = os.getenv("BOOKLET_PATH", "artifacts/booklet_index.jsonl")

_GITHUB_API_TMPL = "https://api.github.com/repos/{repo}/contents/{path}?ref={ref}"
_GITHUB_RAW_TMPL = "https://raw.githubusercontent.com/{repo}/{ref}/{path}"

# Local gazetteers (beside this file)
HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_CONCEPTS = os.path.join(HERE, "gazetteer_concepts.txt")
LOCAL_CASES    = os.path.join(HERE, "gazetteer_cases.txt")
LOCAL_ALIASES  = os.path.join(HERE, "gazetteer_aliases.txt")

# Optional local secret fallbacks (if you mount/copy a secret file at build/deploy time)
LOCAL_TOKEN_CANDIDATES = [
    os.getenv("REPO_XPAT_FILE", "").strip() or None,  # explicit path via env
    os.path.join(HERE, "REPO_XPAT.txt"),
    os.path.join(HERE, ".repo_xpat"),
    os.path.join(HERE, ".xpat"),
]

# Fuzzy thresholds
_SHORT_SNAP   = 0.92
_LONG_SNAP    = 0.85
_SNAP_MARGIN  = 0.05
_FUZZY_ACCEPT = 0.82

# Scoring weights
W_STRUCTURED = 3.0
W_GAZ_EXACT  = 2.5
W_FUZZY      = 0.6
W_COOCCUR    = 0.4

def _dbg(msg: str):
    if os.getenv("RAG_DEBUG", "0") == "1":
        print(f"[booklet_retriever] {msg}")

# =============================================================================
# HTTP helpers (for the booklet)
# =============================================================================

def _http_get(url: str, headers: Dict[str, str], retries: int = 3, backoff: float = 0.75) -> Tuple[int, bytes]:
    last_exc = None
    for attempt in range(retries):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=30) as resp:
                return resp.getcode(), resp.read()
        except HTTPError as e:
            code = getattr(e, "code", 0) or 0
            if code in (429, 500, 502, 503, 504):
                time.sleep(backoff * (2 ** attempt))
                continue
            return code, getattr(e, "read", lambda: b"")()
        except URLError as e:
            last_exc = e
            time.sleep(backoff * (2 ** attempt))
            continue
    if last_exc:
        raise last_exc
    return 0, b""

def _fetch_text_from_github(repo: str, ref: str, path: str, token: Optional[str]) -> str:
    """
    1) Try raw.githubusercontent.com (works for public; avoids rate limits)
    2) Fallback to Contents API (uses token if provided)
    0) If a local file exists at 'path', use it first (dev convenience)
    """
    # 0) Local file shortcut (keeps backward compatibility for dev/offline)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    # 1) Raw
    raw = _GITHUB_RAW_TMPL.format(repo=repo, ref=ref, path=path)
    code_raw, data_raw = _http_get(raw, {"User-Agent": "booklet-retriever"})
    if code_raw == 200 and data_raw:
        return data_raw.decode("utf-8", errors="replace")

    # 2) Contents API
    api = _GITHUB_API_TMPL.format(repo=repo, path=path, ref=ref)
    headers = {"Accept": "application/vnd.github.v3.raw", "User-Agent": "booklet-retriever"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    code_api, data_api = _http_get(api, headers)
    if code_api == 200 and data_api:
        return data_api.decode("utf-8", errors="replace")

    # Helpful error
    body_raw = (data_raw or b"")[:200].decode("utf-8", errors="replace")
    body_api = (data_api or b"")[:200].decode("utf-8", errors="replace")
    msg = [
        f"Failed to fetch '{path}' (booklet).",
        f"raw='{raw}' → HTTP {code_raw}" + (f" body='{body_raw}'" if body_raw else ""),
        f"api='{api}' → HTTP {code_api}" + (f" body='{body_api}'" if body_api else ""),
        "Tips: for private repos provide a token via REPO_XPAT / GITHUB_TOKEN / file."
    ]
    raise RuntimeError(" | ".join(msg))


# =============================================================================
# Token discovery
# =============================================================================

def _read_file_stripped(p: str) -> Optional[str]:
    try:
        if p and os.path.exists(p):
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                v = f.read().strip()
                return v or None
    except Exception:
        return None
    return None

def _discover_token(explicit: Optional[str]) -> Optional[Tuple[str, str]]:
    """
    Returns (token, source) or (None, 'none').
    Search order:
      1) explicit override
      2) env REPO_XPAT
      3) env GITHUB_TOKEN
      4) file pointed by env REPO_XPAT_FILE
      5) local files: REPO_XPAT.txt, .repo_xpat, .xpat
    """
    if explicit:
        return explicit, "override"

    env1 = os.getenv("REPO_XPAT")
    if env1:
        return env1.strip(), "env:REPO_XPAT"

    env2 = os.getenv("GITHUB_TOKEN")
    if env2:
        return env2.strip(), "env:GITHUB_TOKEN"

    # explicit file via env
    env_file = os.getenv("REPO_XPAT_FILE", "").strip()
    if env_file:
        v = _read_file_stripped(env_file)
        if v:
            return v, f"file:{env_file}"

    # well-known local files
    for cand in LOCAL_TOKEN_CANDIDATES:
        if not cand:
            continue
        v = _read_file_stripped(cand)
        if v:
            return v, f"file:{cand}"

    return None, "none"


# =============================================================================
# Local IO for gazetteers
# =============================================================================

def _read_local(path: str) -> str:
    if not os.path.exists(path):
        raise RuntimeError(f"Local file not found: {path}")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


# =============================================================================
# Parsing helpers
# =============================================================================

_HYPHEN_MAP = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), ord("-"))

def _norm_ws_hyphen(s: str) -> str:
    s = (s or "").translate(_HYPHEN_MAP)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[;,\s]*$", "", s)
    return s

def _parse_list(txt: str) -> List[str]:
    out = []
    for line in txt.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(_norm_ws_hyphen(s))
    return out

def _parse_aliases(txt: str) -> Dict[str, Set[str]]:
    mapping: Dict[str, Set[str]] = {}
    for raw in txt.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [_norm_ws_hyphen(p) for p in raw.split("\n") if p.strip()]
        if not parts:
            continue
        canon, aliases = parts[0], parts[1:]
        s = mapping.setdefault(canon, set())
        for a in aliases:
            if a and a != canon:
                s.add(a)
    return mapping

def _dedup_preserve(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        xl = x.lower()
        if xl not in seen:
            seen.add(xl)
            out.append(x)
    return out


# =============================================================================
# Corpus loader (booklet) — same behavior as before
# =============================================================================

def _load_corpus(repo: str, ref: str, booklet_path: str, token: Optional[str]) -> List[Dict]:
    txt = _fetch_text_from_github(repo, ref, booklet_path, token)
    nodes: List[Dict] = []
    for line in txt.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if not (obj.get("text") or "").strip():
            continue
        nodes.append(obj)
    if not nodes:
        raise RuntimeError("Booklet JSONL loaded but contained no usable nodes.")
    return nodes


# =============================================================================
# Gazetteer loader (local files)
# =============================================================================

class Gazetteers:
    def __init__(self, concepts: List[str], cases: List[str], alias_map: Dict[str, Set[str]]):
        self.concepts = concepts
        self.cases = cases
        self.alias_map = alias_map

        bi: Dict[str, Set[str]] = {}
        for canon, alset in alias_map.items():
            cset = bi.setdefault(canon, set())
            for a in alset:
                cset.add(a)
                bi.setdefault(a, set()).add(canon)
        self.alias_bi = bi

def _load_gazetteers_local() -> 'Gazetteers':
    txt_concepts = _read_local(LOCAL_CONCEPTS)
    txt_cases    = _read_local(LOCAL_CASES)
    txt_aliases  = _read_local(LOCAL_ALIASES)
    concepts = _dedup_preserve(_parse_list(txt_concepts))
    cases    = _dedup_preserve(_parse_list(txt_cases))
    aliases  = _parse_aliases(txt_aliases)
    return Gazetteers(concepts, cases, aliases)


# =============================================================================
# Signal extraction
# =============================================================================

RE_SECTION = re.compile(r"§\s*\d+[a-z]?(?:\s*(?:Abs\.?|Satz)\s*\d+)*", re.IGNORECASE)
RE_ARTICLE = re.compile(r"(?:Art\.?|Artikel)\s*\d+[a-z]?(?:\(\d+\))*", re.IGNORECASE)
RE_DOCKET  = re.compile(r"\b(?:[CE]-\d+/\d{2}|[IVX]+\s+Z[RB]\s+\d+/\d{2})\b", re.IGNORECASE)

def _strip_nonword(s: str) -> str:
    return re.sub(r"\W+", "", s or "")

def _difflib_best(token: str, candidates: List[str]) -> Tuple[Optional[str], float, float]:
    if not token or not candidates:
        return None, 0.0, 0.0
    t = token.lower()
    scored = [(c, difflib.SequenceMatcher(None, t, c.lower()).ratio()) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    best, s1 = scored[0]
    s2 = scored[1][1] if len(scored) > 1 else 0.0
    return best, s1, (s1 - s2)

def _should_snap(token: str, score: float, margin: float) -> bool:
    cutoff = _SHORT_SNAP if len(_strip_nonword(token)) <= 6 else _LONG_SNAP
    return (score >= cutoff) and (margin >= _SNAP_MARGIN)

def _wordish_tokens(q: str) -> List[str]:
    q = _norm_ws_hyphen(q)
    return re.findall(r"[A-Za-zÄÖÜäöüß0-9\-/]+", q)

def _expand_aliases(seed: Set[str], alias_bi: Dict[str, Set[str]]) -> Set[str]:
    out = set(seed)
    for s in list(seed):
        for a in alias_bi.get(s, ()):
            out.add(a)
    return out


# =============================================================================
# Corpus auto-aliasing
# =============================================================================

def _find_case_numbers(text: str) -> Set[str]:
    return set(RE_DOCKET.findall(text))

def _find_case_names(text: str, case_names: List[str]) -> Set[str]:
    t = text.lower()
    out: Set[str] = set()
    for name in case_names:
        if len(_strip_nonword(name)) < 4:
            continue
        if name.lower() in t:
            out.add(name)
    return out

def build_corpus_auto_alias(nodes: List[Dict], gaz: Gazetteers) -> Dict[str, Set[str]]:
    auto: Dict[str, Set[str]] = {}
    for n in nodes:
        text = _norm_ws_hyphen(n.get("text", "") or "")
        if not text:
            continue
        names = _find_case_names(text, gaz.cases)
        dockets = _find_case_numbers(text)
        if not names or not dockets:
            continue
        for name in names:
            for num in dockets:
                auto.setdefault(name, set()).add(num)
                auto.setdefault(num, set()).add(name)
    return auto


# =============================================================================
# Matching & scoring
# =============================================================================

_WORD_REGEX = re.compile(r"\w+", re.UNICODE)

def _words(text: str) -> List[str]:
    return _WORD_REGEX.findall(text.lower())

def _has_exact(text_lc: str, needle: str) -> bool:
    n = needle.lower()
    if not n:
        return False
    if re.search(r"\W", n):
        return n in text_lc
    return re.search(rf"\b{re.escape(n)}\b", text_lc) is not None

def _best_fuzzy_against_words(needle: str, words: List[str]) -> float:
    n = needle.lower()
    best = 0.0
    m = len(_strip_nonword(needle))
    if m == 0:
        return 0.0
    for w in words:
        if abs(len(w) - m) > max(2, m // 2):
            continue
        r = difflib.SequenceMatcher(None, n, w).ratio()
        if r > best:
            best = r
    return best

def score_node(text: str, signals: List[Dict]) -> float:
    t_norm = _norm_ws_hyphen(text)
    t_lc = t_norm.lower()
    tokens = _words(t_norm)
    score = 0.0
    saw_case_name = False
    saw_case_no = False

    for s in signals:
        if s["type"] in {"section", "article", "case_no"}:
            if _has_exact(t_lc, s["canonical"]):
                score += W_STRUCTURED
                if s["type"] == "case_no":
                    saw_case_no = True
            continue

        hit_exact = False
        for cand in s["expanded"]:
            if _has_exact(t_lc, cand):
                score += W_GAZ_EXACT
                hit_exact = True
                if s["type"] == "case_name":
                    saw_case_name = True
                if RE_DOCKET.fullmatch(cand):
                    saw_case_no = True
        if hit_exact:
            continue

        if s.get("fuzzy_eligible", False):
            sim = _best_fuzzy_against_words(s["canonical"], tokens)
            if sim >= _FUZZY_ACCEPT:
                score += (W_FUZZY * sim)

    if saw_case_name and saw_case_no:
        score += W_COOCCUR

    return score


# =============================================================================
# Signal extraction (main)
# =============================================================================

def extract_signals(query: str, gaz: Gazetteers, corpus_auto_alias: Dict[str, Set[str]]) -> List[Dict]:
    q = _norm_ws_hyphen(query or "")
    if not q:
        return []

    signals: List[Dict] = []

    for m in RE_SECTION.finditer(q):
        s = m.group(0)
        signals.append(dict(type="section", surface=s, canonical=s,
                            confidence=1.0, expanded={s}, fuzzy_eligible=False))

    for m in RE_ARTICLE.finditer(q):
        s = m.group(0)
        signals.append(dict(type="article", surface=s, canonical=s,
                            confidence=1.0, expanded={s}, fuzzy_eligible=False))

    for m in RE_DOCKET.finditer(q):
        s = m.group(0)
        signals.append(dict(type="case_no", surface=s, canonical=s,
                            confidence=1.0, expanded={s}, fuzzy_eligible=False))

    surface_tokens = _wordish_tokens(q)
    surface_tokens.sort(key=lambda x: (-len(_strip_nonword(x)), x.lower()))

    for tok in surface_tokens:
        if RE_SECTION.fullmatch(tok) or RE_ARTICLE.fullmatch(tok) or RE_DOCKET.fullmatch(tok):
            continue

        canonical = None
        snapped_type = None
        confidence = 0.0

        best, score, margin = _difflib_best(tok, gaz.concepts)
        if best and _should_snap(tok, score, margin):
            canonical = best
            snapped_type = "concept"
            confidence = score
        else:
            best, score, margin = _difflib_best(tok, gaz.cases)
            if best and _should_snap(tok, score, margin):
                canonical = best
                snapped_type = "case_name"
                confidence = score

        if canonical:
            expanded = set([canonical])
            expanded = _expand_aliases(expanded, gaz.alias_bi)
            expanded = _expand_aliases(expanded, corpus_auto_alias)
            signals.append(dict(type=snapped_type, surface=tok, canonical=canonical,
                                confidence=confidence, expanded=expanded, fuzzy_eligible=False))
        else:
            if _strip_nonword(tok):
                signals.append(dict(type="other", surface=tok, canonical=tok,
                                    confidence=0.0, expanded={tok}, fuzzy_eligible=True))

    # Deduplicate
    seen = set()
    out: List[Dict] = []
    for s in signals:
        key = (s["type"], s["canonical"].lower(),
               tuple(sorted(x.lower() for x in s["expanded"])))
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


# =============================================================================
# ParagraphRetriever
# =============================================================================

class ParagraphRetriever:
    def __init__(self, _ignored=None, token: Optional[str] = None):
        # Local gazetteers
        self.gaz = _load_gazetteers_local()

        # Token discovery (keeps your previous setup working and adds fallbacks)
        self._token, source = _discover_token(token)
        _dbg(f"token source: {source}")

        # Booklet loaded exactly like before (HTTP to GitHub; local if present)
        repo  = _DEFAULT_REPO
        ref   = _DEFAULT_REF
        self.nodes: List[Dict] = _load_corpus(repo, ref, _BOOKLET_PATH, self._token)

        # Auto alias from corpus
        auto_alias = build_corpus_auto_alias(self.nodes, self.gaz)

        # Merge alias maps
        self.alias_bi = dict(self.gaz.alias_bi)
        for k, v in auto_alias.items():
            self.alias_bi.setdefault(k, set()).update(v)

        # Precompute locals
        self._texts_lc: List[str] = [_norm_ws_hyphen(n.get("text", "") or "").lower() for n in self.nodes]
        self._tokens_list: List[List[str]] = [_words(_norm_ws_hyphen(n.get("text", "") or "")) for n in self.nodes]

    def search(self, query: str, top_k: int = 6, **_kwargs) -> List[Dict]:
        if not (query or "").strip():
            return []
        signals = extract_signals(query, self.gaz, self.alias_bi)
        if not signals:
            return []

        scored: List[Tuple[int, float]] = []
        for i, n in enumerate(self.nodes):
            s = score_node(n.get("text", "") or "", signals)
            if s >= 1.0:
                scored.append((i, s))
        if not scored:
            return []

        scored.sort(key=lambda x: (-x[1],
                                   len(self.nodes[x[0]].get("text", "") or ""),
                                   x[0]))

        indices = [i for (i, _s) in scored[:max(1, top_k)]]
        scores  = [float(_s) for (_i, _s) in scored[:max(1, top_k)]]
        return self._package_hits(indices, scores)

    def _package_hits(self, indices: List[int], scores: List[float]) -> List[Dict]:
        out: List[Dict] = []
        for rank, (i, score) in enumerate(zip(indices, scores), start=1):
            n = self.nodes[i]
            breadcrumb = n.get("breadcrumb", None)
            if breadcrumb is None:
                breadcrumb = n.get("breadcrumbs", None)
            item = {
                "text": n.get("text", ""),
                "score": score,
                "rank": rank,
                "node_id": n.get("node_id"),
                "doc_id": n.get("doc_id"),
                "type": n.get("type"),
                "anchor": n.get("anchor"),
                "breadcrumb": breadcrumb,
                "lang": n.get("lang"),
                "links": n.get("links", {}),
            }
            out.append(item)
        return out


if __name__ == "__main__":
    # Optional: set RAG_DEBUG=1 to see the token source used
    q = "Lafonat § 33 WpHG Ad-hoc-Publizität"
    r = ParagraphRetriever()
    hits = r.search(q, top_k=6)
    for h in hits:
        print(f"- rank={h['rank']} score={h['score']:.2f} | {h['text'][:120]}")
