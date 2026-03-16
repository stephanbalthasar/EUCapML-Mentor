# -*- coding: utf-8 -*-
"""
Single-file booklet retriever (TXT gazetteers + JSONL booklet, standard-library only).

Pipeline
--------
(1) Load gazetteers (TXT) and booklet (JSONL) from GitHub Contents API using REPO_XPAT.
(2) Extract legal signals from query:
      - regex for sections (e.g., "§ 33", "§ 33 Abs. 1"),
      - regex for articles (e.g., "Art. 7(1)"),
      - regex for case numbers (EU/EFTA/DE dockets, e.g., "C-628/13", "E-1/10", "II ZR 9/21"),
      - lookups against gazetteer_concepts.txt and gazetteer_cases.txt with conservative fuzzy snapping,
      - alias expansion from gazetteer_aliases.txt + auto-links learned from the corpus (name <-> number).
(3) Match against local corpus paragraphs and score:
      - exact hits on structured signals and gazetteer/alias terms (high weight),
      - fuzzy hits for non-snapped tokens (discounted),
      - small co-occurrence bonus (case name + number in the same paragraph),
      - return top 6 with score >= 1.0.

Outputs
-------
Returns a list[dict] of hits shaped like the legacy code:
  {
    "text": str,
    "score": float | None,     # here it is a float; None if unavailable
    "rank": int,
    "node_id": any,
    "doc_id": any,
    "type": any,
    "anchor": any,
    "breadcrumb": any,         # if the JSONL uses "breadcrumbs", we pass that through here
    "lang": any,
    "links": dict | any,
  }

Configuration (env)
-------------------
REPO_XPAT      : GitHub token (Fine-grained PAT) for private repo read.
BOOKLET_REPO   : optional override; default "stephanbalthasar/b2-eucapml-content"
BOOKLET_REF    : optional override; default "main"
GAZ_BASE       : optional override; default "assets"
BOOKLET_PATH   : optional override; default "artifacts/booklet_index.jsonl"

Author: M365 Copilot for Stephan Balthasar
"""

from __future__ import annotations

import json
import os
import re
import time
import html
from typing import Dict, List, Optional, Tuple, Set, Iterable
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import difflib

# ----------------------------- Config & constants -----------------------------

_DEFAULT_REPO = os.getenv("BOOKLET_REPO", "stephanbalthasar/b2-eucapml-content")
_DEFAULT_REF = os.getenv("BOOKLET_REF", "main")
_ASSETS_BASE = os.getenv("GAZ_BASE", "assets")
_BOOKLET_PATH = os.getenv("BOOKLET_PATH", "artifacts/booklet_index.jsonl")

_GZ_CONCEPTS = f"{_ASSETS_BASE}/gazetteer_concepts.txt"
_GZ_CASES = f"{_ASSETS_BASE}/gazetteer_cases.txt"
_GZ_ALIASES = f"{_ASSETS_BASE}/gazetteer_aliases.txt"

_GITHUB_API_TMPL = "https://api.github.com/repos/{repo}/contents/{path}?ref={ref}"
_GITHUB_RAW_TMPL = "https://raw.githubusercontent.com/{repo}/{ref}/{path}"

# Fuzzy thresholds (conservative, tuned for legal acronyms/names)
_SHORT_SNAP = 0.92  # short tokens (<=6 chars): require high similarity
_LONG_SNAP = 0.85   # longer names: allow slightly lower but still strict
_SNAP_MARGIN = 0.05 # uniqueness margin vs 2nd best
_FUZZY_ACCEPT = 0.82  # stage (2) fuzzy acceptance threshold

# Scoring weights (simple & transparent)
W_STRUCTURED = 3.0      # §§, Art., docket numbers
W_GAZ_EXACT = 2.5       # exact match of canonical or alias
W_FUZZY = 0.6           # multiplier applied to similarity (0..1)
W_COOCCUR = 0.4         # bonus if case name & number co-occur in node


# ----------------------------- Utility: HTTP fetch ----------------------------

def _get_token() -> Optional[str]:
    # Token from environment/secrets; name agreed: REPO_XPAT
    return os.getenv("REPO_XPAT")

def _http_get(url: str, headers: Dict[str, str], retries: int = 3, backoff: float = 0.75) -> Tuple[int, bytes]:
    last_exc = None
    for attempt in range(retries):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=30) as resp:
                return resp.getcode(), resp.read()
        except HTTPError as e:
            code = getattr(e, "code", 0) or 0
            # retry on transient statuses
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
    Fetch raw file content from GitHub. Use Contents API if token is present; fall back to raw.
    """
    if token:
        api = _GITHUB_API_TMPL.format(repo=repo, path=path, ref=ref)
        code, data = _http_get(api, {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3.raw",
            "User-Agent": "booklet-retriever"
        })
        if code == 200:
            return data.decode("utf-8", errors="replace")
        # If API fails, try raw as a fallback (in case repo is public or token scopes differ)
    raw = _GITHUB_RAW_TMPL.format(repo=repo, ref=ref, path=path)
    code, data = _http_get(raw, {
        "User-Agent": "booklet-retriever"
    })
    if code != 200:
        raise RuntimeError(f"Failed to fetch {path} (HTTP {code}).")
    return data.decode("utf-8", errors="replace")


# ----------------------------- Parsing helpers -------------------------------

_HYPHEN_MAP = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), ord("-"))

def _norm_ws_hyphen(s: str) -> str:
    # unify hyphens/dashes; collapse whitespace; strip trailing comma/semicolon
    s = (s or "").translate(_HYPHEN_MAP)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[;,]\s*$", "", s)
    return s

def _parse_list(txt: str) -> List[str]:
    out: List[str] = []
    for line in txt.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(_norm_ws_hyphen(s))
    return out

def _parse_aliases(txt: str) -> Dict[str, Set[str]]:
    """
    Each line: Canonical | alias1 | alias2 | ...
    Returns canonical -> set(aliases). (We add reverse mapping at runtime.)
    """
    mapping: Dict[str, Set[str]] = {}
    for raw in txt.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # split by '|' and normalize each part
        parts = [_norm_ws_hyphen(p) for p in line.split("|")]
        parts = [p for p in parts if p]
        if not parts:
            continue
        canon, aliases = parts[0], parts[1:]
        s = mapping.setdefault(canon, set())
        for a in aliases:
            if a and a != canon:
                s.add(a)
    return mapping


# ----------------------------- Corpus loader ---------------------------------

def _load_corpus(repo: str, ref: str, booklet_path: str, token: Optional[str]) -> List[Dict]:
    """
    Load JSONL booklet. Each line is a JSON object; we require at least a "text" field.
    We pass through any other fields (node_id, doc_id, type, anchor, breadcrumb(s), lang, links).
    """
    txt = _fetch_text_from_github(repo, ref, booklet_path, token)
    nodes: List[Dict] = []
    for i, line in enumerate(txt.splitlines()):
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        t = (obj.get("text") or "").strip()
        if not t:
            continue
        nodes.append(obj)
    if not nodes:
        raise RuntimeError("Booklet JSONL loaded but contained no usable nodes.")
    return nodes


# ----------------------------- Gazetteer loader ------------------------------

class Gazetteers:
    def __init__(self,
                 concepts: List[str],
                 cases: List[str],
                 alias_map: Dict[str, Set[str]]):
        # store original casing; build lowercase indices for lookups
        self.concepts = concepts
        self.cases = cases
        self.alias_map = alias_map

        self._concepts_lc = {c.lower(): c for c in concepts}
        self._cases_lc = {c.lower(): c for c in cases}

        # Build reverse alias links (two-way) in memory (lossless)
        bi: Dict[str, Set[str]] = {}
        for canon, alset in alias_map.items():
            cset = bi.setdefault(canon, set())
            for a in alset:
                cset.add(a)
                bi.setdefault(a, set()).add(canon)
        self.alias_bi = bi


def _load_gazetteers(repo: str, ref: str, token: Optional[str]) -> Gazetteers:
    txt_concepts = _fetch_text_from_github(repo, ref, _GZ_CONCEPTS, token)
    txt_cases = _fetch_text_from_github(repo, ref, _GZ_CASES, token)
    txt_aliases = _fetch_text_from_github(repo, ref, _GZ_ALIASES, token)

    concepts = _dedup_preserve(_parse_list(txt_concepts))
    cases = _dedup_preserve(_parse_list(txt_cases))
    alias_map = _parse_aliases(txt_aliases)
    return Gazetteers(concepts, cases, alias_map)

def _dedup_preserve(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        xl = x.lower()
        if xl not in seen:
            seen.add(xl)
            out.append(x)
    return out


# ----------------------------- Signal extraction -----------------------------

# Structured patterns
RE_SECTION = re.compile(r"§\s*\d+[a-z]?(?:\s*(?:Abs\.?|Satz)\s*\d+)*", re.IGNORECASE)
RE_ARTICLE = re.compile(r"(?:Art\.?|Artikel)\s*\d+[a-z]?(?:\(\d+\))*", re.IGNORECASE)
RE_DOCKET = re.compile(
    r"\b(?:[CTF]-\d+/\d{2}|E-\d+/\d{2}|[IVX]+\s+Z[RB]\s+\d+/\d{2})\b",
    re.IGNORECASE
)

def _difflib_best(token: str, candidates: List[str]) -> Tuple[Optional[str], float, float]:
    """
    Return (best_candidate, score, margin_to_second).
    Score is difflib ratio (0..1), case-insensitive.
    """
    if not token or not candidates:
        return None, 0.0, 0.0
    t = token.lower()
    lows = [c.lower() for c in candidates]
    scored = [(candidates[i], difflib.SequenceMatcher(None, t, lows[i]).ratio())
              for i in range(len(candidates))]
    scored.sort(key=lambda x: x[1], reverse=True)
    best, s1 = scored[0]
    s2 = scored[1][1] if len(scored) > 1 else 0.0
    return best, s1, (s1 - s2)

def _should_snap(token: str, score: float, margin: float) -> bool:
    cutoff = _SHORT_SNAP if len(_strip_nonword(token)) <= 6 else _LONG_SNAP
    return (score >= cutoff) and (margin >= _SNAP_MARGIN)

def _strip_nonword(s: str) -> str:
    return re.sub(r"\W+", "", s or "")

def _wordish_tokens(q: str) -> List[str]:
    # Keep tokens; preserve hyphenated words; drop surrounding punctuation
    # We still keep 'C-628/13' as a single token
    q = _norm_ws_hyphen(q)
    toks = re.findall(r"[A-Za-zÄÖÜäöüß0-9\-\/]+", q)
    return [t for t in toks if t]

def _expand_aliases(seed: Set[str], alias_bi: Dict[str, Set[str]]) -> Set[str]:
    out = set(seed)
    for s in list(seed):
        for a in alias_bi.get(s, ()):
            out.add(a)
    return out

def extract_signals(query: str, gaz: Gazetteers, corpus_auto_alias: Dict[str, Set[str]]) -> List[Dict]:
    """
    Returns a list of signals. Each signal:
      {
        "type": "section"|"article"|"case_no"|"concept"|"case_name"|"other",
        "surface": ...,
        "canonical": ...,
        "confidence": float (0..1),
        "expanded": set[str],       # canonical plus aliases (bi-directional)
        "fuzzy_eligible": bool,
      }
    """
    q = _norm_ws_hyphen(query or "")
    if not q:
        return []

    signals: List[Dict] = []

    # 1) Structured from regex
    for m in RE_SECTION.finditer(q):
        s = m.group(0)
        signals.append(dict(type="section", surface=s, canonical=s, confidence=1.0,
                            expanded=set([s]), fuzzy_eligible=False))
    for m in RE_ARTICLE.finditer(q):
        s = m.group(0)
        signals.append(dict(type="article", surface=s, canonical=s, confidence=1.0,
                            expanded=set([s]), fuzzy_eligible=False))
    for m in RE_DOCKET.finditer(q):
        s = m.group(0)
        signals.append(dict(type="case_no", surface=s, canonical=s, confidence=1.0,
                            expanded=set([s]), fuzzy_eligible=False))

    # 2) Gazetteer lookups with conservative snapping
    surface_tokens = _wordish_tokens(q)
    # For matching, prefer longer tokens first to avoid snapping tiny words
    surface_tokens.sort(key=lambda x: (-len(_strip_nonword(x)), x.lower()))

    # Lowercased lookup tables
    concepts = gaz.concepts
    cases = gaz.cases

    for tok in surface_tokens:
        tl = tok.lower()

        # Skip if already captured as structured docket/section/article
        if RE_SECTION.fullmatch(tok) or RE_ARTICLE.fullmatch(tok) or RE_DOCKET.fullmatch(tok):
            continue

