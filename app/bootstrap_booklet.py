# app/bootstrap_booklet.py
# Plain-vanilla loader for the NEW booklet index (JSONL) in a private GitHub repo.
# Returns a dict compatible with your existing app UI:
#   {"paragraphs": [...], "chapters": [...]}
#
# Required settings (Streamlit secrets preferred; env fallback works too):
#   BOOKLET_REPO  = "owner/repo"
#   BOOKLET_REF   = "main"
#   BOOKLET_PATH  = "artifacts/booklet_index.jsonl"
#
# Token (first present wins):
#   GITHUB_TOKEN  (preferred)
#   REPO_XPAT
#   BOOKLET_TOKEN
#
# Notes:
# - We deliberately parse JSONL (newline-delimited JSON). Each line must be a JSON object.
# - Paragraph-like nodes (paragraph/case_note/footnote) become "paragraphs" with only {"text": ...}.
# - Section nodes become minimal "chapters" with {"chapter_num": None, "text": anchor-or-text}.

from __future__ import annotations

import json
import os
from typing import Dict, Any, List, Optional, Tuple

import requests

try:
    import streamlit as st  # preferred (so we can use st.secrets and st.cache_data)
except Exception:
    # Minimal stub so local tests don't crash if Streamlit isn't present.
    class _Stub:
        def __getattr__(self, _):
            raise AttributeError
    st = _Stub()  # type: ignore


# ---------- helpers ----------

def _secret_or_env(key: str) -> Optional[str]:
    """Read from Streamlit secrets first; fall back to environment variables."""
    try:
        if hasattr(st, "secrets"):
            val = st.secrets.get(key)
            if val:
                return str(val)
    except Exception:
        pass
    return os.getenv(key)


def _raw_url_and_headers() -> Tuple[str, Dict[str, str]]:
    """
    Build raw.githubusercontent URL + auth headers from secrets/env.
    """
    repo = _secret_or_env("BOOKLET_REPO")
    ref  = _secret_or_env("BOOKLET_REF") or "main"
    path = _secret_or_env("BOOKLET_PATH") or "artifacts/booklet_index.jsonl"

    if not repo:
        raise RuntimeError("BOOKLET_REPO is not configured.")
    raw_url = f"https://raw.githubusercontent.com/{repo}/{ref}/{path}"

    token = (
        _secret_or_env("GITHUB_TOKEN")  # preferred
        or _secret_or_env("REPO_XPAT")
        or _secret_or_env("BOOKLET_TOKEN")
    )
    headers = {"Authorization": f"token {token}"} if token else {}
    return raw_url, headers


# ---------- loader ----------

# Cache for 24h so the file isn’t fetched on every run.
if hasattr(st, "cache_data"):
    _cache = st.cache_data(show_spinner=False, ttl=86400)  # type: ignore
else:
    def _cache(func):  # no-op decorator if Streamlit isn't available
        return func


@_cache
def load_booklet_index() -> Dict[str, Any]:
    """
    Download and parse the JSONL booklet index → legacy-friendly dict:
      {"paragraphs": [{"text": ...}, ...], "chapters": [{"chapter_num": None, "text": ...}, ...]}
    """
    raw_url, headers = _raw_url_and_headers()

    # Fetch
    r = requests.get(raw_url, headers=headers, timeout=30)
    r.raise_for_status()
    text = r.text or ""

    # Parse JSONL → shape into legacy structure
    paragraphs: List[Dict[str, str]] = []
    chapters:   List[Dict[str, str]] = []

    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            row = json.loads(ln)
        except Exception:
            # Malformed line: skip but continue
            continue

        ntype = (row.get("type") or "").strip()
        if ntype in ("paragraph", "case_note", "footnote"):
            t = (row.get("text") or "").strip()
            if t:
                paragraphs.append({"text": t})
        elif ntype == "section":
            # Minimal chapter record (good enough for counts/labels in your UI)
            anchor = (row.get("anchor") or "").strip()
            ch_txt = anchor or (row.get("text") or "").strip()
            if ch_txt:
                chapters.append({"chapter_num": None, "text": ch_txt})

    return {"paragraphs": paragraphs, "chapters": chapters}
