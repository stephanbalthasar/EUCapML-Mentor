# app/bootstrap_booklet.py
# Server-side loader for the booklet index stored in a private GitHub repo.
# Returns a dict like: {"paragraphs": [...], "chapters": [...]}
# Works with BOTH:
#   • legacy JSON (single object)
#   • JSONL (one JSON object per line)

from __future__ import annotations

import json
import os
from typing import Optional, Tuple, Dict, Any, List

import requests

try:
    import streamlit as st  # type: ignore
except Exception:
    # If Streamlit isn't available (e.g., during unit tests),
    # we still allow env-based configuration.
    class _Stub:
        def __getattr__(self, _):  # minimal stub for st.secrets/st.cache_data
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
