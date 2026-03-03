# app/bootstrap_booklet.py
# Server-side loader for the booklet_index.json sitting in a private GitHub repo.

import streamlit as st
import requests

# Configure via Streamlit secrets (set in the app’s deployment):
REPO  = st.secrets.get("BOOKLET_REPO")                    # e.g. "your-org/eucapml-content-private"
REF   = st.secrets.get("BOOKLET_REF", "main")             # branch / tag / commit SHA
PATH  = st.secrets.get("BOOKLET_PATH", "artifacts/booklet_index.json")
TOKEN = st.secrets.get("GITHUB_TOKEN")                    # fine-grained, contents:read for that repo

RAW_URL = f"https://raw.githubusercontent.com/{REPO}/{REF}/{PATH}"

@st.cache_data(show_spinner=False, ttl=86400)  # cache for 24h; adjust if you prefer
def load_booklet_index():
    if not (REPO and PATH and TOKEN):
        raise RuntimeError("BOOKLET_* secrets or GITHUB_TOKEN missing.")
    headers = {"Authorization": f"Bearer {TOKEN}"}
    r = requests.get(RAW_URL, headers=headers, timeout=20)
    r.raise_for_status()
    return r.json()  # dict with {"paragraphs": [...], "chapters": [...]}
