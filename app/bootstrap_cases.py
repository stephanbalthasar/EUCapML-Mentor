# app/bootstrap_cases.py
import streamlit as st
import requests

# Reuse the booklet repo/ref by default; override with CASES_* if you want.
REPO  = st.secrets.get("CASES_REPO", st.secrets.get("BOOKLET_REPO"))
REF   = st.secrets.get("CASES_REF",  st.secrets.get("BOOKLET_REF", "main"))
PATH  = st.secrets.get("CASES_PATH", "artifacts/cases.json")
TOKEN = st.secrets.get("GITHUB_TOKEN")

def _contents_api_url(repo: str, ref: str, path: str) -> str:
    owner, name = repo.split("/", 1)
    return f"https://api.github.com/repos/{owner}/{name}/contents/{path}?ref={ref}"

@st.cache_data(show_spinner=False, ttl=86400)
def load_cases() -> list[dict]:
    if not (REPO and PATH and TOKEN):
        raise RuntimeError("CASES_* secrets or GITHUB_TOKEN missing.")
    url = _contents_api_url(REPO, REF, PATH)
    r = requests.get(
        url,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Accept": "application/vnd.github.v3.raw",   # return raw file content
        },
        timeout=20,
    )
    r.raise_for_status()
    return r.json()  # list of case dicts
