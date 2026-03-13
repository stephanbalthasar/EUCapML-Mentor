# mentor/rag/web_curia_eurlex.py
from __future__ import annotations

import html
import io
import re
import time
from typing import List, Dict, Tuple
from urllib.parse import urlparse, parse_qs, unquote, urljoin

import requests

# Optional HTML parsing (preferred). If bs4 is absent, we fall back to regex.
try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None  # we will fall back if needed

# Optional PDF text extraction
try:
    from PyPDF2 import PdfReader  # type: ignore
except Exception:
    PdfReader = None


# -----------------------
# Google + Official sites
# -----------------------

_GOOGLE_SEARCH_URL = "https://www.google.com/search"

# Domains we accept (in this order of preference)
_ALLOWED_DOMAINS = ("curia.europa.eu", "eur-lex.europa.eu", "esma.europa.eu")


def _ua() -> str:
    # A friendly UA reduces blocks in many environments
    return (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/119.0 Safari/537.36"
    )


def _is_english_url(u: str) -> bool:
    # Prefer English variants in URL path if present
    u_lower = u.lower()
    return ("/en/" in u_lower) or u_lower.endswith("en.pdf") or "lang=en" in u_lower


def _classify(url: str) -> Tuple[int, str]:
    """
    Priority tuple (smaller is better), and a coarse type label:
      1 -> CURIA press release
      2 -> EUR-Lex judgment summary
      3 -> Judgment text (CURIA/EUR-Lex)
      4 -> ESMA
      5 -> Other within allowed domains
    """
    u = url.lower()

    # CURIA press releases are often cpNNNNNNen.pdf
    if "curia.europa.eu" in u and ("cp" in u and u.endswith(".pdf")):
        return (1, "CURIA_PRESS_RELEASE")

    # EUR-Lex summary uses ...CELEX:...._SUM
    if "eur-lex.europa.eu" in u and "_sum" in u:
        return (2, "EURLEX_SUMMARY")

    # Judgment text (no _SUM) from EUR-Lex or CURIA
    if ("eur-lex.europa.eu" in u or "curia.europa.eu" in u) and (
        "document" in u or "docid=" in u or "celex:" in u
    ):
        return (3, "JUDGMENT")

    # ESMA (used for general regulatory topics)
    if "esma.europa.eu" in u:
        return (4, "ESMA")

    # Default (still allowed domain)
    return (5, "OTHER")


def _clean_snippet_text(txt: str, limit: int = 700) -> str:
    txt = html.unescape(txt or "")
    txt = re.sub(r"\s+", " ", txt).strip()
    if len(txt) <= limit:
        return txt

    # Cut at the nearest sentence boundary not too far before the limit
    cut = txt[:limit]
    m = re.search(r"(?s)^(.+?[.!?])(?=\s|$)", cut[::-1])  # reverse search
    if m:
        # Reverse back to original
        boundary_len = len(m.group(1))
        # index from the start:
        idx = limit - boundary_len
        return txt[:idx + boundary_len].strip()
    return cut.strip()


def _extract_visible_text(html_text: str) -> str:
    if BeautifulSoup is None:
        # fallback: rough strip of tags
        return re.sub(r"<[^>]+>", " ", html_text or "")

    soup = BeautifulSoup(html_text, "html.parser")

    # EUR‑Lex summary often in main content; CURIA press pages (if HTML) are small.
    # Heuristic: gather first few meaningful <p> from body (skip nav/footer)
    body = soup.body or soup
    paras = []
    for p in body.find_all("p"):
        t = p.get_text(strip=True)
        if t and len(t.split()) > 4:
            paras.append(t)
        if len(paras) >= 6:
            break
    return " ".join(paras) if paras else soup.get_text(separator=" ", strip=True)


def _pdf_to_text(content: bytes) -> str:
    if not content:
        return ""
    if PdfReader is None:
        return ""  # unable to parse PDFs in this environment
    try:
        reader = PdfReader(io.BytesIO(content))
        pages = min(2, len(reader.pages))  # first 1-2 pages normally contain the holding
        out = []
        for i in range(pages):
            try:
                out.append(reader.pages[i].extract_text() or "")
            except Exception:
                pass
        return " ".join(out)
    except Exception:
        return ""


def _request(url: str, *, timeout: float = 6.0) -> Tuple[int, bytes, str]:
    """
    Return (status_code, content_bytes, content_type)
    """
    try:
        headers = {"User-Agent": _ua(), "Accept": "*/*"}
        r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        ctype = r.headers.get("Content-Type", "")
        return r.status_code, r.content, ctype
    except Exception:
        return 0, b"", ""


def _extract_google_results(html_text: str) -> List[str]:
    # Parse Google SERP safely: look for '/url?q=' anchors
    urls: List[str] = []
    for m in re.finditer(r'href="/url\?q=([^"&]+)', html_text or ""):
        target = unquote(m.group(1))
        urls.append(target)
    return urls


def _filter_and_rank(urls: List[str], prefer_lang_en: bool = True) -> List[Tuple[str, int, str, int]]:
    """
    Keep only allowed domains, de-duplicate, and rank by:
      (priority, language_penalty), lower is better.
    Return tuples: (url, priority, typ, penalty)
    """
    seen: set[str] = set()
    ranked: List[Tuple[str, int, str, int]] = []

    for u in urls:
        parsed = urlparse(u)
        host = parsed.netloc.lower()
        if not any(host.endswith(dom) for dom in _ALLOWED_DOMAINS):
            continue
        # de-duplicate by normalized URL
        norm = u.split("#")[0]
        if norm in seen:
            continue
        seen.add(norm)

        prio, typ = _classify(u)
        penalty = 0
        if prefer_lang_en and not _is_english_url(u):
            penalty = 1  # small penalty for non-EN

        ranked.append((u, prio, typ, penalty))

    ranked.sort(key=lambda t: (t[1], t[3]))
    return ranked


def _extract_text_from_url(url: str, ctype: str, content: bytes) -> str:
    if "pdf" in ctype.lower() or url.lower().endswith(".pdf"):
        return _pdf_to_text(content)
    # HTML-like
    try:
        text = content.decode("utf-8", errors="replace")
    except Exception:
        text = content.decode("latin-1", errors="replace")
    return _extract_visible_text(text)


def _lang_from_url(url: str) -> str:
    u = url.lower()
    if "/de/" in u or u.endswith("de.pdf"):
        return "DE"
    if "/fr/" in u or u.endswith("fr.pdf"):
        return "FR"
    if "/en/" in u or u.endswith("en.pdf"):
        return "EN"
    return "EN"  # default


class CuriaEurlexRetriever:
    """
    Google-first retriever for official EU legal sources (CURIA, EUR-Lex, ESMA).
    - Queries Google with a site filter.
    - Keeps the first N hits from allowed domains.
    - Ranks: CURIA press release > EUR-Lex summary > judgment > ESMA > rest.
    - Fetches and extracts small, high-quality snippets (<= 700 chars).
    - Returns up to 'top_k' snippets as list[dict] with keys: text, source_url, source_type, lang.
    """

    def __init__(self, lang: str = "EN", timeout_sec: float = 6.0):
        self.lang = lang.upper()
        self.timeout = float(timeout_sec)

    def retrieve(self, query: str, keywords: List[str] | None = None, top_k: int = 4) -> List[Dict]:
        if not (query or "").strip():
            return []

        # 1) Google with site filters
        site_filter = "site:eur-lex.europa.eu OR site:curia.europa.eu OR site:esma.europa.eu"
        q = f'{site_filter} "{query.strip()}"'

        params = {
            "q": q,
            "hl": self.lang.lower(),  # help ranking in the chosen language
            "num": "10",
            "safe": "off"
        }
        headers = {"User-Agent": _ua(), "Accept": "text/html,*/*"}

        try:
            r = requests.get(_GOOGLE_SEARCH_URL, params=params, headers=headers, timeout=self.timeout)
            html_serp = r.text if r.status_code == 200 else ""
        except Exception:
            html_serp = ""

        if not html_serp:
            return []  # no web fallback in this simple version

        # 2) Extract candidate URLs and rank
        raw_urls = _extract_google_results(html_serp)
        ranked = _filter_and_rank(raw_urls, prefer_lang_en=(self.lang == "EN"))
        if not ranked:
            return []

        # 3) Take at most five candidates (your requirement) before fetching
        candidates = ranked[:5]

        # 4) Fetch pages, extract snippets, and build outputs
        out: List[Dict] = []
        for url, prio, typ, penalty in candidates:
            status, content, ctype = _request(url, timeout=self.timeout)
            if status != 200 or not content:
                continue

            text = _extract_text_from_url(url, ctype, content)
            if not text:
                continue

            snippet = _clean_snippet_text(text, limit=700)
            if not snippet or len(snippet.split()) < 12:
                # too short -> skip
                continue

            out.append({
                "text": snippet,
                "source_url": url,
                "source_type": typ,
                "lang": _lang_from_url(url)
            })

            if len(out) == top_k:
                break

        return out
