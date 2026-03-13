# mentor/rag/web_curia_eurlex.py
from __future__ import annotations

import html
import io
import re
from typing import List, Dict, Tuple
from urllib.parse import parse_qs, unquote

import requests

# Optional HTML parsing (preferred). If bs4 is absent, we fall back to regex.
try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None  # fallback will be used

# Optional PDF text extraction. If PyPDF2 is absent, we skip PDFs.
try:
    from PyPDF2 import PdfReader  # type: ignore
except Exception:
    PdfReader = None


# -----------------------
# Config / constants
# -----------------------

_GOOGLE_SEARCH_URL = "https://www.google.com/search"
_DDG_SEARCH_URL = "https://html.duckduckgo.com/html/"

# Timeouts keep the app snappy even if engines/sites are slow or blocked.
_SERP_TIMEOUT = 2.0    # seconds per search request
_PAGE_TIMEOUT = 2.5    # seconds per page fetch
_MAX_GOOGLE_LINKS = 3
_MAX_DDG_LINKS = 3
_SNIPPET_LEN = 300     # characters


def _ua() -> str:
    """Minimal UA to reduce blocks in common environments."""
    return (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )


# -----------------------
# Simple helpers
# -----------------------

def _http_get(url: str, *, timeout: float) -> Tuple[int, bytes, str]:
    """Return (status_code, content, content_type)."""
    try:
        r = requests.get(
            url,
            headers={"User-Agent": _ua(), "Accept": "*/*"},
            timeout=timeout,
            allow_redirects=True,
        )
        return r.status_code, (r.content or b""), (r.headers.get("Content-Type", "") or "")
    except Exception:
        return 0, b"", ""


def _extract_text_from_pdf(content: bytes) -> str:
    """Extract text from first 1–2 pages of a PDF (if PyPDF2 is available)."""
    if not content or PdfReader is None:
        return ""
    try:
        reader = PdfReader(io.BytesIO(content))
        pages = min(2, len(reader.pages))
        out = []
        for i in range(pages):
            try:
                out.append(reader.pages[i].extract_text() or "")
            except Exception:
                pass
        return " ".join(out)
    except Exception:
        return ""


def _extract_text_from_html(content: bytes) -> str:
    """Extract visible text from HTML (BeautifulSoup if present, else regex)."""
    # Try UTF‑8 then Latin‑1
    try:
        text = content.decode("utf-8", errors="replace")
    except Exception:
        text = content.decode("latin-1", errors="replace")

    if BeautifulSoup is None:
        # Rough tag strip fallback
        return re.sub(r"<[^>]+>", " ", text or "")

    soup = BeautifulSoup(text, "html.parser")
    body = soup.body or soup
    paras: List[str] = []

    # Gather a few <p> blocks with some substance
    for p in body.find_all("p"):
        t = p.get_text(" ", strip=True)
        if t and len(t.split()) > 4:
            paras.append(t)
        if len(paras) >= 8:
            break

    if paras:
        return " ".join(paras)
    return soup.get_text(" ", strip=True)


def _to_snippet(text: str, *, limit: int = _SNIPPET_LEN) -> str:
    """Normalize whitespace and trim to <= limit; try to end at a sentence."""
    t = html.unescape(text or "")
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) <= limit:
        return t

    # Try to stop on a sentence boundary within the last ~120 chars
    window_start = max(0, limit - 120)
    window = t[window_start:limit]

    # Find the last '.', '!' or '?' followed by space/end
    last_end = -1
    for m in re.finditer(r"[.!?](?=\s|$)", window):
        last_end = m.end()
    if last_end != -1:
        end = window_start + last_end
        return t[:end].strip()

    return t[:limit].strip()


def _dedupe_keep_order(urls: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for u in urls:
        norm = u.split("#")[0]
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


# -----------------------
# SERP parsers (very permissive)
# -----------------------

def _parse_google_serp(html_text: str) -> List[str]:
    """
    Extract result target URLs from a Google HTML SERP.
    Handles:
      - relative redirects: /url?q=...
      - absolute redirects: https://www.google.com/url?...
      - direct anchors (occasionally)
    """
    urls: List[str] = []

    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(html_text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]

                # Ignore internal google links
                if href.startswith("/search"):
                    continue
                if "google." in href and "/url?" not in href:
                    continue

                # /url?q=<target>
                if href.startswith("/url?"):
                    qs = parse_qs(href.split("?", 1)[-1])
                    target = qs.get("q", [None])[0]
                    if target and target.startswith("http"):
                        urls.append(unquote(target))
                        continue

                # https://www.google.com/url?...
                if href.startswith("https://www.google.com/url"):
                    qs = parse_qs(href.split("?", 1)[-1])
                    target = qs.get("q", [None])[0]
                    if target and target.startswith("http"):
                        urls.append(unquote(target))
                        continue

                # direct
                if href.startswith("http"):
                    urls.append(unquote(href))
        except Exception:
            pass

    # Fallback regex if BS4 missing/failed
    if not urls:
        for m in re.finditer(r'href="/url\?q=([^"&]+)', html_text or ""):
            urls.append(unquote(m.group(1)))
        for m in re.finditer(r'href="(https?://[^"]+)"', html_text or ""):
            urls.append(unquote(m.group(1)))

    # Clean & de‑dup; drop google-owned and cached links
    cleaned: List[str] = []
    seen: set[str] = set()
    for u in urls:
        if not u.startswith("http"):
            continue
        if "google." in u:
            continue
        if u.startswith("https://webcache.googleusercontent.com") or u.startswith("http://webcache.googleusercontent.com"):
            continue
        norm = u.split("#")[0]
        if norm not in seen:
            seen.add(norm)
            cleaned.append(norm)
    return cleaned


def _parse_ddg_html(html_text: str) -> List[str]:
    """
    Extract result URLs from DuckDuckGo's static HTML endpoint.
    Common pattern: /l/?uddg=<encoded-target>
    """
    urls: List[str] = []

    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(html_text, "html.parser")
            # Typical results links
            for a in soup.select("a.result__a[href]"):
                href = a["href"]
                if href.startswith("/l/?"):
                    # extract uddg param
                    try:
                        qs = parse_qs(href.split("?", 1)[-1])
                        target = qs.get("uddg", [None])[0]
                        if target and target.startswith("http"):
                            urls.append(unquote(target))
                    except Exception:
                        pass
                elif href.startswith("http"):
                    urls.append(unquote(href))
        except Exception:
            pass

    # Fallback regex if BS4 missing/failed
    if not urls:
        for m in re.finditer(r'href="(/l/\?[^"]+)"', html_text or ""):
            try:
                qs = parse_qs(m.group(1).split("?", 1)[-1])
                target = qs.get("uddg", [None])[0]
                if target and target.startswith("http"):
                    urls.append(unquote(target))
            except Exception:
                pass
        for m in re.finditer(r'href="(https?://[^"]+)"', html_text or ""):
            urls.append(unquote(m.group(1)))

    # De‑dup
    return _dedupe_keep_order(urls)


# -----------------------
# Public retriever (simple, as requested)
# -----------------------

class CuriaEurlexRetriever:
    """
    Minimal web retriever:

    1) Send the raw user query to Google and DuckDuckGo HTML.
