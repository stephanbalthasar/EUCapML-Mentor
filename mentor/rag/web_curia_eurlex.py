"""
Simple DuckDuckGo-based web retriever for legal research.
No complex parsing, no Google detection, just straightforward search + extraction.
"""

from __future__ import annotations

import html
import io
import re
from typing import List
from urllib.parse import unquote

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


# -----------------------
# Config
# -----------------------

_SNIPPET_LEN = 300      # characters per snippet
_MAX_RESULTS = 5        # number of links to fetch
_TIMEOUT = 5.0          # seconds per page fetch
_RETRIES = 2            # retry failed fetches


def _ua() -> str:
    """Return a simple desktop user agent."""
    return (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )


# -----------------------
# Text extraction
# -----------------------

def _extract_text_from_pdf(content: bytes) -> str:
    """Extract text from first 2 pages of a PDF."""
    if not content or PdfReader is None:
        return ""
    try:
        reader = PdfReader(io.BytesIO(content))
        pages = min(2, len(reader.pages))
        out = []
        for i in range(pages):
            try:
                text = reader.pages[i].extract_text()
                if text:
                    out.append(text)
            except Exception:
                pass
        return " ".join(out)
    except Exception:
        return ""


def _extract_text_from_html(content: bytes) -> str:
    """Extract visible text from HTML."""
    try:
        text = content.decode("utf-8", errors="replace")
    except Exception:
        text = content.decode("latin-1", errors="replace")

    if BeautifulSoup is None:
        # Fallback: rough regex-based tag stripping
        text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    try:
        soup = BeautifulSoup(text, "html.parser")
        
        # Remove script and style tags
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Try to get paragraphs first
        body = soup.body or soup
        paras: List[str] = []
        
        for p in body.find_all("p"):
            t = p.get_text(" ", strip=True)
            if t and len(t.split()) > 4:
                paras.append(t)
            if len(paras) >= 8:
                break
        
        if paras:
            return " ".join(paras)
        
        return body.get_text(" ", strip=True)
    except Exception:
        # Fallback to simple regex
        text = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", text).strip()


def _to_snippet(text: str, *, limit: int = _SNIPPET_LEN) -> str:
    """Normalize whitespace and trim to limit; try to end on a sentence."""
    t = html.unescape(text or "")
    t = re.sub(r"\s+", " ", t).strip()
    
    if len(t) <= limit:
        return t

    # Try to end at a sentence boundary
    window_start = max(0, limit - 120)
    window = t[window_start:limit]

    last_end = -1
    for m in re.finditer(r"[.!?](?=\s|$)", window):
        last_end = m.end()
    
    if last_end != -1:
        end = window_start + last_end
        return t[:end].strip()

    return t[:limit].strip()


# -----------------------
# HTTP fetching
# -----------------------

def _fetch_url(url: str) -> tuple[int, bytes]:
    """
    Fetch a URL with retries and return (status_code, content).
    Returns (0, b"") on failure.
    """
    import requests
    
    for attempt in range(_RETRIES):
        try:
            r = requests.get(
                url,
                headers={"User-Agent": _ua(), "Accept": "text/html,*/*"},
                timeout=_TIMEOUT,
                allow_redirects=True,
            )
            if r.status_code == 200:
                return r.status_code, (r.content or b"")
            elif attempt < _RETRIES - 1:
                # Retry on non-200
                continue
            else:
                return r.status_code, b""
        except Exception as e:
            if attempt < _RETRIES - 1:
                continue
            return 0, b""
    
    return 0, b""


# -----------------------
# Public retriever
# -----------------------

class CuriaEurlexRetriever:
    """
    Web retriever using DuckDuckGo search.
    
    1) Search DuckDuckGo for the query
    2) Fetch top N results
    3) Extract visible text from each page
    4) Return snippets with sources
    """

    def __init__(self, lang: str = "EN", timeout_sec: float = 5.0):
        self.lang = (lang or "EN").upper()
        self.timeout = float(timeout_sec)

    def retrieve(
        self, 
        query: str, 
        keywords: List[str] | None = None, 
        top_k: int = 3
    ) -> List[str]:
        """
        Retrieve snippets for a query.
        
        Args:
            query: Search query
            keywords: Ignored (for API compatibility)
            top_k: Maximum snippets to return
        
        Returns:
            List of strings like "snippet text...\n(Source: URL)"
        """
        if not (query or "").strip():
            return []

        if DDGS is None:
            return []

        # --- Search DuckDuckGo ---
        urls: List[str] = []
        try:
            ddgs = DDGS(timeout=10)
            results = ddgs.text(query, max_results=_MAX_RESULTS)
            urls = [r["href"] for r in results if r.get("href")]
        except Exception as e:
            # Silent fail - no results available
            return []

        if not urls:
            return []

        # --- Fetch and extract snippets ---
        out: List[str] = []
        
        for url in urls:
            status, content = _fetch_url(url)
            
            if status != 200 or not content:
                continue

            # Determine if PDF or HTML
            is_pdf = "pdf" in url.lower() or (
                len(content) > 4 and content[:4] == b"%PDF"
            )

            if is_pdf:
                text = _extract_text_from_pdf(content)
            else:
                text = _extract_text_from_html(content)

            if not text:
                continue

            snippet = _to_snippet(text, limit=_SNIPPET_LEN)
            if not snippet:
                continue

            out.append(f"{snippet}\n(Source: {url})")
            
            if len(out) >= top_k:
                break

        return out
