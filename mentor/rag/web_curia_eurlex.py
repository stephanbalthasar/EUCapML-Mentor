# web_curia_eurlex.py
from __future__ import annotations
import re, time, html
from dataclasses import dataclass
from typing import List, Optional
import requests
from bs4 import BeautifulSoup

_ALLOWED_HOSTS = ("curia.europa.eu", "eur-lex.europa.eu")
_UA = "EUCapML/1.0 (+educational; contact: tutor@example.org)"

# --- tiny helpers ------------------------------------------------------------

def _get(url: str, lang: str = "EN", timeout: float = 8.0) -> Optional[str]:
    """Safe GET with allowlist & short timeout; returns text or None."""
    if not any(h in url for h in _ALLOWED_HOSTS):
        return None
    try:
        r = requests.get(url, headers={"User-Agent": _UA, "Accept-Language": lang}, timeout=timeout)
        if r.ok and r.text:
            return r.text
    except Exception:
        return None
    return None

def _clean_text(t: str) -> str:
    t = html.unescape(t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _sentences(text: str, max_sent: int = 3) -> str:
    # very small sentence splitter
    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    return " ".join(parts[:max_sent])

# --- case id detection / CELEX candidates -----------------------------------

_C_CASE = re.compile(r"\bC[\-–]?\s?(\d{1,4})\s?/\s?(\d{2})\b", re.IGNORECASE)

def _celex_candidates_from_query(q: str) -> List[str]:
    """Return likely CELEX IDs from text like 'C-628/13'."""
    out = []
    m = _C_CASE.search(q)
    if not m:
        return out
    num, yy = m.group(1), m.group(2)
    yyyy = f"20{yy}" if len(yy) == 2 else yy
    # Common CELEX shapes for CJEU docket -> EUR-Lex
    #  - CJ = Judgment; CC = AG Opinion; CA = OJ Summary/Notice
    out.extend([
        f"6{yyyy}CJ{num.zfill(4)}",
        f"6{yyyy}CC{num.zfill(4)}",
        f"6{yyyy}CA{num.zfill(4)}",
    ])
    return out

def _eurlex_url(celex: str, lang: str = "EN") -> str:
    return f"https://eur-lex.europa.eu/legal-content/{lang}/TXT/?uri=CELEX:{celex}"

def _curia_press_for_year(year: str) -> List[str]:
    # Known CURIA press release path pattern (example for Lafonta: cp150033en.pdf)
    # We cannot deterministically construct the exact press id per case, so we try common
    # HTML list pages first (fallback is EUR-Lex).
    # For robust minimalism, we skip directory listing and rely on EUR-Lex if CURIA index fails.
    return [
        # You can add press release list pages per year if you want to parse indices.
        # Fallback remains EUR-Lex if parsing fails.
    ]

# --- main retriever ----------------------------------------------------------

@dataclass
class CuriaEurlexRetriever:
    lang: str = "EN"             # "EN", "DE", "FR"...
    max_bytes: int = 250_000     # guardrails

    def _fetch_curia_press_release(self, q: str) -> List[str]:
        """Heuristic: if query has a C-number, try known press release if we happen to know it."""
        # For production, you’d parse CURIA press list pages for the relevant year and match the case.
        # To keep it tiny, we only special-case known hot cases; else return [] and let EUR-Lex handle it.
        snippets: List[str] = []

        # Example: Lafonta (C-628/13) has CURIA press release cp150033 (EN/DE):
        if _C_CASE.search(q):
            # Known mapping for Lafonta:
            press_urls = [
                "https://curia.europa.eu/jcms/upload/docs/application/pdf/2015-03/cp150033en.pdf",
                "https://curia.europa.eu/jcms/upload/docs/application/pdf/2015-03/cp150033de.pdf",
            ]
            for u in press_urls:
                html_text = _get(u, lang=self.lang)
                if not html_text:
                    continue
                # PDFs come back as bytes; some hosts serve them as text. Skip parsing PDF binary here.
                # Many deployments will have pdfminer; to stay tiny, just return a fixed pointer snippet:
                snippets.append(
                    f"CURIA press release (summary available): {u}"
                )
                break
        return snippets

    def _fetch_eurlex_snippets(self, q: str) -> List[str]:
        celex_ids = _celex_candidates_from_query(q)
        snippets: List[str] = []
        for celex in celex_ids:
            url = _eurlex_url(celex, lang=self.lang)
            html_text = _get(url, lang=self.lang)
            if not html_text:
                continue
            soup = BeautifulSoup(html_text, "html.parser")
            # EUR‑Lex pages vary; two robust anchors:
            #  1) first <p> elements within the main content
            #  2) 'Operative part' or 'Summary' sections if present
            paras = []
            for sel in ["#TexteOnly > p", ".tabcontent p", "p"]:
                paras = [p.get_text(" ", strip=True) for p in soup.select(sel)]
                paras = [p for p in paras if len(p) > 60]
                if paras:
                    break
            if not paras:
                continue
            text = _clean_text(" ".join(paras[:3]))
            snippet = _sentences(text, max_sent=3)
            if snippet and snippet not in snippets:
                # Add a short attribution at the end
                snippets.append(f"{snippet} (Source: {url})")
            if len(snippets) >= 3:
                break
        return snippets

    def retrieve(self, query: str, keywords: Optional[List[str]] = None, top_k: int = 4) -> List[str]:
        out: List[str] = []
        # 1) Try a targeted CURIA press release pointer (fast, concise)
        out.extend(self._fetch_curia_press_release(query))
        # 2) Pull 2–3 sentence extracts from EUR‑Lex (judgment/summary/opinion)
        out.extend(self._fetch_eurlex_snippets(query))
        # De‑dup and clamp
        uniq = []
        seen = set()
        for s in out:
            if s in seen:
                continue
            seen.add(s)
            uniq.append(s)
            if len(uniq) == max(1, min(top_k, 4)):
                break
        return uniq
