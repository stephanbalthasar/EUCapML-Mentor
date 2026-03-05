# scripts/build_booklet_index.py
"""
Build a JSON index for the EUCapML booklet that:
- Detects Chapter headings robustly (localized/custom Heading 1)
- Iterates paragraphs AND tables in document order
- Attaches footnotes to their host paragraphs (stored separately, not inlined)
- Produces a JSON with keys: {"paragraphs": [...], "chapters": [...]}

Paragraph record includes (new fields at the end):
{
  "para_num": int,
  "id": "p_<hash8>",
  "text": "Paragraph text only (no inlined footnotes)",
  "chapter_num": int | null,
  "chapter_title": str | null,
  "style": "Word style name",
  "is_numbered": bool,
  "source": "p" | "tbl",
  "footnote_refs": [<int>, ...],             # NEW
  "footnotes": [{"id": <int>, "text": "..."}]# NEW
}
"""

import argparse
import json
import sys
from hashlib import blake2b
from pathlib import Path
from typing import Iterator, Union, List, Dict, Any

from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml.ns import qn
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph


# ---------- small utilities ----------

def _hash_id(text: str) -> str:
    h = blake2b(digest_size=6)
    h.update(text.strip().encode("utf-8"))
    return "p_" + h.hexdigest()

def _iter_block_items(doc: Document) -> Iterator[Union[Paragraph, Table]]:
    """
    Yield paragraphs and tables in document order (document body level).
    """
    body = doc.element.body
    for child in body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, doc)
        elif isinstance(child, CT_Tbl):
            yield Table(child, doc)

def _table_text(tbl: Table) -> List[str]:
    """
    Extract row-major text from a table as lines. Cells in a row are joined with " | ".
    """
    lines: List[str] = []
    for row in tbl.rows:
        cells = []
        for cell in row.cells:
            t = (cell.text or "").strip()
            if t:
                cells.append(" ".join(t.split()))
        if cells:
            lines.append(" | ".join(cells))
    return lines

def _is_heading1(par: Paragraph) -> bool:
    """
    Robustly detect Heading 1, including localized/custom base styles.
    """
    name = (par.style.name or "").strip().lower()

    # Common localized names for Heading 1
    h1_names = {
        "heading 1", "überschrift 1", "titre 1", "título 1",
        "intestazione 1", "rubrica 1", "rubrique 1"
    }
    if name in h1_names:
        return True

    # Generic fallback: startswith token + " 1"
    startswith_tokens = ("heading", "überschrift", "titre", "título", "rubrica", "rubrique", "intestazione")
    if any(name.startswith(tok + " ") and name.endswith("1") for tok in startswith_tokens):
        return True

    # Climb base styles to detect inheritance from a Heading 1 style (best-effort)
    try:
        base = par.style.base_style
        while base is not None:
            bname = (base.name or "").strip().lower()
            if bname in h1_names:
                return True
            if any(bname.startswith(tok + " ") and bname.endswith("1") for tok in startswith_tokens):
                return True
            base = base.base_style
    except Exception:
        pass

    return False

def _is_numbered_paragraph(par: Paragraph) -> bool:
    """
    Detect list numbering from Word numbering (numPr), not from style name.
    """
    try:
        pPr = par._p.pPr
        return (pPr is not None) and (pPr.numPr is not None)
    except Exception:
        return False


# ---------- footnotes support ----------

def _build_footnote_map(doc: Document) -> Dict[int, str]:
    """
    Build {footnote_id: "footnote text"} from the DOCX footnotes part.
    Skips separators/continuation footnotes. Returns {} if none present.
    """
    try:
        fn_part = doc.part.part_related_by(RT.FOOTNOTES)
    except KeyError:
        return {}

    ns = {"w": fn_part.element.nsmap["w"]}
    fn_map: Dict[int, str] = {}

    # Only real footnotes: <w:footnote> without @w:type
    for fn in fn_part.element.xpath("w:footnote[not(@w:type)]", namespaces=ns):
        try:
            fid = int(fn.get(qn("w:id")))
        except (TypeError, ValueError):
            continue

        chunks: List[str] = []
        for p in fn.xpath(".//w:p", namespaces=ns):
            # Join all text nodes inside the paragraph
            t = "".join(p.itertext())
            t = " ".join(t.split()).strip()
            if t:
                chunks.append(t)

        txt = " ".join(chunks).strip()
        if txt:
            fn_map[fid] = txt

    return fn_map

def _footnote_ids_for_paragraph(par: Paragraph) -> List[int]:
    """
    Find all <w:footnoteReference w:id="..."> IDs referenced in this paragraph.
    """
    ids: List[int] = []
    try:
        ns = {"w": par._p.nsmap["w"]}
        for ref in par._p.xpath(".//w:footnoteReference", namespaces=ns):
            fid = ref.get(qn("w:id"))
            if fid is not None:
                try:
                    ids.append(int(fid))
                except ValueError:
                    pass
    except Exception:
        pass
    return ids


# ---------- main builder ----------

def build_index(docx_path: str, verbose: bool = False) -> Dict[str, Any]:
    doc = Document(docx_path)

    # Build the global footnote text map once
    footnote_map = _build_footnote_map(doc)

    chapters: List[Dict[str, Any]] = []
    paragraphs: List[Dict[str, Any]] = []

    chapter_num = 0
    chapter_title = None
    chapter_buf: List[str] = []
    chapter_para_ids: List[str] = []

    para_counter = 0

    def _flush_chapter():
        nonlocal chapters, chapter_num, chapter_title, chapter_buf, chapter_para_ids
        if chapter_title is not None:
            chapters.append({
                "chapter_num": chapter_num,
                "title": chapter_title,
                "text": "\n".join(chapter_buf).strip(),
                "paragraph_ids": list(chapter_para_ids),  # helpful for UI/debug
            })

    for block in _iter_block_items(doc):
        # Paragraphs
        if isinstance(block, Paragraph):
            text = (block.text or "").strip()
            style = (block.style.name or "").strip()

            # New chapter starts
            if _is_heading1(block):
                _flush_chapter()
                chapter_num += 1  # start at 1
                chapter_title = text or f"Chapter {chapter_num}"
                chapter_buf = []
                chapter_para_ids = []
                if verbose:
                    print(f"[H1] #{chapter_num}: {chapter_title}")
                continue  # the heading itself is not a content paragraph

            # Skip empty non-heading lines
            if not text:
                continue

            para_counter += 1
            pid = _hash_id(f"{chapter_num}|{text}|{style}|p{para_counter}")

            # Detect numbering and footnote refs
            is_num = _is_numbered_paragraph(block)
            fn_ids = _footnote_ids_for_paragraph(block)
            fn_payload = [{"id": i, "text": footnote_map.get(i, "")} for i in fn_ids if i in footnote_map]

            record = {
                "para_num": para_counter,
                "id": pid,
                "text": text,
                "chapter_num": chapter_num or None,
                "chapter_title": chapter_title if chapter_num else None,
                "style": style,
                "is_numbered": bool(is_num),
                "source": "p",
            }
            if fn_ids:
                record["footnote_refs"] = fn_ids
                record["footnotes"] = fn_payload

            paragraphs.append(record)

            # Accumulate into current chapter
            if chapter_num:
                chapter_buf.append(text)
                chapter_para_ids.append(pid)

        # Tables
        else:
            lines = _table_text(block)
            if not lines:
                continue
            for line in lines:
                para_counter += 1
                pid = _hash_id(f"{chapter_num}|{line}|tbl{para_counter}")
                record = {
                    "para_num": para_counter,
                    "id": pid,
                    "text": line,
                    "chapter_num": chapter_num or None,
                    "chapter_title": chapter_title if chapter_num else None,
                    "style": "Table",
                    "is_numbered": False,
                    "source": "tbl",
                }
                # (Usually tables don't carry footnote refs; skip scanning for perf)
                paragraphs.append(record)
                if chapter_num:
                    chapter_buf.append(line)
                    chapter_para_ids.append(pid)

    # Close final chapter
    _flush_chapter()

    return {"paragraphs": paragraphs, "chapters": chapters}


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Build booklet index with footnotes attached to paragraphs")
    ap.add_argument("--src", default="assets/booklet.docx", help="Path to source DOCX")
    ap.add_argument("--out", default="artifacts/booklet_index.json", help="Path to output JSON")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = ap.parse_args(argv)

    src = Path(args.src)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        print(f"ERROR: source DOCX not found: {src}", file=sys.stderr)
        return 2

    index = build_index(str(src), verbose=args.verbose)
    ch = len(index.get("chapters", []))
    ps = len(index.get("paragraphs", []))
    print(f"Indexed {ch} chapters, {ps} paragraphs from {src}")

    out.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
