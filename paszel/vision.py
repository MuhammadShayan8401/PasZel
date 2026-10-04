"""Multimodal helpers: render a PDF page (quote highlighted) and an ADVISORY vision-model check."""
import json
import re

import pymupdf

from .engine import ValidationError

VISION_SYSTEM = """You check ONE claim against a page image of a disaster report. Reply with ONLY JSON:
{"verdict": "supported" | "not_supported" | "unclear", "reason": "one sentence"}.
'supported' only if the image itself shows that number for that subject with that meaning (check the row and column
headings). Use 'unclear' if you cannot read it. Do not use outside knowledge."""


def render_page(pdf_bytes: bytes, page_no: int, quote: str | None = None, zoom: float = 1.6):
    """Returns (png_bytes, match) with match in {'full','partial','none'}. Works on an in-memory copy."""
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        pg, match = doc[page_no - 1], "none"
        words = (quote or "").split()
        for n in dict.fromkeys([len(words), 6, 4, 3, 2]):
            if 0 < n <= len(words) and (rects := pg.search_for(" ".join(words[:n]))):
                pg.add_highlight_annot(rects)
                match = "full" if n == len(words) else "partial"
                break
        return pg.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), annots=True).tobytes("png"), match


def vision_check(provider, png: bytes, claim_desc: str) -> dict:
    raw = provider.complete(VISION_SYSTEM, claim_desc, images=[png])
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        out = json.loads(m.group(0))
    except (AttributeError, json.JSONDecodeError):
        raise ValidationError("vision output was not valid JSON") from None
    if out.get("verdict") not in ("supported", "not_supported", "unclear"):
        raise ValidationError("vision verdict not in allowed set")
    return {"verdict": out["verdict"], "reason": str(out.get("reason", ""))[:300]}
