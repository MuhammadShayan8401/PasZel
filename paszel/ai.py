"""LLM provider interface, report-analysis agent, and PDF ingestion.

Provider is chosen by env vars (no vendor hard-coded elsewhere):
  PASZEL_LLM_PROVIDER = anthropic | openai   (openai = any OpenAI-compatible endpoint)
  PASZEL_LLM_MODEL    = model name (required)
  PASZEL_LLM_API_KEY  = key
  PASZEL_LLM_BASE_URL = optional override
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from datetime import date

from .engine import PREDICATES, ValidationError, validate_claim

SYSTEM = f"""You extract disaster-situation claims from one report. Output ONLY JSON:
{{"claims":[{{"incident_title":"","incident_type":"flood|earthquake|cyclone|...","location":"",
"subject":"area/group the number refers to","predicate":"one of {sorted(PREDICATES)}",
"value":123,"page":1,"quote":"exact text copied verbatim from that page containing the number (for a table, the row)",
"heading":"verbatim column heading or section title from the SAME page, appearing before the quote, that shows what the number means",
"as_of":"YYYY-MM-DD or YYYY-MM-DDTHH:MM if the report states when the figure is as of, else null"}}]}}
Rules: only numbers explicitly stated; copy quotes verbatim; no estimates; no inference; skip anything unclear."""


def _b64(png: bytes) -> str:
    return base64.b64encode(png).decode()


def _content_anthropic(user, images):
    return user if not images else [*({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _b64(i)}} for i in images), {"type": "text", "text": user}]


def _content_openai(user, images):
    return user if not images else [{"type": "text", "text": user}, *({"type": "image_url", "image_url": {"url": "data:image/png;base64," + _b64(i)}} for i in images)]


class LLMProvider:
    def complete(self, system: str, user: str, images=()) -> str:
        raise NotImplementedError


class _HTTPProvider(LLMProvider):
    def __init__(self):
        self.key = os.environ.get("PASZEL_LLM_API_KEY", "")
        self.model = os.environ.get("PASZEL_LLM_MODEL", "")
        if not (self.key and self.model):
            raise RuntimeError("Set PASZEL_LLM_API_KEY and PASZEL_LLM_MODEL (see .env.example).")

    def _post(self, url, headers, body) -> dict:
        req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:  # surface the provider's reason (never includes our key)
            raise RuntimeError(f"LLM HTTP {e.code}: {e.read().decode(errors='replace')[:400]}") from e


class EmptyLLMReply(RuntimeError):
    pass


def _openai_text(out) -> str:
    ch = out.get("choices") if isinstance(out, dict) else None
    if not ch:
        raise RuntimeError(f"LLM returned no choices: {str(out)[:300]}")
    msg = ch[0].get("message") or {}
    c = msg.get("content")
    if isinstance(c, list):  # some servers return content parts
        c = "".join(x.get("text", "") for x in c if isinstance(x, dict))
    if not c or not str(c).strip():
        raise EmptyLLMReply(f"LLM returned an empty reply (finish_reason={ch[0].get('finish_reason')!r}, message fields={sorted(msg)}). "
                            "The model may have been cut off or blocked; try again, shorten the question, or use a different model.")
    return c


class AnthropicProvider(_HTTPProvider):
    def complete(self, system, user, images=()):
        base = os.environ.get("PASZEL_LLM_BASE_URL", "https://api.anthropic.com").rstrip("/")
        out = self._post(f"{base}/v1/messages", {"x-api-key": self.key, "anthropic-version": "2023-06-01"},
                         {"model": self.model, "max_tokens": 4000, "system": system,
                          "messages": [{"role": "user", "content": _content_anthropic(user, images)}]})
        blocks = out.get("content") if isinstance(out, dict) else None
        text = "".join(b.get("text", "") for b in blocks or [] if isinstance(b, dict))
        if not text.strip():
            raise EmptyLLMReply(f"LLM returned no text (stop_reason={out.get('stop_reason') if isinstance(out, dict) else None!r}).")
        return text


class OpenAICompatProvider(_HTTPProvider):
    def complete(self, system, user, images=()):
        base = os.environ.get("PASZEL_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        body = {"model": self.model, "messages": [{"role": "system", "content": system},
                                                  {"role": "user", "content": _content_openai(user, images)}]}
        for attempt in (1, 2):  # an empty reply is retried once
            out = self._post(f"{base}/chat/completions", {"authorization": f"Bearer {self.key}"}, body)
            try:
                return _openai_text(out)
            except EmptyLLMReply:
                if attempt == 2:
                    raise


def get_provider() -> LLMProvider:
    name = os.environ.get("PASZEL_LLM_PROVIDER", "").lower()
    if name == "anthropic":
        return AnthropicProvider()
    if name == "openai":
        return OpenAICompatProvider()
    raise RuntimeError("Set PASZEL_LLM_PROVIDER to 'anthropic' or 'openai'.")


def parse_llm_json(text: str) -> list[dict]:
    """Strict: malformed output raises ValidationError, never silently accepted."""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        claims = json.loads(m.group(0))["claims"] if m else None
    except (json.JSONDecodeError, KeyError, TypeError):
        claims = None
    if not isinstance(claims, list):
        raise ValidationError("LLM output was not valid {'claims': [...]} JSON")
    return claims


def analyze_report(provider: LLMProvider, pages: dict[int, str], default_time: str):
    """Returns (valid_claims, rejected[(raw_claim, reason)]). Retries once on malformed JSON."""
    user = "\n\n".join(f"=== PAGE {n} ===\n{t}" for n, t in pages.items())[:60000]
    for attempt in (1, 2):
        try:
            raw = parse_llm_json(provider.complete(SYSTEM, user))
            break
        except ValidationError:
            if attempt == 2:
                raise
    ok, bad = [], []
    for c in raw:
        try:
            ok.append(validate_claim(c, pages, default_time))
        except ValidationError as e:
            bad.append((c, str(e)))
    return ok, bad


def extract_pages(pdf_bytes: bytes) -> tuple[dict[int, str], list[str]]:
    """Page-numbered text via PyMuPDF. Errors are returned, not swallowed."""
    import pymupdf as fitz  # PyMuPDF
    pages, errors = {}, []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i, pg in enumerate(doc, 1):
            try:
                pages[i] = pg.get_text()
            except Exception as e:  # noqa: BLE001
                errors.append(f"page {i}: {e}")
    if not any(t.strip() for t in pages.values()):
        errors.append("no extractable text (scanned PDF? OCR is out of scope)")
    return pages, errors


def doc_meta(pdf_bytes, filename, source, source_url, publication_date, retrieval_date=None) -> dict:
    sha = hashlib.sha256(pdf_bytes).hexdigest()
    return {"document_id": sha[:12], "source": source, "source_url": source_url, "filename": filename,
            "publication_date": publication_date, "retrieval_date": retrieval_date or date.today().isoformat(),
            "sha256": sha, "file_size": len(pdf_bytes)}


def load_dotenv_file(path: str = ".env") -> list[str]:
    """Tiny .env reader (no dependency). Real environment variables win. Returns names loaded.
    Tolerates UTF-8 BOM, UTF-16 (PowerShell), CRLF and `export ` prefixes."""
    f, loaded = __import__("pathlib").Path(path), []
    if not f.is_file():
        return loaded
    raw = f.read_bytes()
    for enc in ("utf-8-sig", "utf-16"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        return loaded
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.split(" #")[0].strip().strip("\"'")
        if k.strip() and k.strip() not in os.environ:
            os.environ[k.strip()] = v
            loaded.append(k.strip())
    return loaded


def ingest_pdf(sit, provider, data: bytes, filename, source, url, pub_iso, derived="", retrieval_date=None, extraction_note=None) -> dict:
    """Whole flow: hash -> extract pages -> LLM claims -> validate -> state. Raises on hard failure."""
    meta = doc_meta(data, filename, source, url, pub_iso, retrieval_date)
    meta["extraction"] = extraction_note or f"LLM-extracted ({os.environ.get('PASZEL_LLM_MODEL', 'unknown model')})"
    meta["derived_from"] = (derived or "").strip()
    if meta["document_id"] in sit.d["documents"]:
        raise ValidationError("This exact file (same SHA-256) was already ingested.")
    pages, errs = extract_pages(data)
    if not any(t.strip() for t in pages.values()):
        raise ValidationError("; ".join(errs) or "PDF has no pages")
    ok, bad = analyze_report(provider, pages, pub_iso)
    sit.add_document(meta)
    sit.d.setdefault("pages", {})[meta["document_id"]] = {str(n): t for n, t in pages.items()}
    rows = []
    for c in ok:
        r = sit.ingest_claim(c, meta)
        rows.append({"transition": r["transition"], "subject": c["subject"], "predicate": c["predicate"],
                     "value": c["value"], "page": c["page"], "reason": r["reason"]})
    return {"meta": meta, "errors": errs, "rows": rows, "rejected": bad}
