"""PLUMBING TEST - NOT an LLM-quality test.
Real PDMA Sindh PDFs (benchmark/raw) -> real PyMuPDF text -> real validator/engine.
The 'LLM' is a STUB returning 4 hand-selected claims copied from the real page text, because no LLM
was available. It proves the pipeline mechanics on real text, nothing about extraction quality."""
import json
from pathlib import Path

import pytest

from paszel.ai import analyze_report, doc_meta, extract_pages
from paszel.engine import Situation

RAW = Path(__file__).parent.parent / "benchmark" / "raw"
FILES = [("Sitrep_03-08-2025.pdf", "2025-08-03"), ("Sitrep_04-08-2025.pdf", "2025-08-04"),
         ("Sitrep_05-08-2025.pdf", "2025-08-05")]
pytestmark = pytest.mark.skipif(not all((RAW / f).exists() for f, _ in FILES), reason="real PDFs absent")

base = {"incident_title": "Monsoon 2025 - Sindh", "incident_type": "flood", "location": "Sindh"}
STUB = [
    {**base, "subject": "Sindh", "predicate": "deaths", "value": 28, "page": 3, "quote": "Deaths 28 Injured 40", "heading": "Cumulative Causalities", "as_of": None},
    {**base, "subject": "Sindh", "predicate": "injured", "value": 40, "page": 3, "quote": "Deaths 28 Injured 40", "heading": "Cumulative Causalities", "as_of": None},
    {**base, "subject": "Hyderabad", "predicate": "houses_damaged", "value": 83, "page": 4, "quote": "Hyderabad - - 50 33 83", "heading": "Cumulative Damages of Infrastructure & Private Properties", "as_of": None},
    {**base, "subject": "Tharparkar", "predicate": "livestock_lost", "value": 71, "page": 4, "quote": "Tharparkar 60 11 - - - - 71", "heading": "Cumulative Livestock Perished", "as_of": None},
    {**base, "subject": "Sindh", "predicate": "deaths", "value": 28, "page": 3, "quote": "this text is not on the page", "heading": "x", "as_of": None},  # must be rejected
]


class Stub:
    def complete(self, system, user):
        return json.dumps({"claims": STUB})


def test_real_pdfs_through_pipeline():
    s, results = Situation(), []
    for fn, pub in FILES:
        data = (RAW / fn).read_bytes()
        pages, errs = extract_pages(data)
        assert not errs and len(pages) == 7
        meta = doc_meta(data, fn, "PDMA Sindh", "https://pdma.gos.pk/", pub)
        assert s.add_document(meta)
        assert not s.add_document(meta)  # same SHA-256 refused
        ok, bad = analyze_report(Stub(), pages, pub)
        assert len(ok) == 4 and len(bad) == 1 and "quote not found" in bad[0][1]
        results.append([s.ingest_claim(c, meta)["transition"] for c in ok])
    assert results[0] == ["CREATE"] * 4
    assert results[1] == ["DUPLICATE"] * 4 and results[2] == ["DUPLICATE"] * 4
    assert not s.d["proposals"] and len(s.d["incidents"]) == 1
    assert all(c["status"] == "active" for c in s.d["claims"].values())
