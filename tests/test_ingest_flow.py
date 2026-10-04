"""Flow tests. Real PDFs + STUB extractor (plumbing only) and a UI check of the Ingest page."""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from paszel.ai import ingest_pdf
from paszel.engine import Situation, ValidationError

RAW = Path(__file__).parent.parent / "benchmark" / "raw" / "Sitrep_03-08-2025.pdf"
CLAIM = {"incident_title": "Monsoon 2025 - Sindh", "incident_type": "flood", "location": "Sindh", "subject": "Sindh",
         "predicate": "deaths", "value": 28, "page": 3, "quote": "Deaths 28 Injured 40",
         "heading": "Cumulative Causalities", "as_of": None}


class Stub:
    def __init__(self, text=None): self.text = text
    def complete(self, system, user): return self.text or json.dumps({"claims": [CLAIM]})


@pytest.mark.skipif(not RAW.exists(), reason="real PDF absent")
def test_ingest_pdf_whole_flow_and_failures():
    s, data = Situation(), RAW.read_bytes()
    out = ingest_pdf(s, Stub(), data, RAW.name, "PDMA Sindh", "https://example.invalid/", "2025-08-03")
    assert [r["transition"] for r in out["rows"]] == ["CREATE"] and not out["rejected"]
    with pytest.raises(ValidationError, match="already ingested"):
        ingest_pdf(s, Stub(), data, RAW.name, "PDMA Sindh", "u", "2025-08-03")
    s2 = Situation()  # malformed LLM output must fail loudly and leave state untouched
    with pytest.raises(ValidationError, match="not valid"):
        ingest_pdf(s2, Stub("sorry, I cannot"), data, RAW.name, "PDMA Sindh", "u", "2025-08-03")
    assert not s2.d["documents"] and not s2.d["claims"]


def test_ingest_page_shows_what_is_missing_and_disables_button(tmp_path, monkeypatch):
    monkeypatch.setenv("PASZEL_STATE", str(tmp_path / "s.json"))
    at = AppTest.from_file(str(Path(__file__).parent.parent / "app.py"), default_timeout=30).run()
    at.sidebar.radio[0].set_value("Ingest report").run()
    assert not at.exception
    assert "Still needed" in at.info[0].value and "PDF file" in at.info[0].value
    assert at.button[0].disabled
