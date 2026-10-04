"""RAG / agent / multimodal tests. Agent and vision use SCRIPTED STUB providers (no live LLM).
Synthetic states are test fixtures only. Retrieval and rendering tests use the real PDFs."""
import json
from pathlib import Path

import pytest

from paszel.agent import AgentError, describe_id, run_agent
from paszel.ai import AnthropicProvider, OpenAICompatProvider, ingest_pdf
from paszel.engine import Situation, ValidationError, validate_claim
from paszel.retrieval import search
from paszel.vision import render_page, vision_check

RAW = Path(__file__).parent.parent / "benchmark" / "raw"
PDF = RAW / "Sitrep_03-08-2025.pdf"
real = pytest.mark.skipif(not PDF.exists(), reason="real PDF absent")


class Script:
    def __init__(self, *msgs): self.msgs, self.images = list(msgs), []
    def complete(self, system, user, images=()):
        self.images.append(images); return self.msgs.pop(0)


def synth_state():  # SYNTHETIC fixture
    s = Situation()
    d = {"document_id": "A", "source": "SrcA", "filename": "A.pdf", "publication_date": "2000-01-01",
         "retrieval_date": "2000-01-02", "sha256": "0" * 64, "source_url": "test://", "file_size": 0}
    s.add_document(d)
    s.d["pages"]["A"] = {"1": "SYNTHETIC. Area X: 120 houses were damaged."}
    c = {"incident_title": "SYNTHETIC", "incident_type": "flood", "location": "Area X", "subject": "Area X",
         "predicate": "houses_damaged", "value": 120, "page": 1, "quote": "120 houses were damaged",
         "heading": "SYNTHETIC", "as_of": "2000-01-01"}
    s.ingest_claim(validate_claim(c, s.d["pages"]["A"] and {1: s.d["pages"]["A"]["1"]}, None), d)
    return s


@real
def test_bm25_finds_the_right_real_page():
    class Stub:
        def complete(self, system, user, images=()): return json.dumps({"claims": []})
    s = Situation()
    ingest_pdf(s, Stub(), PDF.read_bytes(), PDF.name, "PDMA Sindh", "u", "2025-08-03")
    top = search(s.d, "Tharparkar goat sheep livestock perished", k=3)
    # Real behaviour: the empty "last 24 hours" livestock table (page 2) can outrank the cumulative one on generic
    # words, so we only assert the right chunk is within the top 3, not first (known BM25 limitation).
    assert any("Tharparkar" in r["text"] and r["page"] == 4 for r in top)
    assert search(s.d, "zzzz qqqq") == [] and search(s.d, "") == []


def test_agent_answers_with_verified_citations_and_is_read_only():
    s = synth_state(); before = s.dumps()
    p = Script(json.dumps({"tool": "search_evidence", "args": {"query": "houses damaged"}}),
               json.dumps({"final": "Source A page 1 reports 120 houses damaged.", "citations": ["A:1:0"]}))
    out = run_agent(p, s, "How many houses?")
    assert out["citations"] == ["A:1:0"] and out["trace"][0]["tool"] == "search_evidence"
    assert describe_id(s, "A:1:0")["page"] == 1 and s.dumps() == before  # nothing changed


def test_agent_rejects_bad_answers():
    s = synth_state()
    with pytest.raises(AgentError, match="not returned by any tool"):  # cites an id no tool returned
        run_agent(Script(json.dumps({"tool": "get_current_claims", "args": {}}),
                         json.dumps({"final": "x", "citations": ["A:9:9"]})), s, "q")
    with pytest.raises(AgentError, match="not returned"):  # no tool used at all, uncited
        run_agent(Script(json.dumps({"final": "120 houses", "citations": []})), s, "q")
    with pytest.raises(AgentError, match="not valid JSON"):
        run_agent(Script("I think it is 120"), s, "q")
    with pytest.raises(AgentError, match="exceeded"):  # endless tool calls
        run_agent(Script(*[json.dumps({"tool": "get_history", "args": {}})] * 10), s, "q", max_steps=2)
    with pytest.raises(AgentError, match="unknown tool"):  # cannot call anything that writes
        run_agent(Script(json.dumps({"tool": "verify", "args": {}})), s, "q")
    out = run_agent(Script(json.dumps({"final": "No relevant evidence.", "insufficient": True})), s, "q")
    assert out["insufficient"] and out["citations"] == []


def test_providers_send_images(monkeypatch):
    monkeypatch.setenv("PASZEL_LLM_API_KEY", "k"); monkeypatch.setenv("PASZEL_LLM_MODEL", "m")
    seen = {}
    a = AnthropicProvider(); a._post = lambda u, h, b: seen.update(a=b) or {"content": [{"text": "{}"}]}
    a.complete("s", "u", images=[b"\x89PNGxx"])
    blk = seen["a"]["messages"][0]["content"]
    assert blk[0]["type"] == "image" and blk[0]["source"]["media_type"] == "image/png" and blk[1]["type"] == "text"
    o = OpenAICompatProvider(); o._post = lambda u, h, b: seen.update(o=b) or {"choices": [{"message": {"content": "{}"}}]}
    o.complete("s", "u", images=[b"\x89PNGxx"])
    assert seen["o"]["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    o.complete("s", "plain")
    assert seen["o"]["messages"][1]["content"] == "plain"  # text-only path unchanged


@real
def test_render_page_png_and_highlight_reporting():
    png, match = render_page(PDF.read_bytes(), 1, "DAILY SITUATION REPORT")
    assert png[:4] == b"\x89PNG" and match == "full"
    png2, match2 = render_page(PDF.read_bytes(), 4, "this text is not on the page at all")
    assert png2[:4] == b"\x89PNG" and match2 == "none"


def test_vision_check_is_validated():
    ok = Script(json.dumps({"verdict": "supported", "reason": "row shows 120"}))
    assert vision_check(ok, b"png", "claim")["verdict"] == "supported" and ok.images[0] == [b"png"]
    with pytest.raises(ValidationError): vision_check(Script(json.dumps({"verdict": "definitely"})), b"png", "c")
    with pytest.raises(ValidationError): vision_check(Script("looks right"), b"png", "c")
