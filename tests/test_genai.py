"""GenAI feature tests.

IMPORTANT: no live LLM is used. Providers here are SCRIPTED STUBS that return canned or state-derived replies, so these
tests prove the deterministic plumbing and guards (retrieval -> prompt payload -> reply checks -> citation resolution,
read-only behaviour). They do NOT prove how a real model (e.g. Gemini) behaves.
State is built from the REAL supplied PDFs where noted; UPDATE/CONFLICT use a SYNTHETIC fixture because the real PDFs
never produce either.
"""
import json
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from paszel.engine import Situation, ValidationError, validate_claim
from paszel.genai import (GroundingError, ask_paszel, change_facts, cite, explain_change, explainable_changes,
                          situation_brief)
from paszel.seed import seed_available, seed_state

ROOT = Path(__file__).parent.parent
real = pytest.mark.skipif(not seed_available(ROOT) or not (ROOT / "benchmark/raw/Sitrep_03-08-2025.pdf").exists(),
                          reason="real PDFs absent")


class Stub:
    """Scripted provider: `replies` is a list of strings or callables(system, payload_dict) -> str."""
    def __init__(self, *replies): self.replies, self.calls = list(replies), []
    def complete(self, system, user, images=()):
        payload = json.loads(user.split("\n\nYour previous reply")[0])
        self.calls.append((system, payload, user))
        r = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return r(system, payload) if callable(r) else r


class Boom:
    def complete(self, *a, **k): raise AssertionError("LLM must not be called")


def synth(second_time="2000-01-01T11:30", second_val=85, second_src="SrcB"):
    """SYNTHETIC fixture: SrcA says 120 houses; second report differs -> CONFLICT (within 24h) or UPDATE (later)."""
    s = Situation()
    for src, val, when, q in (("SrcA", 120, "2000-01-01T09:00", "120 houses were damaged"),
                              (second_src, second_val, second_time, f"{second_val} houses damaged")):
        pages = {1: f"SYNTHETIC. Area X: 120 houses were damaged. {second_val} houses damaged."}
        d = {"document_id": src + when, "source": src, "filename": src + ".pdf", "publication_date": when[:10],
             "retrieval_date": "x", "sha256": "0" * 64, "source_url": "test://", "file_size": 0}
        s.add_document(d)
        s.d["pages"][d["document_id"]] = {"1": pages[1]}
        c = {"incident_title": "SYNTHETIC", "incident_type": "flood", "location": "Area X", "subject": "Area X",
             "predicate": "houses_damaged", "value": val, "page": 1, "quote": q, "as_of": when, "heading": "SYNTHETIC"}
        s.ingest_claim(validate_claim(c, pages, None), d)
    return s


def _claim_id(sit, subject, predicate):
    return next(c["claim_id"] for c in sit.d["claims"].values() if c["subject"] == subject and c["predicate"] == predicate)


# ------------------------------------------------------------------ Ask PasZel
@real
def test_ask_keeps_source_and_page_on_real_state():
    sit = seed_state(ROOT)[0]
    cid = _claim_id(sit, "Hyderabad", "houses_damaged")
    llm = Stub(json.dumps({"answer": "Hyderabad: 83 houses damaged (PDMA Sindh).", "citations": [cid], "unavailable": False}))
    out = ask_paszel(llm, sit, "What happened in Hyderabad?")
    assert not out["unavailable"]
    assert {(r["source"], r["document"], r["page"]) for r in out["sources"]} == {
        ("PDMA Sindh", f"Sitrep_0{d}-08-2025.pdf", 4) for d in (3, 4, 5)}
    assert all("83" in r["quote"] or "Hyderabad" in r["quote"] for r in out["sources"])
    # the LLM payload was built from state only: every claim in it exists in state with the same value
    ctx = llm.calls[0][1]["context"]
    assert {(c["subject"], c["predicate"], c["value"]) for c in ctx["claims"]} == {
        (c["subject"], c["predicate"], c["value"]) for c in sit.d["claims"].values()}


@real
def test_ask_rejects_ungrounded_replies_on_real_state():
    sit = seed_state(ROOT)[0]
    cid = _claim_id(sit, "Sindh", "deaths")
    # 1 general-knowledge answer, no citation -> rejected (also after the retry)
    with pytest.raises(GroundingError):
        ask_paszel(Stub(json.dumps({"answer": "Islamabad is the capital of Pakistan.", "citations": [], "unavailable": False})),
                   sit, "What is the capital of Pakistan?")
    # 2 invented citation id
    with pytest.raises(GroundingError):
        ask_paszel(Stub(json.dumps({"answer": "28 deaths.", "citations": ["clm-deadbeef"], "unavailable": False})), sit, "deaths?")
    # 3 real citation but a fabricated figure
    with pytest.raises(GroundingError):
        ask_paszel(Stub(json.dumps({"answer": "Sindh deaths are 280.", "citations": [cid], "unavailable": False})), sit, "deaths?")
    # 4 non-JSON
    with pytest.raises(GroundingError):
        ask_paszel(Stub("The death toll is high."), sit, "deaths?")


@real
def test_ask_unavailable_is_accepted_and_retry_can_recover():
    sit = seed_state(ROOT)[0]
    out = ask_paszel(Stub(json.dumps({"answer": "No information about Karachi hospitals is stored.", "citations": [], "unavailable": True})),
                     sit, "How many hospital beds are free in Karachi?")
    assert out["unavailable"] and out["sources"] == []
    cid = _claim_id(sit, "Sindh", "deaths")
    bad = json.dumps({"answer": "Sindh deaths are 280.", "citations": [cid], "unavailable": False})
    good = json.dumps({"answer": "Sindh deaths: 28.", "citations": [cid], "unavailable": False})
    llm = Stub(bad, good)
    out = ask_paszel(llm, sit, "deaths?")
    assert out["answer"] == "Sindh deaths: 28." and len(llm.calls) == 2
    assert "rejected" in llm.calls[1][2] and "280" in llm.calls[1][2]


def test_ask_on_empty_state_never_calls_llm():
    out = ask_paszel(Boom(), Situation(), "What is the current situation?")
    assert out["unavailable"] and out["sources"] == []


def test_cite_resolves_every_id_kind_from_state_and_unknown_to_nothing():
    sit = synth()
    d = sit.d
    prop = next(iter(d["proposals"].values()))
    h = next(x for x in d["history"] if x["transition_type"] == "CONFLICT")
    assert cite(sit, prop["new_claim_id"])[0]["document"] == "SrcB.pdf"
    assert cite(sit, prop["proposal_id"])[0]["page"] == 1
    assert cite(sit, h["change_id"])[0]["quote"] == "85 houses damaged"
    assert cite(sit, "clm-nope") == []


# ------------------------------------------------------------------ Situation Brief
@real
def test_brief_on_real_state_is_cited_and_traceable():
    sit = seed_state(ROOT)[0]
    deaths, hyd = _claim_id(sit, "Sindh", "deaths"), _claim_id(sit, "Hyderabad", "houses_damaged")
    llm = Stub(json.dumps({"items": [
        {"section": "key_figures", "text": "Sindh deaths: 28 (as of 2025-08-05).", "citations": [deaths]},
        {"section": "affected_locations", "text": "Hyderabad: 83 houses damaged.", "citations": [hyd]},
        {"section": "unresolved", "text": "No pending changes are recorded.", "citations": [hyd]}]}))
    before = sit.dumps()
    out = situation_brief(llm, sit)
    assert [i["section"] for i in out["items"]] == ["key_figures", "affected_locations", "unresolved"]
    assert {r["page"] for r in out["sources"]} == {3, 4} and {r["document"] for r in out["sources"]} >= {"Sitrep_03-08-2025.pdf"}
    assert sit.dumps() == before  # read-only
    ctx = llm.calls[0][1]["context"]
    assert ctx["counts"] == {"documents": 3, "incidents": 1, "current_claims": 6}


@real
def test_brief_rejects_uncited_invented_and_unknown_section_items():
    sit = seed_state(ROOT)[0]
    cid = _claim_id(sit, "Sindh", "deaths")
    for item in ({"section": "key_figures", "text": "Sindh deaths: 28.", "citations": []},
                 {"section": "key_figures", "text": "Sindh deaths: 28.", "citations": ["clm-deadbeef"]},
                 {"section": "key_figures", "text": "Sindh deaths: 2800.", "citations": [cid]},
                 {"section": "forecast", "text": "Sindh deaths: 28.", "citations": [cid]}):
        with pytest.raises(GroundingError):
            situation_brief(Stub(json.dumps({"items": [item]})), sit)


def test_brief_on_empty_state_never_calls_llm():
    assert situation_brief(Boom(), Situation())["items"] == []


# ------------------------------------------------------------------ Explain Change
def _reply(txt):
    return json.dumps({"explanation": txt})


def test_explain_conflict_is_grounded_read_only_and_engine_stays_authoritative():
    sit = synth()  # SYNTHETIC: 120 (SrcA 09:00) vs 85 (SrcB 11:30) -> CONFLICT
    h = explainable_changes(sit)[0]
    assert h["transition_type"] == "CONFLICT"
    before, hist = sit.dumps(), json.dumps(sit.d["history"])
    llm = Stub(lambda s, p: _reply(
        f"The engine classified this as {p['context']['transition_type']}: {p['context']['previous']['source']} reported "
        f"{p['context']['previous']['value']} and {p['context']['new']['source']} reported {p['context']['new']['value']}. "
        "A human decision is still required."))
    out = explain_change(llm, sit, h["change_id"])
    f = out["facts"]
    assert (f["transition_type"], f["previous"]["value"], f["new"]["value"]) == ("CONFLICT", 120, 85)
    assert f["previous"]["source"] == "SrcA" and f["new"]["source"] == "SrcB"
    assert f["engine_reason"] == h["reason"] and f["human_verification_required"] is True
    assert {(r["document"], r["page"]) for r in out["sources"]} == {("SrcA.pdf", 1), ("SrcB.pdf", 1)}
    assert sit.dumps() == before and json.dumps(sit.d["history"]) == hist and sit.verify_history()
    assert "reclassify" in llm.calls[0][0]  # the prompt forbids changing the classification


def test_explain_update_and_flag_clears_after_human_decision():
    sit = synth(second_time="2000-01-03T09:00", second_val=95)  # SYNTHETIC: >24h later -> UPDATE
    h = explainable_changes(sit)[0]
    assert h["transition_type"] == "UPDATE"
    assert change_facts(sit, h["change_id"])["human_verification_required"] is True
    pid = next(iter(sit.d["proposals"]))
    sit.verify(pid, "accept", reviewer="tester")
    f = change_facts(sit, h["change_id"])
    assert f["human_verification_required"] is False and "VERIFY" in f["decision"] and "tester accepted" in f["decision"]


def test_explain_refuses_non_update_conflict_without_calling_llm():
    sit = synth()
    create = next(x for x in sit.d["history"] if x["transition_type"] == "CREATE")
    with pytest.raises(ValidationError):
        explain_change(Boom(), sit, create["change_id"])
    with pytest.raises(ValidationError):
        explain_change(Boom(), sit, "nope")


def test_explain_rejects_invented_figures():
    sit = synth()
    h = explainable_changes(sit)[0]
    with pytest.raises(GroundingError):
        explain_change(Stub(_reply("SrcA said 120 but SrcB said 850 houses.")), sit, h["change_id"])


@real
def test_real_state_has_nothing_to_explain():
    assert explainable_changes(seed_state(ROOT)[0]) == []  # the supplied PDFs never produce UPDATE/CONFLICT


# ------------------------------------------------------------------ UI (AppTest) with stub provider
def _run_app(tmp_path, monkeypatch, sit, llm):
    f = tmp_path / "state.json"
    f.write_text(sit.dumps())
    monkeypatch.setenv("PASZEL_STATE", str(f))
    for k, v in (("PASZEL_LLM_PROVIDER", "openai"), ("PASZEL_LLM_MODEL", "stub"), ("PASZEL_LLM_API_KEY", "stub")):
        monkeypatch.setenv(k, v)
    monkeypatch.setattr("paszel.ai.get_provider", lambda: llm)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    at.sidebar.radio[0].set_value("Situation overview").run()  # default page is now Home
    return at, f


def _btn(at, label):
    return next(b for b in at.button if b.label == label)


@real
def test_ui_brief_and_ask_on_real_state(tmp_path, monkeypatch):
    sit = seed_state(ROOT)[0]
    deaths = _claim_id(sit, "Sindh", "deaths")
    llm = Stub(lambda s, p: json.dumps({"items": [{"section": "key_figures", "text": "Sindh deaths: 28.", "citations": [deaths]}]})
               if "items" in s else json.dumps({"answer": "Sindh deaths: 28.", "citations": [deaths], "unavailable": False}))
    at, f = _run_app(tmp_path, monkeypatch, sit, llm)
    assert not at.exception
    _btn(at, "Generate Situation Brief").click().run()
    assert not at.exception and any("Sindh deaths: 28." in m.value for m in at.markdown)
    assert any("Sitrep_03-08-2025.pdf" in str(df.value.to_dict()) for df in at.dataframe)
    at.sidebar.radio[0].set_value("Ask PasZel").run()
    at.text_input[0].set_value("What is the death toll?").run()
    _btn(at, "Ask").click().run()
    assert not at.exception and any("Sindh deaths: 28." in m.value for m in at.markdown)
    assert Situation.loads(f.read_text()).dumps() == sit.dumps()  # state untouched


@real
def test_ui_ask_shows_unavailable_message(tmp_path, monkeypatch):
    sit = seed_state(ROOT)[0]
    llm = Stub(json.dumps({"answer": "Nothing about Karachi is stored.", "citations": [], "unavailable": True}))
    at, _ = _run_app(tmp_path, monkeypatch, sit, llm)
    at.sidebar.radio[0].set_value("Ask PasZel").run()
    at.text_input[0].set_value("Free hospital beds in Karachi?").run()
    _btn(at, "Ask").click().run()
    assert not at.exception and any("Information unavailable" in w.value for w in at.warning)


def test_ui_explain_change_on_synthetic_conflict(tmp_path, monkeypatch):
    sit = synth()
    llm = Stub(lambda s, p: _reply(f"Engine says {p['context']['transition_type']}: {p['context']['previous']['value']} vs {p['context']['new']['value']}."))
    at, f = _run_app(tmp_path, monkeypatch, sit, llm)
    assert not at.exception
    _btn(at, "Explain Change").click().run()  # Situation overview -> Recent changes
    assert not at.exception, [e.value for e in at.exception]
    assert any("Engine says CONFLICT: 120 vs 85." in i.value for i in at.info)
    assert any("Human verification still required: **YES**" in c.value for c in at.caption)
    assert Situation.loads(f.read_text()).dumps() == sit.dumps()
    at.sidebar.radio[0].set_value("Verification (1)").run()  # button beside the pending proposal
    assert not at.exception and any(b.label == "Explain Change" for b in at.button)
    at.sidebar.radio[0].set_value("Incident details").run()
    assert not at.exception and any(b.label == "Explain Change" for b in at.button)
