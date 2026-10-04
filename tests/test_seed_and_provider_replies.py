"""Seeded homepage (REAL PDFs + developer-transcribed claims) and provider empty-reply handling (stub replies)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from paszel.ai import AnthropicProvider, EmptyLLMReply, OpenAICompatProvider
from paszel.brief import report_log
from paszel.seed import seed_state

ROOT = Path(__file__).parent.parent


def test_seed_runs_real_pipeline_on_three_real_pdfs():
    s, results = seed_state(ROOT)
    assert [len(r["rows"]) for r in results] == [6, 6, 6] and all(not r["rejected"] for r in results)
    log = report_log(s)
    assert [(r["CREATE"], r["DUPLICATE"]) for r in log] == [(6, 0), (0, 6), (0, 6)]
    assert len(s.d["documents"]) == 3 and len(s.d["incidents"]) == 1 and not s.d["proposals"] and s.verify_history()
    assert all(c["status"] == "active" and len(c["evidence_ids"]) == 3 for c in s.d["claims"].values())
    assert all(m["extraction"].startswith("Developer-transcribed") for m in s.d["documents"].values())


def test_homepage_seeds_on_first_run_and_labels_it(tmp_path, monkeypatch):
    monkeypatch.delenv("PASZEL_NO_SEED"); monkeypatch.setenv("PASZEL_STATE", str(tmp_path / "s.json"))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert any("NOT extracted by an LLM" in i.value for i in at.info)
    assert at.metric[0].value == "1" and at.metric[1].value == "6" and at.metric[3].value == "0"  # incidents, claims, pending


@pytest.mark.parametrize("reply", [{"choices": [{"message": {"role": "assistant"}, "finish_reason": "length"}]},
                                   {"choices": [{"message": {"content": None}, "finish_reason": "content_filter"}]},
                                   {"choices": []}, [{"error": "x"}]])
def test_openai_reply_without_content_is_a_clear_error_not_keyerror(monkeypatch, reply):
    monkeypatch.setenv("PASZEL_LLM_API_KEY", "k"); monkeypatch.setenv("PASZEL_LLM_MODEL", "m")
    o = OpenAICompatProvider(); o._post = lambda u, h, b: reply
    with pytest.raises(RuntimeError, match="LLM returned"):
        o.complete("s", "u")


def test_empty_reply_is_retried_once_and_parts_are_joined(monkeypatch):
    monkeypatch.setenv("PASZEL_LLM_API_KEY", "k"); monkeypatch.setenv("PASZEL_LLM_MODEL", "m")
    replies = [{"choices": [{"message": {}}]}, {"choices": [{"message": {"content": [{"type": "text", "text": "ok"}, {"text": "!"}]}}]}]
    o = OpenAICompatProvider(); o._post = lambda u, h, b: replies.pop(0)
    assert o.complete("s", "u") == "ok!"
    a = AnthropicProvider(); a._post = lambda u, h, b: {"stop_reason": "max_tokens"}
    with pytest.raises(EmptyLLMReply, match="stop_reason"): a.complete("s", "u")
