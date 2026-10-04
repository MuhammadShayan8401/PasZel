"""Brief / report log / playground tests. Synthetic fixtures only (labelled); playground values are hypothetical."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from paszel.brief import build_brief, report_log
from paszel.engine import Situation, ValidationError
from paszel.playground import add_report


def test_playground_runs_real_rules_on_hypothetical_values():  # SYNTHETIC
    s = Situation()
    assert add_report(s, "Source A", "houses_damaged", 120, "2000-01-01T09:00")["transition"] == "CREATE"
    r = add_report(s, "Source B", "houses_damaged", 85, "2000-01-01T11:30")
    assert r["transition"] == "CONFLICT"
    s.verify(r["proposal_id"], "accept", reviewer="demo")
    assert sorted(c["status"] for c in s.d["claims"].values()) == ["active", "superseded"]
    assert add_report(s, "Source C", "houses_damaged", 85, "2000-01-01T12:00")["transition"] == "CORROBORATE"
    assert add_report(s, "Source A", "houses_damaged", 90, "2000-01-04T09:00")["transition"] == "UPDATE"
    z = Situation(); add_report(z, "A", "deaths", 0, "2000-01-01")  # zero is a valid value
    assert z.d["claims"] and s.verify_history()


def test_report_log_and_brief():  # SYNTHETIC
    s = Situation()
    add_report(s, "Source A", "deaths", 5, "2000-01-01T09:00")
    add_report(s, "Source A", "deaths", 5, "2000-01-02T09:00")
    r = add_report(s, "Source B", "deaths", 7, "2000-01-02T09:30")
    log = report_log(s)
    assert [(x["CREATE"], x["DUPLICATE"], x["CONFLICT"]) for x in log] == [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    md = build_brief(s)
    assert "No LLM was used" in md and "Awaiting human decision" in md and "deaths" in md
    assert "7 (Source B" in md and "5 (Source A" in md and "CONFLICT" in md  # both sides of a dispute are shown
    s.verify(r["proposal_id"], "reject", reviewer="demo")
    assert "Awaiting human decision: none" in build_brief(s)


def test_playground_page_renders_with_banner(tmp_path, monkeypatch):
    monkeypatch.setenv("PASZEL_STATE", str(tmp_path / "s.json"))
    at = AppTest.from_file(str(Path(__file__).parent.parent / "app.py"), default_timeout=30).run()
    at.sidebar.radio[0].set_value("Rules playground (demo)").run()
    assert not at.exception
    assert any("RULES PLAYGROUND" in w.value and "never touches the real situation" in w.value for w in at.warning)
    assert not (tmp_path / "s.json").exists()  # playground never writes the real state file


def test_conflict_is_judged_against_last_confirmation_not_first_report():  # SYNTHETIC (found by the log test above)
    s = Situation()
    add_report(s, "Source A", "deaths", 5, "2000-01-01T09:00")
    add_report(s, "Source A", "deaths", 5, "2000-01-02T09:00")  # confirms at Jan 2
    r = add_report(s, "Source B", "deaths", 7, "2000-01-02T09:30")  # 30 min after last confirmation
    assert r["transition"] == "CONFLICT"
