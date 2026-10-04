"""Home page tests. Real PDFs for the seeded state; SYNTHETIC fixture for the UPDATE/CONFLICT counts. No LLM involved."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from paszel.home import PIPELINE_DOT, RULES_DOT, home_stats
from paszel.engine import Situation
from paszel.seed import seed_available, seed_state
from tests.test_genai import synth

ROOT = Path(__file__).parent.parent
real = pytest.mark.skipif(not seed_available(ROOT) or not (ROOT / "benchmark/raw/Sitrep_03-08-2025.pdf").exists(), reason="real PDFs absent")


@real
def test_stats_on_real_state_match_state_exactly():
    sit = seed_state(ROOT)[0]
    s = home_stats(sit)
    assert (s["documents"], s["incidents"], s["current_claims"], s["pending"]) == (3, 1, 6, 0)
    assert s["transitions"] == {"CREATE": 6, "DUPLICATE": 12, "CORROBORATE": 0, "UPDATE": 0, "CONFLICT": 0, "STALE": 0}
    assert s["figures"]["Hyderabad"] == {"houses_damaged": 83} and s["figures"]["Sindh"]["deaths"] == 28
    assert s["history_intact"] and [m["publication_date"] for m in s["documents_list"]] == ["2025-08-03", "2025-08-04", "2025-08-05"]


def test_stats_count_a_pending_conflict_and_do_not_mutate():
    sit = synth()  # SYNTHETIC
    before = sit.dumps()
    s = home_stats(sit)
    assert s["conflicts"] == 1 and s["pending"] == 1 and s["transitions"]["CONFLICT"] == 1
    assert sit.dumps() == before


def test_stats_on_empty_state():
    s = home_stats(Situation())
    assert s["documents"] == 0 and s["figures"] == {} and sum(s["transitions"].values()) == 0


@real
def test_home_is_default_page_and_renders_diagrams_and_charts(tmp_path, monkeypatch):
    monkeypatch.delenv("PASZEL_NO_SEED"); monkeypatch.setenv("PASZEL_STATE", str(tmp_path / "s.json"))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.sidebar.radio[0].value == "Home" and at.title[0].value == "PasZel"
    assert len(at.get("graphviz_chart")) == 2 and "LLM proposes claims" in PIPELINE_DOT and "CONFLICT" in RULES_DOT
    assert any("transcribed" in i.value for i in at.info)
    assert [m.label for m in at.metric][:4] == ["Incidents", "Current figures", "Conflicts", "Awaiting human"]


def test_home_on_empty_state_shows_guidance(tmp_path, monkeypatch):
    monkeypatch.setenv("PASZEL_STATE", str(tmp_path / "s.json"))  # PASZEL_NO_SEED=1 from conftest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not at.exception and any("No data yet" in w.value for w in at.warning)
