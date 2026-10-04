"""UI TEST on a SYNTHETIC state (test fixture only; written to a temp file, never to data/)."""
import os
from pathlib import Path

from streamlit.testing.v1 import AppTest

from paszel.engine import Situation, validate_claim

PAGES = {1: "SYNTHETIC. Area X: 120 houses were damaged. 85 houses damaged."}


def _state(tmp_path):
    s = Situation()
    for src, val, when, q in (("SrcA", 120, "2000-01-01T09:00", "120 houses were damaged"),
                              ("SrcB", 85, "2000-01-01T11:30", "85 houses damaged")):
        d = {"document_id": src, "source": src, "filename": src + ".pdf", "publication_date": "2000-01-01",
             "retrieval_date": "2000-01-02", "sha256": "0" * 64, "source_url": "test://", "file_size": 0}
        s.add_document(d)
        c = {"incident_title": "SYNTHETIC", "incident_type": "flood", "location": "Area X", "subject": "Area X",
             "predicate": "houses_damaged", "value": val, "page": 1, "quote": q, "as_of": when, "heading": "SYNTHETIC"}
        s.ingest_claim(validate_claim(c, PAGES, None), d)
    f = tmp_path / "state.json"
    f.write_text(s.dumps())
    return f


def test_all_pages_render_and_verification_accept_works(tmp_path):
    f = _state(tmp_path)
    os.environ["PASZEL_STATE"] = str(f)
    at = AppTest.from_file(str(Path(__file__).parent.parent / "app.py"), default_timeout=30).run()
    assert not at.exception
    for label in ("Incident details", "Verification (1)", "Ingest report", "Situation overview"):
        at.sidebar.radio[0].set_value(label).run()
        assert not at.exception, (label, [e.value for e in at.exception])
    at.sidebar.radio[0].set_value("Verification (1)").run()
    at.button[0].click().run()  # Accept without a reviewer name -> must be refused
    assert [p["status"] for p in Situation.loads(f.read_text()).d["proposals"].values()] == ["pending"]
    at.text_input(key="reviewer").set_value("tester").run()
    at.button[0].click().run()  # Accept
    assert not at.exception
    saved = Situation.loads(f.read_text())
    assert [p["status"] for p in saved.d["proposals"].values()] == ["verified"]
    assert saved.verify_history() and "tester accepted" in saved.d["history"][-1]["reason"]
    assert sorted(c["status"] for c in saved.d["claims"].values()) == ["active", "superseded"]
