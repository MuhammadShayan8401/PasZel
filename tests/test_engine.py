"""TEST FIXTURES ONLY. All values below are SYNTHETIC and exist solely to exercise
the transition rules. They are never loaded by the app and are not disaster data."""
import pytest
from paszel.engine import Situation, ValidationError, classify, validate_claim

PAGES = {1: "SYNTHETIC TEST TEXT. In Area X, 120 houses were damaged. Later 85 houses damaged."}


def claim(value, as_of, quote="120 houses were damaged"):
    return {"incident_title": "T", "incident_type": "flood", "location": "Area X", "subject": "Area X",
            "predicate": "houses_damaged", "value": value, "page": 1, "quote": quote, "as_of": as_of,
            "heading": "SYNTHETIC TEST TEXT"}


def doc(name):
    return {"document_id": name, "source": name, "filename": name + ".pdf", "publication_date": "2000-01-01",
            "retrieval_date": "2000-01-02", "sha256": "0" * 64, "source_url": "test://", "file_size": 0}


def run(sit, value, when, src, quote="120 houses were damaged"):
    d = doc(src)
    sit.add_document(d)
    return sit.ingest_claim(validate_claim(claim(value, when, quote), PAGES, None), d)


def test_validation_rejects_bad_claims():
    with pytest.raises(ValidationError): validate_claim({**claim(120, "2000-01-01"), "predicate": "vibes"}, PAGES, None)
    with pytest.raises(ValidationError): validate_claim(claim(120, "2000-01-01", "not on page"), PAGES, None)
    with pytest.raises(ValidationError): validate_claim(claim(999, "2000-01-01"), PAGES, None)  # value not in quote
    with pytest.raises(ValidationError): validate_claim(claim(120, None), PAGES, None)  # no timestamp


def test_create_duplicate_corroborate():
    s = Situation()
    assert run(s, 120, "2000-01-01", "A")["transition"] == "CREATE"
    assert run(s, 120, "2000-01-01", "A")["transition"] == "DUPLICATE"
    assert run(s, 120, "2000-01-01", "B")["transition"] == "CORROBORATE"
    assert len(s.d["claims"]) == 1 and len(next(iter(s.d["claims"].values()))["evidence_ids"]) == 3


def test_conflict_needs_human_and_preserves_history():
    s = Situation()
    run(s, 120, "2000-01-01T09:00", "A")
    r = run(s, 85, "2000-01-01T11:30", "B", "85 houses damaged")
    assert r["transition"] == "CONFLICT"
    old = r["target"]
    assert old["status"] == "disputed" and old["value"] == 120  # not auto-replaced
    s.verify(r["proposal_id"], "accept", reviewer="tester")
    assert old["status"] == "superseded" and r["claim"]["status"] == "active"
    assert len(s.d["claims"]) == 2  # previous claim preserved


def test_update_later_same_source_and_reject():
    s = Situation()
    run(s, 120, "2000-01-01", "A")
    r = run(s, 85, "2000-01-03", "A", "85 houses damaged")
    assert r["transition"] == "UPDATE"
    s.verify(r["proposal_id"], "reject", reviewer="tester")
    assert r["target"]["status"] == "active" and r["claim"]["status"] == "rejected"
    with pytest.raises(ValidationError): s.verify(r["proposal_id"], "accept", reviewer="tester")  # already decided


def test_modify_and_stale():
    s = Situation()
    run(s, 120, "2000-01-05", "A")
    r = run(s, 85, "2000-01-08", "B", "85 houses damaged")
    s.verify(r["proposal_id"], "modify", value=90, reviewer="tester")
    assert r["claim"]["value"] == 90 and r["claim"]["original_proposed_value"] == 85
    assert classify(r["claim"], {**r["claim"], "value": 1, "reported_time": "2000-01-01", "source": "Z"})[0] == "STALE"


def test_value_must_match_whole_number_in_quote():  # SYNTHETIC
    p = {1: "SYNTHETIC: 28 deaths, 83 houses, code 03"}
    base = {"incident_title": "T", "incident_type": "flood", "location": "L", "subject": "S",
            "predicate": "deaths", "page": 1, "as_of": "2000-01-01", "heading": "SYNTHETIC"}
    with pytest.raises(ValidationError): validate_claim({**base, "value": 8, "quote": "28 deaths"}, p, None)
    with pytest.raises(ValidationError): validate_claim({**base, "value": 3, "quote": "83 houses"}, p, None)
    assert validate_claim({**base, "value": 28, "quote": "28 deaths"}, p, None)["value"] == 28
    assert validate_claim({**base, "value": 3, "quote": "code 03"}, p, None)["value"] == 3


def test_insufficient_context_is_rejected():  # SYNTHETIC
    for bad, why in (({"heading": ""}, "heading not found"), ({"heading": "not on page"}, "heading not found"),
                     ({"heading": "Later 85 houses"}, "does not precede"), ({"subject": "Elsewhere"}, "subject not found")):
        with pytest.raises(ValidationError, match=why):
            validate_claim({**claim(120, "2000-01-01"), **bad}, PAGES, None)


def test_derived_source_is_not_independent_corroboration():  # SYNTHETIC
    s = Situation()
    run(s, 120, "2000-01-01", "A")
    d = doc("B"); d["derived_from"] = "A"; s.add_document(d)
    r = s.ingest_claim(validate_claim(claim(120, "2000-01-01"), PAGES, None), d)
    assert r["transition"] == "DUPLICATE" and "not independent" in r["reason"]


def test_history_hash_chain_detects_tampering():  # SYNTHETIC
    s = Situation()
    run(s, 120, "2000-01-01", "A"); run(s, 120, "2000-01-01", "B")
    assert s.verify_history()
    s.d["history"][0]["reason"] = "edited"
    assert not s.verify_history()


def test_reviewer_required_and_recorded():  # SYNTHETIC
    s = Situation()
    run(s, 120, "2000-01-05", "A")
    r = run(s, 85, "2000-01-08", "B", "85 houses damaged")
    with pytest.raises(ValidationError): s.verify(r["proposal_id"], "accept", reviewer=" ")
    s.verify(r["proposal_id"], "modify", value=90, reviewer="Ayesha")
    assert s.d["proposals"][r["proposal_id"]]["reviewer"] == "Ayesha"
    assert "Ayesha modified" in s.d["history"][-1]["reason"] and s.verify_history()
