"""Rules playground: runs the REAL engine on hypothetical values typed by the user, in a throwaway state."""
from .engine import validate_claim


def add_report(sit, source: str, predicate: str, value: int, when: str) -> dict:
    n = len(sit.d["documents"]) + 1
    doc = {"document_id": f"pg-{n}", "source": source.strip(), "filename": f"playground-report-{n}", "publication_date": when[:10],
           "retrieval_date": when[:10], "sha256": "hypothetical", "source_url": "playground://hypothetical", "file_size": 0}
    label = predicate.replace("_", " ")
    quote = f"Playground area: {int(value)} {label}"
    page = {1: f"Hypothetical playground report. {quote}"}
    c = {"incident_title": "Playground scenario (hypothetical)", "incident_type": "hypothetical", "location": "Playground area",
         "subject": "Playground area", "predicate": predicate, "value": int(value), "page": 1, "quote": quote,
         "heading": "Hypothetical playground report", "as_of": when}
    claim = validate_claim(c, page, None)  # same validation as real claims
    sit.add_document(doc)
    return sit.ingest_claim(claim, doc)
