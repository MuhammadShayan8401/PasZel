"""Load the three supplied PDMA Sindh SITREPs through the REAL pipeline using developer-transcribed claims.
Not LLM extraction: the claims come from benchmark/processed/seed_claims.json and are validated per document."""
import hashlib
import json
from pathlib import Path

from .ai import ingest_pdf
from .engine import Situation, ValidationError

NOTE = "Developer-transcribed seed claims (NOT LLM-extracted); validated against this PDF's page text."


class SeedProvider:
    def __init__(self, claims): self.claims = claims
    def complete(self, system, user, images=()): return json.dumps({"claims": self.claims})


def seed_available(root: Path) -> bool:
    return all((root / p).exists() for p in ("benchmark/metadata/manifest.json", "benchmark/processed/seed_claims.json"))


def seed_state(root: Path):
    """Returns (Situation, per-document results). Raises if a PDF does not match its manifest hash."""
    manifest = json.loads((root / "benchmark/metadata/manifest.json").read_text())
    claims = json.loads((root / "benchmark/processed/seed_claims.json").read_text())["claims"]
    sit, results = Situation(), []
    for m in sorted(manifest, key=lambda r: r["publication_date"]):
        data = (root / "benchmark/raw" / m["filename"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != m["sha256"]:
            raise ValidationError(f"{m['filename']} does not match its manifest SHA-256")
        results.append(ingest_pdf(sit, SeedProvider(claims), data, m["filename"], m["source"], m["source_url"],
                                  m["publication_date"], retrieval_date="unknown (not provided)", extraction_note=NOTE))
    return sit, results
