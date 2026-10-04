# PasZel (hackathon prototype)

PasZel keeps an evidence-backed, evolving disaster situation instead of just summarizing reports.
**The LLM proposes claims; deterministic code validates them; a human decides UPDATE/CONFLICT.**

Full report (pipeline, tests, limitations, demo script, pitch, what NOT to claim): `docs/FINAL_REPORT.md`

## Run locally
    pip install -r requirements.txt
    export PASZEL_LLM_PROVIDER=anthropic PASZEL_LLM_MODEL=<model> PASZEL_LLM_API_KEY=<key>   # see .env.example
    streamlit run app.py
## Test
    pip install -r requirements-dev.txt && python -m pytest -q
## Data
`benchmark/raw/` holds three real PDMA Sindh SITREPs (3, 4, 5 Aug 2025) with SHA-256 in `benchmark/metadata/manifest.json`.
Their cumulative figures are identical across the three days, so they do NOT exercise UPDATE/CONFLICT (see report).

## GenAI features (grounded, read-only) - `paszel/genai.py`
Ask PasZel, Generate Situation Brief (Overview page) and Explain Change (UPDATE/CONFLICT entries). Flow: code retrieves state/evidence -> LLM words it ->
code rejects replies that cite unknown ids or contain numbers absent from the state -> source/page rows are resolved from stored evidence.
The LLM never writes to state or decides a transition. Tests use scripted stubs (`tests/test_genai.py`); live-model behaviour is untested.
