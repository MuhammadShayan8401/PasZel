# PasZel — Final Hackathon Report

## 1. What PasZel does
PasZel ingests disaster situation reports (PDF), uses an LLM to propose structured claims (e.g. "deaths = 28"), validates each claim against the page text, and maintains an evidence-backed **situation state** over time. New reports are compared with the current state using fixed rules; changes that need judgment wait for a human. Every figure on screen links back to document, page, quote, source, dates and file hash.

## 2. The problem
Reports on one event arrive repeatedly, from different sources, with repeated, revised or disagreeing numbers. A chatbot summarizes each document in isolation. PasZel's principle: *the LLM understands the reports; PasZel maintains the evolving situation.*

## 3. Pipeline
PDF → PyMuPDF page text → LLM proposes claims (page + verbatim quote) → **deterministic validation** → incident/claim matching → transition rule → (UPDATE/CONFLICT only) human Accept / Reject / Modify → state + append-only history → Streamlit dashboard.

Validation rejects a claim unless: it cites a heading that appears on the same page *before* the quote and its subject appears on that page (otherwise rejected as "insufficient evidence context"); predicate is in the fixed list (deaths, injured, missing, houses_damaged, houses_destroyed, people_affected, displaced, livestock_lost); value is a non-negative whole number; the quote exists verbatim on the cited page; the value appears as a whole number in the quote; a timestamp exists. Rejected claims are shown, never applied.

Transition rules (same incident + subject + predicate): no existing claim → CREATE; same value, same source → DUPLICATE; same value, different source → CORROBORATE; different value and later (>24 h, or later from the same source) → UPDATE; different value, different source, within 24 h → CONFLICT; older than current → STALE (history only). UPDATE/CONFLICT create a pending proposal; during a CONFLICT the current claim is shown as *disputed*. "Newest wins" is never applied.

## 4. Components
- `paszel/engine.py` — validation, transition rules, state, verification (no LLM, no third-party deps).
- `paszel/ai.py` — provider interface (Anthropic or OpenAI-compatible via env vars, plain HTTP), report-analysis call with one retry on malformed JSON, PDF extraction, document metadata/SHA-256.
- `app.py` — Streamlit: Situation overview, Incident details, Verification, Ingest report.
- State: one JSON file (`data/state.json`). No database, API server, or Docker. State saves are atomic (temp file + rename).

## 5. What is actually implemented
PDF ingestion with page text; duplicate-file refusal by SHA-256; LLM claim extraction (code written, **never run against a live LLM**); claim validation; incident/claim matching by type + fuzzy location/subject; all six transitions; human verification; append-only history; evidence trail (claim → evidence → document → page → source URL, dates, hash); visible rejected-claim list; reviewer name on every decision; hash-chained history with integrity indicator; CSV export of the current situation with evidence; dashboard with four pages.

## 5b. Added: RAG, analyst agent, multimodal evidence (all constrained, none can change trusted state)
- **Retrieval (RAG):** page text of every ingested report is stored; BM25 keyword search over line-window chunks (`paszel/retrieval.py`). Lexical only, no embeddings, so it will not match synonyms. On the real PDFs a multi-word query can rank an empty "last 24 hours" table above the cumulative table (the right chunk was in the top 3); known limitation.
- **Analyst agent ("Ask PasZel"):** a read-only tool-using loop (max 5 tool calls): `search_evidence`, `get_current_claims`, `get_pending_proposals`, `get_history`. The answer is accepted only if every citation id was actually returned by a tool; uncited, invalid-JSON, unknown-tool and over-limit outputs are rejected, and an explicit "insufficient evidence" answer is allowed. The tool trace is shown. It has no write tool.
- **Multimodal:** the evidence view renders the cited PDF page as an image with the quote highlighted (match reported as full / partial / none). An optional vision-model check sends the page image and claim to the LLM and stores an **advisory** verdict (supported / not_supported / unclear) that never changes a claim. Both providers support image input.
- **Verified here:** quote highlighting returned a full match on the real PDFs for the three table/summary quotes tried; agent rules, retrieval, and provider image payloads are unit-tested with scripted stubs. **Not verified:** any live LLM or vision-model behaviour (agent planning quality, vision accuracy). The vision verdict is a second opinion, not ground truth.

## 5c. Added: report log, situation brief, rules playground
- **Report log + brief (real use):** the Overview shows, per ingested report, how many figures it created, repeated, corroborated, revised, disputed or found stale, and offers a Markdown situation brief generated deterministically from the state (no LLM). With the three supplied PDFs this should show a first report creating figures and later reports repeating them; that expectation is unconfirmed until a live run.
- **Rules playground (demo):** a separate page where the presenter types hypothetical values (source, value, date/time) and the real engine and verification run on a throwaway in-memory state. It carries a permanent warning banner, is never saved, and never touches the real situation. It exists because the supplied real data cannot show UPDATE or CONFLICT. **Say it is hypothetical when you use it; it is not evidence that PasZel detects conflicts in real data.**
- **Logic fix found while testing:** transitions are now judged against the time a claim was last confirmed, not only when first reported (a disagreement 30 minutes after a repeat is a CONFLICT, not an UPDATE). Regression-tested.

## 5d. Homepage shows the three supplied reports (seeded) and empty-reply fix
- On first run (no saved state) the app loads PDMA Sindh SITREP-38/39/40 (3-5 Aug 2025) from `benchmark/raw/` through the real pipeline (hash check against the manifest, PyMuPDF text, validation, state engine, history). The figures are **developer-transcribed from the PDFs (`benchmark/processed/seed_claims.json`), NOT LLM-extracted**; each is re-validated against every document's own page text, and the UI says so. Result: 6 figures (Sindh deaths 28, injured 40, livestock 85, houses 87; Tharparkar livestock 71; Hyderabad houses 83) created from the 3 Aug report and repeated (DUPLICATE) by the 4 and 5 Aug reports; no UPDATE or CONFLICT. Set `PASZEL_NO_SEED=1` to start empty; the Overview has a button to reset to these reports.
- Providers no longer crash with `KeyError: 'content'`: an empty, cut-off or blocked model reply is retried once and then reported as a clear error with the finish reason.

## 6. What was tested, and results
`python -m pytest -q` → **35 passed** (4.2 s). Starting the server (`streamlit run`) → health endpoint `ok`, HTTP 200.
- 10 engine tests on **synthetic fixtures** (labelled in the file): validation failures (bad predicate, quote not on page, value not in quote, missing timestamp, digit-boundary, insufficient heading/subject context), dependent sources not counted as corroboration, hash-chain tamper detection, reviewer required and recorded, CREATE/DUPLICATE/CORROBORATE, CONFLICT needs a human and preserves old claim, UPDATE + reject + double-decision refusal, Modify and STALE.
- 1 **plumbing test on the real PDFs** with a **stub extractor** (4 hand-picked claims, each with a heading copied from the real page text, + 1 deliberately bad quote). It shows: PyMuPDF reads all 7 pages of each file; the bad quote is rejected; first report → 4 CREATE; second and third → 4 DUPLICATE each; no proposals; duplicate file refused. This tests mechanics, **not** LLM quality.
- 6 tests for retrieval, the agent (verified citations, read-only, rejection cases), provider image payloads, page rendering and vision-verdict validation: agent and vision use **scripted stub providers**; retrieval and rendering use the real PDFs.
- 4 tests for the brief, report log, playground rules and the last-confirmation fix (synthetic/hypothetical values) plus a page render check that the playground never writes the real state file.
- 7 tests: the seeded pipeline on the three real PDFs (6 figures each, none rejected, CREATE then DUPLICATE, history intact), first-run homepage seeding and its label, and provider handling of empty/odd replies (stubbed replies).
- 1 UI test on a **synthetic state**: all four pages render; Accept is refused without a reviewer name; with one, Accept on a pending CONFLICT supersedes the old claim, activates the new one, keeps both.

**Not tested:** any live LLM call (no key was available), including both provider HTTP implementations; extraction quality on tables/charts; the browser upload → analyze flow end to end; visual layout in a browser; Streamlit Cloud deployment.

## 7. Known data limitations
- Supplied data: three PDMA Sindh SITREPs (SITREP-38/39/40, dated 3–5 Aug 2025), one source. Text comparison shows pages 2–7 identical after masking dates: deaths 28, injured 40, houses 87, livestock 85, crop 350 acres, 7 breaches. All "last 24 hours" tables are empty.
- Therefore, with these files PasZel can only be expected to CREATE then DUPLICATE. **UPDATE, CONFLICT, CORROBORATE and STALE are verified by synthetic tests only, not demonstrated on real data.**
- The only figures that change (Indus barrage discharge, decimals) are unsupported by design.
- Manifest: retrieval dates are blank (not provided); source URL is the directory as supplied, which returned HTTP 404 to my fetcher, and per-file URLs for these three files are unconfirmed.

## 8. Current limitations
- Table meaning: PyMuPDF text keeps table rows in order but headings are separate. Validation now checks that the cited heading exists on the page before the quote, but it still does **not prove which column a number belongs to**; the heading is supplied by the LLM and only its presence and order are checked. Charts extract as loose, unlabelled numbers and will mostly yield rejected claims.
- Whole numbers only ("1.2 million" is rejected). No OCR. One LLM call per report; no separate retrieval agent (matching is deterministic string similarity).
- "Source" and the optional "derived from" field are typed by the user per upload; PasZel cannot tell whether two sources are truly independent unless the user says so.
- History is append-only and hash-chained, so edits to earlier entries are *detected* (dashboard shows intact/BROKEN) but not prevented; someone who rewrites the whole chain is not detected. Claims are never deleted but their status field changes. Reviewer names are self-typed, not authenticated.
- No persistence guarantees on Streamlit Cloud (disk resets).

## 9. Run locally
`pip install -r requirements.txt`, set the variables below, `streamlit run app.py`. Ingest page: upload a PDF, enter source, URL, publication date (and optionally which source it is derived from), press *Extract and analyze*. Tests: `pip install -r requirements-dev.txt && python -m pytest -q`.

## 10. Environment variables
They can be exported, placed in a local `.env` file (read automatically, never commit it), or set as Streamlit Cloud secrets. Example for a Gemini OpenAI-compatible endpoint: `PASZEL_LLM_PROVIDER=openai`, `PASZEL_LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/`. Provider URL building and `.env` parsing are unit-tested; no call to any live endpoint has been made by me.
`PASZEL_LLM_PROVIDER` (`anthropic` | `openai`), `PASZEL_LLM_MODEL` (required), `PASZEL_LLM_API_KEY` (required), `PASZEL_LLM_BASE_URL` (optional), `PASZEL_STATE` (optional state-file path).

## 11. Deploy on Streamlit Cloud
Push the repo to GitHub → share.streamlit.io → New app → main file `app.py` → Advanced settings → Secrets, paste the four `PASZEL_LLM_*` keys as TOML (`PASZEL_LLM_API_KEY = "..."`). Not deployed or tested by me. State resets on restart, so ingest the reports at the start of the demo.

## 12. Demo walkthrough (3–5 min, requires an LLM key)
1. (30 s) Problem: repeated reports, no memory. Overview page is empty — nothing is pre-loaded.
2. (30 s) The homepage already shows the three supplied reports, processed; say plainly their figures were transcribed, not LLM-extracted. For a live LLM demo, use *Reset* or `PASZEL_NO_SEED=1`, then 2b.
2b. (60 s) Ingest the 3 Aug PDF. Show validated claims and the **rejected** list (invalid claims never reach the state). Open Incident details → a claim → evidence: page, verbatim quote, source, SHA-256.
3. (60 s) Ingest 4 Aug, then 5 Aug. Expect DUPLICATE: PasZel recognizes that cumulative figures did not change and attaches the new evidence instead of creating noise. Say plainly that these reports do not change. (This expectation is unconfirmed until a live LLM run.)
4. (45 s) History table: every change, append-only.
5. (60 s) Verification: queue is empty for these files. Explain the Accept/Reject/Modify workflow and run `pytest -q` to show UPDATE/CONFLICT/verification passing on **synthetic** fixtures, labelled as such.
6. (60 s) Open **Rules playground**: type Source A = 120 at 09:00, Source B = 85 at 11:30. Say clearly these are hypothetical numbers. Show CONFLICT, then Accept/Reject/Modify, then the history. This is the only place the audience sees conflict handling live.
7. (30 s) Limits slide (section 15).

## 13. Pitch
"After a disaster, the same numbers arrive many times from many sources, and they change. PasZel keeps a living, evidence-backed picture of the situation: an LLM reads each report and proposes claims, strict rules check every claim against the page it came from, and a person decides when a number should change. Nothing is overwritten; every figure links to its source page."

## 14. Next steps (not done)
Real multi-day/multi-source data where figures change; column-aware table evidence; time-series claims; OCR; database; auth and reviewer identity.

## 15. What is NOT implemented and must not be claimed
- Detecting conflicts, updates or corroboration **on real data** (only synthetic tests).
- LLM extraction accuracy (never measured), table/chart understanding, OCR.
- Semantic (embedding) retrieval, multi-agent systems, or any claim that the agent's or vision model's quality was measured; there is one read-only tool-using agent and an advisory vision check.
- Integration with PDMA, NDMA or any live feed; real-time updates.
- Confidence scores, independence checks, reliability of any source.
- Production readiness: auth, audit-grade immutability, persistence, scalability.
