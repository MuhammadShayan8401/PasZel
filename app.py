import json
import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from paszel.ai import get_provider, ingest_pdf, load_dotenv_file
from paszel.vision import render_page, vision_check
from paszel.brief import build_brief, report_log
from paszel.engine import PREDICATES, Situation, ValidationError
from paszel.home import render_home
from paszel.genai import SECTIONS, ask_paszel, explain_change, explainable_changes, situation_brief
from paszel.playground import add_report
from paszel.seed import seed_available, seed_state

load_dotenv_file(str(Path(__file__).parent / ".env"))
load_dotenv_file()
try:  # Streamlit Cloud secrets; absent when running locally with env vars
    for k in ("PASZEL_LLM_PROVIDER", "PASZEL_LLM_MODEL", "PASZEL_LLM_API_KEY", "PASZEL_LLM_BASE_URL"):
        if k not in os.environ and k in st.secrets:
            os.environ[k] = st.secrets[k]
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).parent
STATE = Path(os.environ.get("PASZEL_STATE", "data/state.json"))
RAW = Path("benchmark/raw")
st.set_page_config(page_title="PasZel", layout="wide")

if "sit" not in st.session_state:
    st.session_state.sit = Situation.loads(STATE.read_text()) if STATE.exists() else Situation()
sit: Situation = st.session_state.sit


def save():
    STATE.parent.mkdir(exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(sit.dumps())
    tmp.replace(STATE)  # atomic


if not STATE.exists() and not sit.d["documents"] and os.environ.get("PASZEL_NO_SEED") != "1" and seed_available(ROOT):
    sit.d = seed_state(ROOT)[0].d  # first run: show the three supplied reports, processed
    save()
D = sit.d


def ev_rows(ids):
    return [{"source": D["evidence"][e]["source"], "document": D["documents"][D["evidence"][e]["document_id"]]["filename"],
             "page": D["evidence"][e]["page"], "heading": D["evidence"][e].get("heading", ""), "as of": D["evidence"][e]["timestamp"],
             "text": D["evidence"][e]["text"]} for e in ids]


def claim_row(c):
    return {"subject": c["subject"], "predicate": c["predicate"], "value": c["value"], "source": c["source"],
            "reported": c["reported_time"], "status": c["status"], "evidence": len(c["evidence_ids"]), "id": c["claim_id"]}


def show_sources(rows):
    """Evidence resolved from stored state (never from LLM text)."""
    if rows:
        st.caption("Sources / evidence (from PasZel state):")
        st.dataframe(pd.DataFrame(rows)[["source", "document", "page", "heading", "quote", "as_of"]], hide_index=True, use_container_width=True)


def explain_ui(change_id, key):
    """'Explain Change' button + result for one engine-classified UPDATE/CONFLICT. Read-only: never touches state."""
    store = st.session_state.setdefault("explained", {})
    if st.button("Explain Change", key=f"expl_{key}_{change_id}", disabled=not cfg_ok, help=None if cfg_ok else "Configure the LLM first"):
        try:
            with st.spinner("Explaining (the engine's decision is not changed)..."):
                store[change_id] = explain_change(get_provider(), sit, change_id)
        except (ValidationError, RuntimeError, OSError) as err:
            store.pop(change_id, None)
            st.error(f"Explanation failed or was rejected: {err}")
    out = store.get(change_id)
    if out:
        f = out["facts"]
        st.info(out["explanation"])
        st.caption(f"Engine classification (authoritative, deterministic): **{f['transition_type']}** · {f['engine_reason']} · "
                   f"Human verification still required: **{'YES' if f['human_verification_required'] else 'NO'}** ({f['decision']})")
        st.dataframe(pd.DataFrame([{"": "previous", "value": f["previous"]["value"], "source": f["previous"]["source"], "reported": f["previous"]["reported_time"]},
                                   {"": "new", "value": f["new"]["value"], "source": f["new"]["source"], "reported": f["new"]["reported_time"]}]),
                     hide_index=True)
        show_sources(out["sources"])


pending = [p for p in D["proposals"].values() if p["status"] == "pending"]
page = st.sidebar.radio("PasZel", ["Home", "Situation overview", "Incident details",
                                   f"Verification ({len(pending)})", "Ingest report", "Ask PasZel", "Rules playground (demo)"])
ENV_PATH = Path(__file__).parent / ".env"
cfg_ok = all(os.environ.get(k) for k in ("PASZEL_LLM_PROVIDER", "PASZEL_LLM_MODEL", "PASZEL_LLM_API_KEY"))
st.sidebar.caption(f"LLM: {os.environ['PASZEL_LLM_PROVIDER']} / {os.environ['PASZEL_LLM_MODEL']}" if cfg_ok
                   else "LLM: NOT CONFIGURED (set PASZEL_LLM_* or .env)")
if not cfg_ok:
    with st.sidebar.expander("LLM settings (this session only, local use)", expanded=True):
        st.caption(f"Looked for .env at: {ENV_PATH} -> {'found but incomplete' if ENV_PATH.exists() else 'NOT FOUND'}")
        if (ENV_PATH.parent / ".env.txt").exists():
            st.warning("Found '.env.txt'. Rename it to exactly '.env'.")
        prov = st.selectbox("Provider", ["openai", "anthropic"])
        model_in = st.text_input("Model")
        key_in = st.text_input("API key", type="password")
        base_in = st.text_input("Base URL (optional)")
        st.caption("Held in this process only, never saved. Local use only: on a shared hosted app use Secrets.")
        if st.button("Use these settings") and model_in and key_in:
            os.environ.update(PASZEL_LLM_PROVIDER=prov, PASZEL_LLM_MODEL=model_in.strip(), PASZEL_LLM_API_KEY=key_in.strip())
            if base_in.strip():
                os.environ["PASZEL_LLM_BASE_URL"] = base_in.strip()
            st.rerun()
st.sidebar.caption("The LLM proposes. Validation is deterministic. A human decides UPDATE/CONFLICT.")

if page == "Home":
    render_home(st, pd, sit, any(str(m.get("extraction", "")).startswith("Developer-transcribed") for m in D["documents"].values()))

elif page == "Ingest report":
    st.header("Ingest a real report")
    f = st.file_uploader("Disaster report (PDF)", type="pdf")
    c1, c2, c3 = st.columns(3)
    source = c1.text_input("Source organisation (e.g. NDMA)")
    url = c2.text_input("Source URL")
    pub = c3.date_input("Publication date", value=None)
    derived = st.text_input("Derived from (optional): if this source repeats another source's figures, name it")
    missing = [n for n, v in (("PDF file", f), ("source organisation", source.strip()), ("source URL", url.strip()),
                              ("publication date", pub)) if not v]
    if missing:
        st.info("Still needed before you can analyze: " + ", ".join(missing) + ". (Press Enter after typing in a box; dates look like 2025/08/03.)")
    if st.button("Extract and analyze", type="primary", disabled=bool(missing)):
        data = f.getvalue()
        try:
            with st.spinner("Extracting text and asking the LLM (can take up to a minute)..."):
                out = ingest_pdf(sit, get_provider(), data, f.name, source.strip(), url.strip(), pub.isoformat(), derived)
        except (ValidationError, RuntimeError, OSError) as e:
            st.error(f"Analysis failed: {e}")
            st.stop()
        RAW.mkdir(parents=True, exist_ok=True)
        (RAW / f.name).write_bytes(data)
        save()
        for e in out["errors"]:
            st.warning(e)
        st.success(f"{len(out['rows'])} claims validated and processed, {len(out['rejected'])} rejected by validation.")
        if out["rows"]:
            st.dataframe(pd.DataFrame(out["rows"]), hide_index=True, use_container_width=True)
        if out["rejected"]:
            with st.expander("Rejected LLM claims (kept visible, never applied)", expanded=True):
                st.json([{"claim": c, "reason": r} for c, r in out["rejected"]])

elif page == "Situation overview":
    st.header("Situation overview")
    if any(str(m.get("extraction", "")).startswith("Developer-transcribed") for m in D["documents"].values()):
        st.info("Showing the three supplied PDMA Sindh SITREPs (3-5 Aug 2025). Their figures were transcribed by the developer from the PDFs, "
                "NOT extracted by an LLM, then run through the same validation and state rules. Use 'Ingest report' for LLM extraction.")
    if seed_available(ROOT) and st.button("Reset to the three supplied reports (discards the current state)"):
        sit.d = seed_state(ROOT)[0].d
        save()
        st.rerun()
    conflicts = [p for p in pending if p["transition_type"] == "CONFLICT"]
    a, b, c, d = st.columns(4)
    a.metric("Incidents", len(D["incidents"]))
    b.metric("Current claims", sum(x["status"] in ("active", "disputed") for x in D["claims"].values()))
    c.metric("Conflicts", len(conflicts))
    d.metric("Pending verification", len(pending))
    if not D["incidents"]:
        st.info("No data yet. Use 'Ingest report' with real PDFs (see README for what's needed).")
    if D["incidents"]:
        if st.button("Generate Situation Brief", type="primary", disabled=not cfg_ok, help=None if cfg_ok else "Configure the LLM first"):
            try:
                with st.spinner("Writing the brief from stored state only..."):
                    st.session_state["brief_out"] = {"fp": D["history"][-1]["hash"] if D["history"] else "", **situation_brief(get_provider(), sit)}
            except (ValidationError, RuntimeError, OSError) as err:
                st.session_state.pop("brief_out", None)
                st.error(f"Brief failed or was rejected: {err}")
        bo = st.session_state.get("brief_out")
        if bo and bo["fp"] == (D["history"][-1]["hash"] if D["history"] else ""):
            st.subheader("Situation brief (LLM wording of stored state; every line cites evidence)")
            if bo["message"]:
                st.warning(bo["message"])
            for key, title in SECTIONS.items():
                its = [i for i in bo["items"] if i["section"] == key]
                st.markdown(f"**{title}**")
                if not its:
                    st.caption("Nothing stated in PasZel state for this section.")
                for i in its:
                    st.write(f"- {i['text']}")
            show_sources(bo["sources"])
        elif bo:
            st.caption("The earlier brief is hidden because the situation state has changed since; generate a new one.")
    for inc in D["incidents"].values():
        st.subheader(f"{inc['title']}  ·  {inc['location']}")
        cur = [claim_row(x) for x in D["claims"].values()
               if x["incident_id"] == inc["incident_id"] and x["status"] in ("active", "disputed")]
        st.dataframe(pd.DataFrame(cur), hide_index=True, use_container_width=True)
    st.caption("History integrity (hash chain): " + ("intact" if sit.verify_history() else "BROKEN - history was edited"))
    rows = []
    for x in D["claims"].values():
        if x["status"] in ("active", "disputed"):
            e = D["evidence"][x["evidence_ids"][0]]; m = D["documents"][e["document_id"]]
            rows.append({"incident": D["incidents"][x["incident_id"]]["title"], "subject": x["subject"], "predicate": x["predicate"],
                         "value": x["value"], "status": x["status"], "source": x["source"], "reported": x["reported_time"],
                         "document": m["filename"], "page": e["page"], "heading": e["heading"], "quote": e["text"],
                         "sha256": m["sha256"], "source_url": m["source_url"]})
    if rows:
        st.download_button("Download current situation (CSV)", pd.DataFrame(rows).to_csv(index=False),
                           "paszel_current_situation.csv", "text/csv")
    if D["incidents"]:
        st.download_button("Download situation brief (Markdown, no LLM)", build_brief(sit), "paszel_situation_brief.md", "text/markdown")
        st.subheader("What each report did")
        st.caption("Per ingested report: how many figures it created, repeated, corroborated, revised, disputed or found out of date. Zeros everywhere except DUPLICATE mean the report changed nothing.")
        st.dataframe(pd.DataFrame(report_log(sit)), hide_index=True, use_container_width=True)
    st.subheader("Recent changes")
    for h in reversed(D["history"][-8:]):
        st.write(f"`{h['timestamp']}` **{h['transition_type']}** {'' if h['applied'] else '(awaiting human)'} — {h['reason']}")
        if h["transition_type"] in ("UPDATE", "CONFLICT"):
            explain_ui(h["change_id"], "recent")

elif page == "Incident details":
    if not D["incidents"]:
        st.info("No incidents yet.")
        st.stop()
    iid = st.selectbox("Incident", list(D["incidents"]), format_func=lambda i: D["incidents"][i]["title"])
    inc = D["incidents"][iid]
    st.header(inc["title"])
    st.caption(f"{inc['type']} · {inc['location']} · created {inc['created_at']}")
    claims = [x for x in D["claims"].values() if x["incident_id"] == iid]
    st.subheader("Current state")
    st.dataframe(pd.DataFrame([claim_row(x) for x in claims if x["status"] in ("active", "disputed")]),
                 hide_index=True, use_container_width=True)
    st.subheader("All claims, including superseded / rejected / stale")
    for x in sorted(claims, key=lambda x: (x["predicate"], x["reported_time"])):
        with st.expander(f"{x['predicate']} = {x['value']} · {x['source']} · {x['reported_time']} · {x['status']}"):
            st.dataframe(pd.DataFrame(ev_rows(x["evidence_ids"])), hide_index=True, use_container_width=True)
            for e in x["evidence_ids"]:
                m = D["documents"][D["evidence"][e]["document_id"]]
                st.caption(f"{m['filename']} · published {m['publication_date']} · retrieved {m['retrieval_date']} · "
                           f"sha256 {m['sha256'][:16]}… · {m['source_url']}")
        ev = D["evidence"][e]
        if st.checkbox("Show page image with quote highlighted", key=f"img{e}"):
            pdf = RAW / m["filename"]
            if not pdf.exists():
                st.warning("Original PDF not found in benchmark/raw; cannot render.")
            else:
                png, match = render_page(pdf.read_bytes(), ev["page"], ev["text"])
                st.image(png, caption={"full": "Quote highlighted.", "partial": "Only the start of the quote could be located (approximate highlight).",
                                       "none": "Quote could not be located on the page image; page shown without highlight."}[match])
                if "vision_check" in ev:
                    st.info(f"Vision model second opinion (advisory, does not change the claim): {ev['vision_check']['verdict']} - {ev['vision_check']['reason']}")
                if cfg_ok and st.button("Ask a vision model to check this page (advisory)", key=f"vc{e}"):
                    try:
                        with st.spinner("Asking the vision model..."):
                            ev["vision_check"] = vision_check(get_provider(), png, f"Subject: {x['subject']}; measure: {x['predicate']}; value: {x['value']}; heading: {ev.get('heading', '')}; quote: {ev['text']}")
                    except (ValidationError, RuntimeError, OSError) as err:
                        st.error(f"Vision check failed: {err}")
                    else:
                        save()
                        st.rerun()
    st.subheader("History")
    st.dataframe(pd.DataFrame([h for h in D["history"] if h["incident_id"] == iid]).drop(columns=["incident_id"]),
                 hide_index=True, use_container_width=True)
    chg = explainable_changes(sit, iid)
    if chg:
        st.subheader("Explain a change (UPDATE / CONFLICT)")
        cid = st.selectbox("Change", [h["change_id"] for h in chg], format_func=lambda c: next(f"{h['transition_type']} · {h['reason']}" for h in chg if h["change_id"] == c))
        explain_ui(cid, "incident")

elif page == "Rules playground (demo)":
    st.header("Rules playground")
    st.warning("RULES PLAYGROUND: hypothetical values that YOU type. Nothing here is real data, nothing is saved, and it never touches the real situation. "
               "It runs the same deterministic rules as the real pipeline, so you can show how PasZel treats repeated, revised and disagreeing reports.")
    pg = st.session_state.setdefault("pg", Situation())
    measure = st.selectbox("Measure", sorted(PREDICATES), index=sorted(PREDICATES).index("houses_damaged"))
    with st.form("pgform"):
        a, b, c = st.columns(3)
        src, val, day = a.text_input("Source name (e.g. Source A)"), b.number_input("Value", min_value=0, step=1), c.date_input("Report date")
        tm = st.time_input("Report time")
        go = st.form_submit_button("Add hypothetical report")
    if go and src.strip():
        r = add_report(pg, src, measure, int(val), datetime.combine(day, tm).isoformat(timespec="minutes"))
        st.session_state["pg_last"] = f"{r['transition']}: {r['reason']}"
    if st.session_state.get("pg_last"):
        st.success(st.session_state["pg_last"])
    for p in [p for p in pg.d["proposals"].values() if p["status"] == "pending"]:
        old, new = pg.d["claims"][p["target_claim_id"]], pg.d["claims"][p["new_claim_id"]]
        st.markdown(f"**Pending {p['transition_type']}:** current {old['value']} ({old['source']}) vs proposed {new['value']} ({new['source']})")
        mv = st.number_input("Value if modifying", min_value=0, value=int(new["value"]), key=f"pgv{p['proposal_id']}")
        b1, b2, b3 = st.columns(3)
        for col, dec in ((b1, "accept"), (b2, "reject"), (b3, "modify")):
            if col.button(dec.capitalize(), key=f"pg{dec}{p['proposal_id']}"):
                pg.verify(p["proposal_id"], dec, value=mv if dec == "modify" else None, reviewer="playground demo")
                st.rerun()
    cur = [claim_row(x) for x in pg.d["claims"].values()]
    if cur:
        st.subheader("Claims (all statuses)")
        st.dataframe(pd.DataFrame(cur).drop(columns=["id"]), hide_index=True, use_container_width=True)
        st.subheader("History")
        st.dataframe(pd.DataFrame(pg.d["history"])[["transition_type", "reason", "applied"]], hide_index=True, use_container_width=True)
    if st.button("Reset playground"):
        st.session_state["pg"], st.session_state["pg_last"] = Situation(), ""
        st.rerun()
elif page == "Ask PasZel":
    st.header("Ask PasZel")
    st.caption("Answers ONLY from PasZel's stored situation state, evidence and history. Retrieval is done by code; the LLM only words the answer, "
               "and its citations and numbers are checked. It cannot change the situation.")
    if not cfg_ok:
        st.info("Configure the LLM in the sidebar first.")
        st.stop()
    q = st.text_input("Question", placeholder="e.g. What is the current situation? What happened in Hyderabad? What information is uncertain?")
    if st.button("Ask", disabled=not q.strip()):
        try:
            with st.spinner("Retrieving evidence and asking the LLM..."):
                out = ask_paszel(get_provider(), sit, q.strip())
        except (ValidationError, RuntimeError, OSError) as err:
            st.error(f"Answer rejected or failed (nothing is shown as an answer): {err}")
            st.stop()
        if out["unavailable"]:
            st.warning(f"Information unavailable. {out['answer']}")
        else:
            st.write(out["answer"])
            show_sources(out["sources"])
else:
    st.header("Human verification")
    reviewer = st.text_input("Reviewer name (recorded in history)", key="reviewer")
    if not pending:
        st.info("Nothing pending.")
    for p in pending:
        old, new = D["claims"][p["target_claim_id"]], D["claims"][p["new_claim_id"]]
        st.subheader(f"{p['transition_type']} · {D['incidents'][p['incident_id']]['title']}")
        st.write(p["reason"])
        l, r = st.columns(2)
        l.markdown(f"**Current** — {old['source']} ({old['reported_time']})\n\n### {old['predicate']}: {old['value']}")
        l.dataframe(pd.DataFrame(ev_rows(old["evidence_ids"])), hide_index=True)
        r.markdown(f"**Proposed** — {new['source']} ({new['reported_time']})\n\n### {new['predicate']}: {new['value']}")
        r.dataframe(pd.DataFrame(ev_rows(new["evidence_ids"])), hide_index=True)
        note = st.text_input("Reviewer note", key=f"n{p['proposal_id']}")
        mv = st.number_input("Value if modifying", min_value=0, value=int(new["value"]), key=f"v{p['proposal_id']}")
        b1, b2, b3 = st.columns(3)
        for col, dec in ((b1, "accept"), (b2, "reject"), (b3, "modify")):
            if col.button(dec.capitalize(), key=f"{dec}{p['proposal_id']}"):
                if not reviewer.strip():
                    st.error("Enter a reviewer name first.")
                    st.stop()
                sit.verify(p["proposal_id"], dec, value=mv if dec == "modify" else None, note=note, reviewer=reviewer)
                save()
                st.rerun()
        _h = next((h for h in D["history"] if h["transition_type"] in ("UPDATE", "CONFLICT") and h["new_claim"] == p["new_claim_id"]), None)
        if _h:
            explain_ui(_h["change_id"], "verify")
        st.divider()
