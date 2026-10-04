"""Home page: what PasZel is, how it works, and a live picture of the stored state.

Read-only. Pure data functions (`home_stats`) are separate from rendering so they can be tested without Streamlit.
Nothing here calls an LLM or modifies the situation.
"""
from __future__ import annotations

from collections import Counter

TRANSITIONS = ("CREATE", "DUPLICATE", "CORROBORATE", "UPDATE", "CONFLICT", "STALE")

# Blue = LLM step, green = deterministic code, amber = human.
PIPELINE_DOT = """
digraph G {
  rankdir=LR; bgcolor="transparent"; nodesep=0.35; ranksep=0.45;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#555555", fontcolor="#111111"];
  edge [color="#888888", arrowsize=0.7];
  pdf   [label="Report PDF\\n(page-numbered text)", fillcolor="#EEEEEE"];
  llm   [label="LLM proposes claims\\n(page + verbatim quote)", fillcolor="#BBDEFB"];
  val   [label="Deterministic validation\\nquote on page? value in quote?", fillcolor="#C8E6C9"];
  rej   [label="Rejected\\n(shown, never applied)", fillcolor="#FFCDD2"];
  eng   [label="State engine\\nfixed transition rules", fillcolor="#C8E6C9"];
  auto  [label="CREATE / DUPLICATE /\\nCORROBORATE / STALE\\n(applied automatically)", fillcolor="#C8E6C9"];
  human [label="UPDATE / CONFLICT\\nHuman: accept, reject, modify", fillcolor="#FFE0B2"];
  state [label="Situation state\\n+ evidence + hash-chained history", fillcolor="#C8E6C9"];
  out   [label="Dashboard, Ask PasZel,\\nSituation Brief, Explain Change", fillcolor="#BBDEFB"];
  pdf -> llm -> val; val -> rej [label=" fails", fontsize=9, fontname="Helvetica"];
  val -> eng [label=" passes", fontsize=9, fontname="Helvetica"];
  eng -> auto; eng -> human; auto -> state; human -> state; state -> out;
}
"""

RULES_DOT = """
digraph R {
  rankdir=TB; bgcolor="transparent"; nodesep=0.3; ranksep=0.4;
  node [fontname="Helvetica", fontsize=11, color="#555555", fontcolor="#111111"];
  edge [color="#888888", arrowsize=0.7, fontname="Helvetica", fontsize=9];
  q0 [shape=diamond, style=filled, fillcolor="#EEEEEE", label="Existing figure for this\\nincident + subject + measure?"];
  q1 [shape=diamond, style=filled, fillcolor="#EEEEEE", label="Same value?"];
  q2 [shape=diamond, style=filled, fillcolor="#EEEEEE", label="Same or dependent source?"];
  q3 [shape=diamond, style=filled, fillcolor="#EEEEEE", label="How much later is\\nthe new report?"];
  node [shape=box, style="rounded,filled"];
  create [label="CREATE", fillcolor="#C8E6C9"]; dup [label="DUPLICATE\\n(not independent)", fillcolor="#C8E6C9"];
  cor [label="CORROBORATE\\n(independent source agrees)", fillcolor="#C8E6C9"];
  upd [label="UPDATE\\n(needs a human)", fillcolor="#FFE0B2"]; con [label="CONFLICT\\n(needs a human)", fillcolor="#FFE0B2"];
  stale [label="STALE\\n(history only)", fillcolor="#C8E6C9"];
  q0 -> create [label="no"]; q0 -> q1 [label="yes"];
  q1 -> q2 [label="yes"]; q2 -> dup [label="yes"]; q2 -> cor [label="no"];
  q1 -> q3 [label="no"]; q3 -> upd [label="> 24 h later, or later from\\nthe same/dependent source"];
  q3 -> stale [label="> 24 h earlier"]; q3 -> con [label="otherwise\\n(within 24 h)"];
}
"""


def home_stats(sit) -> dict:
    """Counts for the live panels, computed from stored state only."""
    d = sit.d
    cur = [c for c in d["claims"].values() if c["status"] in ("active", "disputed")]
    tc = Counter(h["transition_type"] for h in d["history"])
    pend = [p for p in d["proposals"].values() if p["status"] == "pending"]
    figures: dict[str, dict[str, int]] = {}
    for c in cur:
        figures.setdefault(c["subject"], {})[c["predicate"]] = c["value"]
    docs = sorted(d["documents"].values(), key=lambda m: m["publication_date"])
    return {"documents": len(d["documents"]), "incidents": len(d["incidents"]), "current_claims": len(cur),
            "evidence": len(d["evidence"]), "pending": len(pend),
            "conflicts": sum(p["transition_type"] == "CONFLICT" for p in pend),
            "transitions": {t: tc.get(t, 0) for t in TRANSITIONS},
            "figures": figures, "documents_list": docs,
            "history_intact": sit.verify_history()}


def render_home(st, pd, sit, seeded_note: bool) -> None:
    s = home_stats(sit)
    st.title("PasZel")
    st.markdown("##### An evidence-backed, evolving picture of a disaster, not just a summary of each report.")
    st.write("Reports on one event arrive repeatedly, from different sources, with repeated, revised or disagreeing numbers. "
             "A chatbot summarises each document in isolation. PasZel keeps a **situation state**: every figure is tied to a document, "
             "page and verbatim quote, and changes are classified by fixed rules, with a person deciding anything ambiguous.")
    st.info("**The LLM understands the reports. PasZel's code maintains the situation. A human decides UPDATE and CONFLICT.**")
    if seeded_note:
        st.info("Showing the three supplied PDMA Sindh SITREPs (3-5 Aug 2025). Their figures were transcribed by the developer from the PDFs, "
                "NOT extracted by an LLM, then run through the same validation and state rules. Use 'Ingest report' for LLM extraction.")

    st.subheader("Current situation at a glance")
    a, b, c, d, e, f = st.columns(6)
    a.metric("Incidents", s["incidents"]); b.metric("Current figures", s["current_claims"])
    c.metric("Conflicts", s["conflicts"]); d.metric("Awaiting human", s["pending"])
    e.metric("Reports", s["documents"]); f.metric("Evidence items", s["evidence"])
    if not s["incidents"]:
        st.warning("No data yet. Open 'Ingest report' to add a real PDF, or reset to the supplied reports on the overview page.")
    else:
        left, right = st.columns(2)
        with left:
            st.markdown("**Current figures by area**")
            df = pd.DataFrame(s["figures"]).T.fillna(0).astype(int)
            st.bar_chart(df, horizontal=True)
            st.caption("Each bar is a stored, validated figure (latest accepted value). Different measures share one axis, so compare within a measure.")
        with right:
            st.markdown("**What the reports did to the state**")
            tdf = pd.DataFrame({"count": s["transitions"]})
            st.bar_chart(tdf)
            st.caption("DUPLICATE = a report repeated a known figure. UPDATE and CONFLICT appear only when numbers actually change or disagree.")
        st.markdown("**Reports ingested**")
        st.dataframe(pd.DataFrame([{"published": m["publication_date"], "report": m["filename"], "source": m["source"],
                                    "extraction": m.get("extraction", "")[:60]} for m in s["documents_list"]]),
                     hide_index=True, use_container_width=True)
        st.caption("History integrity (hash chain): " + ("intact" if s["history_intact"] else "BROKEN - history was edited"))

    st.subheader("How it works")
    st.graphviz_chart(PIPELINE_DOT, use_container_width=True)
    st.caption("Blue = LLM, green = deterministic code, amber = human, red = rejected. The LLM never writes to the state.")

    st.subheader("How PasZel decides what a new figure means")
    l2, r2 = st.columns([3, 2])
    with l2:
        st.graphviz_chart(RULES_DOT, use_container_width=True)
    with r2:
        st.markdown("- **\"Newest wins\" is never applied.**\n"
                    "- Two sources that disagree within 24 hours are a **CONFLICT**, not an update.\n"
                    "- A source that repeats another source (declared as *derived from*) is **not** independent corroboration.\n"
                    "- A figure is rejected unless its quote exists verbatim on the cited page and contains the value.\n"
                    "- Every decision is logged in an append-only, hash-chained history with the reviewer's name.")

    st.subheader("What you can do")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Ask PasZel**\n\nQuestions answered only from stored state and evidence, with source and page. Unknown means \"unavailable\".")
    c2.markdown("**Situation Brief**\n\nA short operator summary of the current state, every line cited. Button on *Situation overview*.")
    c3.markdown("**Explain Change**\n\nPlain-language explanation of an UPDATE or CONFLICT the engine already classified. It cannot change the decision.")
    st.caption("Use the sidebar to open each page. Limits: whole numbers only, no OCR, one LLM call per report, "
               "and the supplied real reports never produce UPDATE or CONFLICT (those are demonstrated in the Rules playground, with hypothetical values).")
