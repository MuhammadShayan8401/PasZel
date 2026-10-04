"""Deterministic situation brief and per-report change log. No LLM involved."""
from datetime import datetime, timezone

TRANS = ("CREATE", "DUPLICATE", "CORROBORATE", "UPDATE", "CONFLICT", "STALE")


def report_log(sit) -> list[dict]:
    """What each ingested report did to the state, oldest first."""
    ev_doc = {e["evidence_id"]: e["document_id"] for e in sit.d["evidence"].values()}
    rows = {i: {"report": m["filename"], "source": m["source"], "published": m["publication_date"], **{t: 0 for t in TRANS}}
            for i, m in sit.d["documents"].items()}
    for h in sit.d["history"]:
        if h["transition_type"] in TRANS and h["evidence_ids"]:
            did = ev_doc.get(h["evidence_ids"][0])
            if did in rows:
                rows[did][h["transition_type"]] += 1
    return sorted(rows.values(), key=lambda r: r["published"])


def build_brief(sit) -> str:
    d, out = sit.d, []
    out += ["# PasZel situation brief", f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} from stored state. "
            "No LLM was used to write this brief; every figure comes from a validated claim.", ""]
    for inc in d["incidents"].values():
        out += [f"## {inc['title']} ({inc['type']}, {inc['location']})", "",
                "| Subject | Measure | Value | Status | Source | As of | Supporting evidence | First evidence |", "|---|---|---|---|---|---|---|---|"]
        for c in d["claims"].values():
            if c["incident_id"] == inc["incident_id"] and c["status"] in ("active", "disputed"):
                e = d["evidence"][c["evidence_ids"][0]]
                out.append(f"| {c['subject']} | {c['predicate']} | {c['value']} | {c['status']} | {c['source']} | {c['reported_time']} | "
                           f"{len(c['evidence_ids'])} | {d['documents'][e['document_id']]['filename']} p.{e['page']} |")
        out.append("")
    pend = [p for p in d["proposals"].values() if p["status"] == "pending"]
    out += ["## Awaiting human decision" if pend else "## Awaiting human decision: none", ""]
    for p in pend:
        o, n = d["claims"][p["target_claim_id"]], d["claims"][p["new_claim_id"]]
        out.append(f"- **{p['transition_type']}** {o['predicate']}: {o['value']} ({o['source']}, {o['reported_time']}) vs "
                   f"{n['value']} ({n['source']}, {n['reported_time']}). {p['reason']}")
    out += ["", "## What each report did", "", "| Report | Source | Published | " + " | ".join(TRANS) + " |", "|---|---|---|" + "---|" * len(TRANS)]
    out += [f"| {r['report']} | {r['source']} | {r['published']} | " + " | ".join(str(r[t]) for t in TRANS) + " |" for r in report_log(sit)]
    out += ["", "## Documents", ""] + [f"- {m['filename']} | {m['source']} | published {m['publication_date']} | sha256 {m['sha256'][:16]} | {m['source_url']}"
                                       for m in d["documents"].values()]
    return "\n".join(out) + "\n"
