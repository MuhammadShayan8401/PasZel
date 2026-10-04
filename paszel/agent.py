"""Read-only analyst agent: plans tool calls, answers only from tool results, citations are verified.
It can NOT modify the situation: tools only read."""
import json
import re

from .engine import ValidationError
from .retrieval import chunk_pages, search

SYSTEM = """You are a read-only analyst for a disaster situation tracker. Answer ONLY from tool results.
Reply with ONLY one JSON object per turn:
  {"tool": "<name>", "args": {...}}   to call a tool, or
  {"final": "<answer>", "citations": ["<id>", ...]}   when done, or
  {"final": "<why evidence is missing>", "insufficient": true}   if the tools returned nothing relevant.
Tools: search_evidence(query, k) - keyword search over stored report pages (ids look like doc:page:n);
get_current_claims(predicate) - current tracked claims (ids clm-...); get_pending_proposals() - changes awaiting
a human (ids prop-...); get_history(limit) - recent state changes (ids = change ids).
Rules: cite only ids returned by tools; never invent numbers; at most 5 tool calls; state when reports disagree."""


class AgentError(ValidationError):
    def __init__(self, msg, trace=None):
        super().__init__(msg)
        self.trace = trace or []


def _tools(sit):
    d = sit.d
    name = lambda i: d["documents"].get(i, {}).get("filename", i)

    def search_evidence(query="", k=5):
        return [{"id": r["id"], "document": name(r["doc"]), "page": r["page"], "text": r["text"][:700]}
                for r in search(d, str(query), k=min(int(k), 8))]

    def get_current_claims(predicate=None):
        return [{"id": c["claim_id"], "incident": d["incidents"][c["incident_id"]]["title"], "subject": c["subject"],
                 "predicate": c["predicate"], "value": c["value"], "source": c["source"], "reported": c["reported_time"],
                 "status": c["status"], "evidence": len(c["evidence_ids"])}
                for c in d["claims"].values() if c["status"] in ("active", "disputed")
                and (not predicate or c["predicate"] == predicate)]

    def get_pending_proposals():
        return [{"id": p["proposal_id"], "type": p["transition_type"], "reason": p["reason"],
                 "current": d["claims"][p["target_claim_id"]]["value"], "proposed": d["claims"][p["new_claim_id"]]["value"]}
                for p in d["proposals"].values() if p["status"] == "pending"]

    def get_history(limit=10):
        return [{"id": h["change_id"], "type": h["transition_type"], "reason": h["reason"], "at": h["timestamp"]}
                for h in d["history"][-min(int(limit), 25):]]

    return {f.__name__: f for f in (search_evidence, get_current_claims, get_pending_proposals, get_history)}


def _ids(o) -> set:
    if isinstance(o, dict):
        return ({o["id"]} if isinstance(o.get("id"), str) else set()) | set().union(*map(_ids, o.values()), set())
    return set().union(*map(_ids, o), set()) if isinstance(o, list) else set()


def run_agent(provider, sit, question: str, max_steps: int = 5) -> dict:
    tools, seen, trace = _tools(sit), set(), []
    transcript = f"QUESTION: {question}\n"
    for step in range(max_steps + 1):
        raw = provider.complete(SYSTEM, transcript)
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            msg = json.loads(m.group(0))
        except (AttributeError, json.JSONDecodeError):
            raise AgentError("agent output was not valid JSON", trace) from None
        if "final" in msg:
            cites = msg.get("citations") or []
            if msg.get("insufficient") is True:
                return {"answer": str(msg["final"]), "citations": [], "insufficient": True, "trace": trace}
            if not cites or not set(cites) <= seen:
                raise AgentError(f"answer rejected: citations {sorted(set(cites) - seen) or 'missing'} were not returned by any tool", trace)
            return {"answer": str(msg["final"]), "citations": cites, "insufficient": False, "trace": trace}
        tool = msg.get("tool")
        if tool not in tools or step == max_steps:
            raise AgentError(f"agent asked for unknown tool {tool!r} or exceeded {max_steps} tool calls", trace)
        try:
            res = tools[tool](**(msg.get("args") or {}))
        except (TypeError, ValueError) as e:
            res = {"error": str(e)}
        seen |= _ids(res)
        trace.append({"step": step + 1, "tool": tool, "args": msg.get("args") or {}, "results": len(res) if isinstance(res, list) else 1})
        transcript += f"\nTOOL CALL {step + 1}: {json.dumps(msg)}\nRESULT: {json.dumps(res)[:6000]}\n"
    raise AgentError("no final answer", trace)


def describe_id(sit, i: str):
    d = sit.d
    for c in chunk_pages(d):
        if c["id"] == i:
            return {"kind": "evidence page text", "document": d["documents"][c["doc"]]["filename"], "page": c["page"], "text": c["text"]}
    for coll, kind in (("claims", "claim"), ("proposals", "proposal")):
        if i in d[coll]:
            return {"kind": kind, **d[coll][i]}
    return next(({"kind": "history entry", **h} for h in d["history"] if h["change_id"] == i), None)
