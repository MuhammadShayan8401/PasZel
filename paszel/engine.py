"""PasZel situation-state engine.

The LLM only *proposes* claims. Everything in this file is deterministic:
validation against the source page text, transition classification, and
state changes. UPDATE and CONFLICT never apply without a human decision.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
import uuid
from datetime import datetime, timedelta, timezone

PREDICATES = {
    "deaths", "injured", "missing", "houses_damaged", "houses_destroyed",
    "people_affected", "displaced", "livestock_lost",
}
CONFLICT_WINDOW = timedelta(hours=24)  # reports this close in time are "same relevant time"
FUZZY = 0.85


class ValidationError(Exception):
    pass


def norm(s) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def _t(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s)
    except (TypeError, ValueError) as e:
        raise ValidationError(f"missing/invalid timestamp: {s!r}") from e


def _same(a: str, b: str) -> bool:
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio() >= FUZZY


def validate_claim(c: dict, pages: dict[int, str], default_time: str | None) -> dict:
    """Reject anything not provable from the page text. Returns a cleaned claim."""
    if not isinstance(c, dict):
        raise ValidationError("claim is not an object")
    pred = norm(c.get("predicate")).replace(" ", "_")
    if pred not in PREDICATES:
        raise ValidationError(f"predicate {pred!r} not in allowed set")
    try:
        value = float(c.get("value"))
    except (TypeError, ValueError):
        raise ValidationError("value is not numeric")
    if value < 0 or value != int(value):
        raise ValidationError("value must be a non-negative integer")
    value = int(value)
    for k in ("subject", "location", "incident_title", "incident_type"):
        if not norm(c.get(k)):
            raise ValidationError(f"missing {k}")
    page = c.get("page")
    if page not in pages:
        raise ValidationError(f"page {page!r} not in document")
    quote = c.get("quote") or ""
    if not norm(quote) or norm(quote) not in norm(pages[page]):
        raise ValidationError("quote not found verbatim on cited page")
    # whole-number match: 8 must not match inside 28 / 1.28 / 83; leading zeros ("03") allowed
    if not re.search(rf"(?<![\d.])0*{value}(?!\d|\.\d)", quote.replace(",", "")):
        raise ValidationError("value does not appear in the quote")
    ph, pq = norm(pages[page]), norm(quote)
    nh = norm(c.get("heading"))
    if not nh or nh not in ph:
        raise ValidationError("insufficient evidence context: heading not found on cited page")
    if ph.find(nh) > ph.find(pq):
        raise ValidationError("insufficient evidence context: heading does not precede the quote")
    if norm(c["subject"]) not in ph:
        raise ValidationError("insufficient evidence context: subject not found on cited page")
    when = c.get("as_of") or default_time
    _t(when)  # raises if missing/invalid
    return {**c, "predicate": pred, "value": value, "as_of": when}


def _dependent(a: dict, b: dict) -> bool:
    """Same source, or one declares it was derived from the other: not independent."""
    sa, sb = norm(a["source"]), norm(b["source"])
    return sa == sb or norm(a.get("derived_from")) == sb or norm(b.get("derived_from")) == sa


def classify(existing: dict | None, new: dict) -> tuple[str, str]:
    """Deterministic transition rules. Never 'newest wins'."""
    if existing is None:
        return "CREATE", "No existing claim for this incident/subject/predicate."
    same_src = _dependent(existing, new)
    if existing["value"] == new["value"]:
        return ("DUPLICATE", "Same value from the same (or a dependent) source; not independent corroboration.") if same_src else (
            "CORROBORATE", f"Independent source ({new['source']}) reports the same value.")
    # compare with the most recent time this claim was confirmed, not only when first reported
    dt = _t(new["reported_time"]) - _t(existing.get("last_confirmed") or existing["reported_time"])
    if dt > CONFLICT_WINDOW or (same_src and dt > timedelta(0)):
        return "UPDATE", f"Later report revises {existing['value']} -> {new['value']}."
    if dt < -CONFLICT_WINDOW:
        return "STALE", "Report is older than the current claim; kept as history only."
    return "CONFLICT", (f"{existing['source']} says {existing['value']}, {new['source']} says "
                        f"{new['value']} at the same relevant time.")


class Situation:
    def __init__(self, data: dict | None = None):
        self.d = data or {"documents": {}, "evidence": {}, "incidents": {}, "claims": {},
                          "history": [], "proposals": {}, "pages": {}}

    # ---- persistence
    def dumps(self) -> str:
        return json.dumps(self.d, indent=1)

    @classmethod
    def loads(cls, s: str) -> "Situation":
        return cls(json.loads(s))

    # ---- helpers
    def _log(self, incident_id, transition, prev, new, evidence_ids, reason, applied=True):
        h = self.d["history"]
        e = {"change_id": uuid.uuid4().hex[:8], "incident_id": incident_id,
             "transition_type": transition, "previous_claim": prev, "new_claim": new,
             "evidence_ids": evidence_ids, "reason": reason, "applied": applied,
             "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "prev_hash": h[-1]["hash"] if h else ""}
        e["hash"] = hashlib.sha256(json.dumps(e, sort_keys=True).encode()).hexdigest()
        h.append(e)

    def verify_history(self) -> bool:
        """Tamper-EVIDENT (not tamper-proof): any edit to an earlier entry breaks the chain."""
        prev = ""
        for e in self.d["history"]:
            body = {k: v for k, v in e.items() if k != "hash"}
            if e.get("prev_hash") != prev or hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != e.get("hash"):
                return False
            prev = e["hash"]
        return True

    def add_document(self, meta: dict) -> bool:
        """False if this exact file (sha256) was already ingested."""
        if meta["document_id"] in self.d["documents"]:
            return False
        self.d["documents"][meta["document_id"]] = meta
        return True

    def _incident_for(self, c: dict) -> str:
        for iid, inc in self.d["incidents"].items():
            if norm(inc["type"]) == norm(c["incident_type"]) and _same(inc["location"], c["location"]):
                return iid
        iid = "inc-" + uuid.uuid4().hex[:6]
        self.d["incidents"][iid] = {"incident_id": iid, "title": c["incident_title"],
                                    "type": c["incident_type"], "location": c["location"],
                                    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        return iid

    def _existing(self, iid, subject, predicate):
        live = [x for x in self.d["claims"].values()
                if x["incident_id"] == iid and x["predicate"] == predicate
                and _same(x["subject"], subject) and x["status"] in ("active", "disputed")]
        return max(live, key=lambda x: x["reported_time"]) if live else None

    # ---- main entry
    def ingest_claim(self, c: dict, doc: dict) -> dict:
        """c must already have passed validate_claim. Returns a result record."""
        eid = "ev-" + uuid.uuid4().hex[:8]
        self.d["evidence"][eid] = {"evidence_id": eid, "document_id": doc["document_id"],
                                   "page": c["page"], "text": c["quote"], "source": doc["source"],
                                   "timestamp": c["as_of"], "heading": c["heading"]}
        iid = self._incident_for(c)
        new = {"claim_id": "clm-" + uuid.uuid4().hex[:8], "incident_id": iid, "subject": c["subject"],
               "predicate": c["predicate"], "value": c["value"], "source": doc["source"],
               "evidence_ids": [eid], "reported_time": c["as_of"], "status": "active",
               "derived_from": doc.get("derived_from", ""), "last_confirmed": c["as_of"]}
        ex = self._existing(iid, c["subject"], c["predicate"])
        tr, reason = classify(ex, new)
        res = {"transition": tr, "reason": reason, "claim": new, "target": ex}
        if tr == "CREATE":
            self.d["claims"][new["claim_id"]] = new
            self._log(iid, tr, None, new["claim_id"], [eid], reason)
        elif tr in ("DUPLICATE", "CORROBORATE"):
            ex["evidence_ids"].append(eid)
            if _t(new["reported_time"]) > _t(ex.get("last_confirmed") or ex["reported_time"]):
                ex["last_confirmed"] = new["reported_time"]
            self._log(iid, tr, ex["claim_id"], None, [eid], reason)
        elif tr == "STALE":
            new["status"] = "stale"
            self.d["claims"][new["claim_id"]] = new
            self._log(iid, tr, ex["claim_id"], new["claim_id"], [eid], reason)
        else:  # UPDATE / CONFLICT -> needs a human
            new["status"] = "proposed"
            self.d["claims"][new["claim_id"]] = new
            if tr == "CONFLICT":
                ex["status"] = "disputed"
            pid = "prop-" + uuid.uuid4().hex[:6]
            self.d["proposals"][pid] = {"proposal_id": pid, "incident_id": iid, "transition_type": tr,
                                        "target_claim_id": ex["claim_id"], "new_claim_id": new["claim_id"],
                                        "reason": reason, "evidence_ids": [eid], "status": "pending"}
            self._log(iid, tr, ex["claim_id"], new["claim_id"], [eid], reason, applied=False)
            res["proposal_id"] = pid
        return res

    # ---- human verification
    def verify(self, pid: str, decision: str, value: int | None = None, note: str = "", reviewer: str = "") -> None:
        p = self.d["proposals"].get(pid)
        if not p or p["status"] != "pending":
            raise ValidationError("proposal not found or already decided")
        if not norm(reviewer):
            raise ValidationError("reviewer name required")
        old, new = self.d["claims"][p["target_claim_id"]], self.d["claims"][p["new_claim_id"]]
        verb = {"accept": "accepted", "modify": "modified", "reject": "rejected"}.get(decision)
        if verb is None:
            raise ValidationError(f"unknown decision {decision!r}")
        if decision == "modify" and (value is None or int(value) < 0):
            raise ValidationError("modify needs a non-negative value")
        if decision == "reject":
            new["status"] = "rejected"
            if old["status"] == "disputed":
                old["status"] = "active"
            p["status"] = "rejected"
            kind = "REJECT"
        else:
            if decision == "modify":
                new["original_proposed_value"], new["value"] = new["value"], int(value)
            old["status"], new["status"] = "superseded", "active"
            p["status"] = "verified"
            kind = "VERIFY"
        p["reviewer"] = reviewer.strip()
        self._log(p["incident_id"], kind, old["claim_id"], new["claim_id"], p["evidence_ids"],
                  f"{reviewer.strip()} {verb} {p['transition_type']}. {note}".strip())
