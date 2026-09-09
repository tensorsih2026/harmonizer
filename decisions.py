"""The audit trail: every decision a person made, in the order they made it.

Why this exists
---------------
Before this module the app had three real reviewer verbs — approve, split,
merge — and each of them mutated the run in place and then forgot how the run
had got that way. That produced a specific, quiet failure: `export.csv` was
byte-identical whether a reviewer had confirmed forty families, merged three
and split two, or had done nothing at all. Every hour of human judgement
evaporated into a file that did not know it had happened.

For a material master that is the wrong way round. The mapping is the easy
half; an evaluator's actual question is *who decided this, and when*. A code
that a person confirmed and a code the matcher guessed at 0.71 are not the
same claim, and a published mapping that cannot tell them apart is asking to
be trusted rather than read.

So: an append-only log. One entry per decision, in sequence, with a sentence
of plain words beside it. It is a record, not a cache — nothing in the app
reads back from it to decide anything.

Two properties are deliberate.

**It is written in the same call that makes the change.** `record()` is
invoked inside `apply_merge` and friends, after the frames have actually
moved, so a decision cannot appear in the log without having happened or
happen without appearing. There is no separate "commit" step to forget.

**Every entry carries what it would take to reverse it** — the row indices
that moved and the values they held before. That costs a list of integers per
decision and it is the difference between a log you can read and a log you can
replay. Undo is not built yet; when it is, it reads this and nothing else.
A decision that moved more rows than `REVERSIBLE_ROW_LIMIT` records its
counts but not its per-row detail, and marks itself `reversible: False`
rather than pretending.

Nothing here is written back to an ERP. The log lives in the run, for the
session, exactly like the decisions it describes.
"""

from __future__ import annotations

import datetime as _dt
from typing import Optional

import pandas as pd

# Past this many moved rows an entry stops carrying its per-row restore data.
# 20,000 integers is a few hundred kilobytes; a 50,000-row family merged in
# one action is not something a session should hold three copies of.
REVERSIBLE_ROW_LIMIT = 20_000

# The verbs, and how each one reads in a sentence.
VERBS = ("approve", "split", "merge")

# Columns of the exported log, in the order an auditor reads them.
LOG_COLUMNS = [
    "Seq",
    "Decided_At",
    # Who. An audit trail that records what changed but not who changed it
    # answers half the question a Ministry evaluator is actually asking.
    "Decided_By",
    "Action",
    "Canonical_Code",
    "Family_ID",
    "Affected_Records",
    "Related_Code",
    "Related_Family_ID",
    "Summary",
    "Reversible",
    # An undone decision is not deleted from the trail. "Merged and then
    # unmerged" and "never touched" are different histories.
    "Undone",
    "Undone_At",
    "Undone_By",
]


UNATTRIBUTED = "unattributed"

# A signature, not a login. Long enough for a name and a unit, short enough
# that nobody pastes a paragraph into an audit file.
MAX_NAME = 64


def _clean_name(raw: str) -> str:
    """One line, trimmed, no commas to break the CSV it is written into."""
    text = " ".join(str(raw or "").split())
    text = text.replace(",", " ").replace('"', "").strip()
    return text[:MAX_NAME]


def _now() -> str:
    """UTC, to the second, in a form a spreadsheet will not reinterpret."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def log_of(result) -> list:
    """The run's decision list, created on first use.

    Older results — and any pickled from a previous version — will not have the
    attribute, so this never assumes it is there."""
    existing = getattr(result, "decisions", None)
    if existing is None:
        existing = []
        try:
            result.decisions = existing
        except Exception:      # pragma: no cover - frozen result objects
            return []
    return existing


def record(result, verb: str, *, family_id: str, code: str, summary: str,
           affected: int, related_id: str = "", related_code: str = "",
           by: str = "", restore: Optional[dict] = None,
           detail: Optional[dict] = None) -> dict:
    """Append one decision. Returns the entry, so callers can echo it back.

    `summary` is the sentence shown to a person and written into the CSV, so it
    is written for a reader, not for a machine: "MTL-0000002 merged into
    MTL-0000001; 48 records re-pointed", not "merge(a,b)".

    `restore` is the reversal payload — whatever undo will need and nothing
    else. It is dropped, with `reversible` set False, when it is too large to
    hold honestly."""
    if verb not in VERBS:
        raise ValueError(f"unknown decision verb {verb!r}")

    log = log_of(result)
    reversible = True
    if restore is None:
        restore = {}
    rows = restore.get("indices") or []
    if len(rows) > REVERSIBLE_ROW_LIMIT:
        restore = {"rows_dropped": len(rows)}
        reversible = False

    entry = {
        "seq": len(log) + 1,
        "at": _now(),
        # Typed by the reviewer in the workspace. There is no login, so this is
        # a signature rather than an authenticated identity, and it is labelled
        # that way everywhere it is shown — a name a person chose to put on
        # their own work is still the difference between an accountable
        # decision and an anonymous one.
        "by": _clean_name(by),
        "verb": verb,
        "family_id": str(family_id),
        "code": str(code),
        "related_id": str(related_id or ""),
        "related_code": str(related_code or ""),
        "affected": int(affected),
        "summary": summary,
        "reversible": reversible,
        "restore": restore,
        "detail": detail or {},
    }
    log.append(entry)
    return entry


def undoable(result) -> Optional[dict]:
    """The one entry that may be undone right now, or None.

    Last in, first out, and nothing else. Decisions compound: a split creates a
    family that a later merge can absorb, so reversing them out of order would
    restore rows into a family that no longer exists. Rather than build a
    dependency graph for a session-scoped log, undo does what undo does
    everywhere — it takes back the last thing you did."""
    for entry in reversed(log_of(result)):
        if entry.get("undone"):
            continue
        return entry if entry.get("reversible") else None
    return None


def mark_undone(entry: dict, by: str = "") -> None:
    """An undone decision stays in the log, marked.

    Deleting it would make the audit trail lie by omission: "this family was
    merged and then unmerged" and "nothing ever happened to this family" are
    different histories, and only one of them is true."""
    entry["undone"] = True
    entry["undone_at"] = _now()
    entry["undone_by"] = _clean_name(by)


def public(entry: dict) -> dict:
    """One entry as the browser sees it — without the reversal payload.

    The restore data is row indices and prior scores. It is what undo needs and
    it is of no use to a reader, so it does not travel to the page."""
    return {
        "seq": entry["seq"],
        "at": entry["at"],
        "by": entry.get("by", ""),
        "verb": entry["verb"],
        "family_id": entry["family_id"],
        "code": entry["code"],
        "related_id": entry["related_id"],
        "related_code": entry["related_code"],
        "affected": entry["affected"],
        "summary": entry["summary"],
        "reversible": entry["reversible"],
        "undone": bool(entry.get("undone")),
        "undone_at": entry.get("undone_at", ""),
        "undone_by": entry.get("undone_by", ""),
        "detail": entry["detail"],
    }


def summary(result) -> dict:
    """Counts for the header of the decisions surface."""
    log = log_of(result)
    counts = {verb: 0 for verb in VERBS}
    records_touched = 0
    live = [e for e in log if not e.get("undone")]
    for entry in live:
        counts[entry["verb"]] = counts.get(entry["verb"], 0) + 1
        records_touched += int(entry["affected"])
    reviewers: dict[str, int] = {}
    for entry in live:
        who = entry.get("by") or UNATTRIBUTED
        reviewers[who] = reviewers.get(who, 0) + 1
    can = undoable(result)
    return {
        "reviewers": dict(sorted(reviewers.items(), key=lambda kv: -kv[1])),
        # `total` is decisions still standing. `logged` is everything ever
        # done, undone included, because that is what the audit file holds.
        "total": len(live),
        "logged": len(log),
        "undone": len(log) - len(live),
        "by_verb": counts,
        "records_touched": records_touched,
        "first_at": log[0]["at"] if log else "",
        "last_at": log[-1]["at"] if log else "",
        "undoable_seq": can["seq"] if can else 0,
        "undoable_summary": can["summary"] if can else "",
    }


def log_frame(result) -> pd.DataFrame:
    """The log as a table, for `decisions.csv`.

    Always returns the full column set, even with no decisions in it — an empty
    audit file with headers says "nobody changed anything", which is a real
    answer. A file with no columns says nothing at all."""
    rows = []
    for entry in log_of(result):
        rows.append({
            "Seq": entry["seq"],
            "Decided_At": entry["at"],
            "Decided_By": entry.get("by") or UNATTRIBUTED,
            "Action": entry["verb"],
            "Canonical_Code": entry["code"],
            "Family_ID": entry["family_id"],
            "Affected_Records": entry["affected"],
            "Related_Code": entry["related_code"],
            "Related_Family_ID": entry["related_id"],
            "Summary": entry["summary"],
            "Reversible": "yes" if entry["reversible"] else "no",
            "Undone": "yes" if entry.get("undone") else "no",
            "Undone_At": entry.get("undone_at", ""),
            "Undone_By": (entry.get("undone_by") or UNATTRIBUTED) if entry.get("undone") else "",
        })
    return pd.DataFrame(rows, columns=LOG_COLUMNS)


# --------------------------------------------------------------------------
# Per-record provenance, derived from the log
# --------------------------------------------------------------------------

def review_states(result) -> dict:
    """family_id -> (state, when, merged_from, split_from) from the log alone.

    Derived rather than stored, so it cannot drift from the decisions it claims
    to describe. A family touched by more than one verb keeps the LAST thing
    that happened to it, which is what "review state" means.

    The four states:
      unreviewed  the matcher's own grouping, untouched by a person
      confirmed   a reviewer signed off on this mapping
      split       a reviewer separated this family, or it came out of one
      merged      a reviewer folded another family into this one
    """
    state: dict[str, dict] = {}
    for entry in log_of(result):
        if entry.get("undone"):
            continue
        for fid in (entry["family_id"], entry.get("detail", {}).get("new_family_id", "")):
            if not fid:
                continue
            current = state.setdefault(fid, {
                "state": "unreviewed", "at": "", "by": "",
                "merged_from": "", "split_from": "",
            })
            current["at"] = entry["at"]
            current["by"] = entry.get("by") or UNATTRIBUTED
            if entry["verb"] == "approve":
                current["state"] = "confirmed"
            elif entry["verb"] == "split":
                current["state"] = "split"
                current["split_from"] = entry["code"]
            elif entry["verb"] == "merge":
                current["state"] = "merged"
                current["merged_from"] = entry["related_code"]
    return state


def audit_frame(records: pd.DataFrame, result) -> pd.DataFrame:
    """The published mapping with four provenance columns added.

    This is a SEPARATE export, reached with `?audit=1`. The default
    `export.csv` keeps exactly the shape it has always had, because a
    downstream loader that has been fed that shape for weeks should not
    silently receive four new columns because a reviewer clicked something."""
    states = review_states(result)
    frame = records.copy()
    fids = frame["Family_ID"].astype(str)

    def column(key: str, default: str = "") -> list:
        return [states.get(fid, {}).get(key, default) for fid in fids]

    # The proposed code rides along in the audited export, because a mapping a
    # person is going to load somewhere is the place a proposed scheme is
    # actually useful. Absent from the plain export, like everything proposed.
    if "Proposed_Code" in getattr(result, "clusters", frame).columns:
        lookup = dict(zip(result.clusters["Family_ID"].astype(str),
                          result.clusters["Proposed_Code"].astype(str)))
        frame["Proposed_Code"] = [lookup.get(f, "") for f in fids]

    frame["Review_State"] = column("state", "unreviewed")
    frame["Reviewed_At"] = column("at")
    frame["Reviewed_By"] = column("by")
    frame["Merged_From"] = column("merged_from")
    frame["Split_From"] = column("split_from")
    return frame
