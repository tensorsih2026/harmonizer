"""Why 0.82? — answered with a curve instead of an assertion.

The auto-merge floor is the single most consequential number in this system.
Above it a family is accepted without anybody looking; below it a person has to
read it. Move it up and you catch more mistakes and create more work. Move it
down and the queue empties and errors go out unexamined.

Until now the honest answer to "why 0.82?" was "we picked it", which is the
weakest sentence in the whole project — and it is the first question anyone
who has run a matching system will ask.

This module answers it properly. It does not re-cluster; it re-reads. Family
membership is fixed by the similarity threshold at run time, but the *band* a
family lands in is a pure function of its mean score, so every candidate floor
can be evaluated instantly against the run that already exists.

What it reports at each floor:

  auto        families that would be accepted with no human involvement
  review      families a person would have to read
  records     how many source records sit behind the review pile, because
              "18 families" and "9,000 records" are different amounts of work

and, when the upload carries a ground-truth column, the number that turns the
whole thing from a preference into a decision:

  auto_correct   of the families accepted WITHOUT review at this floor, the
                 fraction that really are one item

That is the operating-point question a material master actually poses. Not
"what is the accuracy" — the F1 for the run is fixed, and raising the band
cannot change how the matcher grouped anything. The question is: how much
unreviewed output are you willing to be wrong about, and how much reading does
buying that certainty cost. A curve answers it. A constant in a file does not.

Correctness of a family is measured by purity: every labelled member carries
the same ground-truth label. A family that mixes two real items is wrong no
matter how confident the matcher was about it, and it is exactly the kind of
family a higher floor is supposed to catch.
"""

from __future__ import annotations

import pandas as pd

from harmonizer import CANONICAL_LABEL, STATUS_BANDS

# The floors worth showing. Below 0.55 nothing is being filtered; above 0.99
# everything goes to review and the tool has stopped making decisions.
FLOOR_LOW = 0.55
FLOOR_HIGH = 0.99
FLOOR_STEP = 0.01


def current_floor() -> float:
    """The floor the run actually used."""
    return float(next(f for n, f in STATUS_BANDS if n == "accepted"))


def _family_purity(result) -> dict[str, bool]:
    """family_id -> is every labelled member the same real item?

    Families with no labelled members are absent rather than assumed correct.
    Counting an unmeasurable family as a success is how a sweep flatters
    itself."""
    records = result.records
    if CANONICAL_LABEL not in records.columns:
        return {}
    labelled = records[records[CANONICAL_LABEL].astype(str).str.strip() != ""]
    if labelled.empty:
        return {}
    pure: dict[str, bool] = {}
    for family_id, group in labelled.groupby("Family_ID"):
        pure[str(family_id)] = group[CANONICAL_LABEL].astype(str).nunique() == 1
    return pure


def sweep(result) -> dict:
    """The whole curve, plus the point the run is standing on."""
    clusters = result.clusters
    if clusters.empty:
        return {"available": False, "reason": "This run produced no families."}

    scores = pd.to_numeric(clusters["Mean_Similarity"], errors="coerce").fillna(0.0)
    members = pd.to_numeric(clusters["Members"], errors="coerce").fillna(0).astype(int)
    ids = clusters["Family_ID"].astype(str)

    purity = _family_purity(result)
    measurable = bool(purity)

    rows = []
    floor = FLOOR_LOW
    while floor <= FLOOR_HIGH + 1e-9:
        above = scores >= floor
        auto = int(above.sum())
        review = int((~above).sum())
        review_records = int(members[~above].sum())

        entry = {
            "floor": round(floor, 2),
            "auto": auto,
            "review": review,
            "review_records": review_records,
            "auto_share": round(auto / len(clusters), 4) if len(clusters) else 0.0,
        }

        if measurable:
            # Only families we can actually judge count in either direction.
            judged = [pid for pid, keep in zip(ids, above) if keep and pid in purity]
            correct = sum(1 for pid in judged if purity[pid])
            entry["auto_judged"] = len(judged)
            entry["auto_correct"] = round(correct / len(judged), 4) if judged else None
            # The other half of the trade: wrong families that a person WOULD
            # now catch, because they fell below the floor.
            caught = [pid for pid, keep in zip(ids, above)
                      if not keep and pid in purity and not purity[pid]]
            missed = [pid for pid, keep in zip(ids, above)
                      if keep and pid in purity and not purity[pid]]
            entry["wrong_caught"] = len(caught)
            entry["wrong_shipped"] = len(missed)

        rows.append(entry)
        floor += FLOOR_STEP

    active = current_floor()
    at_current = min(rows, key=lambda r: abs(r["floor"] - active))

    return {
        "available": True,
        "measurable": measurable,
        "families": int(len(clusters)),
        "records": int(members.sum()),
        "labelled_families": len(purity),
        "current": round(active, 2),
        "at_current": at_current,
        "points": rows,
        "reading": _reading(at_current, measurable, len(clusters)),
    }


def _reading(point: dict, measurable: bool, families: int) -> str:
    """The sentence under the chart, in the run's own numbers."""
    base = (
        f"At the floor this run used, {point['auto']:,} of {families:,} families "
        f"are accepted without anybody reading them and {point['review']:,} "
        f"{'goes' if point['review'] == 1 else 'go'} to the queue — "
        f"{point['review_records']:,} source records to look at."
    )
    if not measurable or point.get("auto_correct") is None:
        return base + (
            " Upload a file carrying a verified grouping column and this becomes a "
            "decision rather than a preference: you would also see what fraction of "
            "the unread families are actually right at each floor."
        )
    shipped = point.get("wrong_shipped", 0)
    return base + (
        f" Of the families accepted unread, {point['auto_correct'] * 100:.1f}% really "
        f"are one item; {shipped:,} wrong "
        f"{'family goes' if shipped == 1 else 'families go'} out unexamined. "
        "Raising the floor buys some of that back and costs reading time. That "
        "trade is the decision — not the number itself."
    )
