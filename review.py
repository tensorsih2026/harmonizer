"""Manual family splits.

The matcher's job is to group. A reviewer's job is to be able to disagree with
it. Before this module the "Split family" button raised a toast that said the
family had been sent to the review queue and then did precisely nothing — the
queue was unchanged, the family was unchanged, and anyone who clicked it twice
and then opened the queue caught the lie.

Two things were missing, and the first is the one that matters more:

  1. A reviewer could not see WHAT would be separated before agreeing to it.
     `plan_split` returns the exact rows that would break away, grouped by
     spelling, with the seam it proposes to cut on and why that seam.
  2. The split did not change the run. `apply_split` moves those rows into a
     new family, re-encodes them against their own new golden record so the
     similarity figures are real rather than inherited, puts BOTH resulting
     families into the review queue, and rewrites the run's statistics so
     every count on the page still agrees with every other one.

Where the seam comes from
-------------------------
Not a fixed threshold. A fixed cut-off answers "which rows are below 0.82",
which is a question about the scale, not about this family. The cut here is
the widest gap between consecutive similarity scores inside the family — the
one place where the members stop resembling each other gradually and start
resembling each other not at all. It is a one-dimensional Jenks break, it is
computed from the family's own numbers, and it can be shown to the reviewer as
a single sentence: "the widest gap is between 0.94 and 0.71".

A family whose members all score identically has no such seam, and this module
refuses to invent one. It says so instead.

Nothing here is written back to an ERP. It edits the in-memory result for this
session, which is what the detail pane's "Session only" label has always said.
"""

from __future__ import annotations

import math

import pandas as pd

import decisions as decision_log
import uom
from harmonizer import (
    CANONICAL_PRICE,
    CANONICAL_UOM,
    threshold_manifest,
    HIGH_CONFIDENCE,
    MEDIUM_CONFIDENCE,
    MIN_ROWS_FOR_UNIT_OUTLIER,
    status_for,
)

# Two members is the smallest thing that can be cut in half.
MIN_MEMBERS_TO_SPLIT = 2

# How wide the seam has to be before it means anything.
#
# The widest-gap rule always finds a seam: any family with two distinct scores
# has a widest gap, so the tool would happily offer to cut a family whose worst
# member still matches at 0.96. That is how one pump family, spelled ten ways
# and correctly grouped, got proposed for a split at a gap of 0.04 — and the
# dialog presented it with exactly the same confidence as a real 0.18 cliff.
#
# A split is SUPPORTED when either signal is present:
#   * the gap is a genuine cliff, not noise, or
#   * the breakaway rows sit below the auto-merge floor — the matcher would
#     not have merged them on its own, so a human was always going to have to
#     look at them.
# Requiring both together would be wrong: a family that splits cleanly at
# 0.99 / 0.81 is a real find even though every row is above the floor.
#
# When neither holds the split is WEAK. It is not forbidden — the reviewer may
# know something the scores do not — but it has to be taken deliberately,
# against a stated reason not to.
MIN_MEANINGFUL_GAP = 0.10

# How many distinct spellings the preview lists per side. A reviewer needs to
# recognise the group, not audit all 200 rows of it; the count is stated in
# full either way.
PREVIEW_SPELLINGS = 12


class SplitError(Exception):
    """The family cannot be split, with a reason fit to show a user."""


# --------------------------------------------------------------------------
# Reading the family


def _members(result, family_id: str) -> pd.DataFrame:
    frame = result.records[result.records["Family_ID"] == family_id]
    if frame.empty:
        raise SplitError(f"Family {family_id} is not in this run.")
    return frame


def _seam(scores: list[float]) -> tuple[float, float]:
    """The widest gap between consecutive distinct scores.

    Returns (below, above): every member at or under `below` breaks away.
    Ties are resolved toward the LOWER cut, which keeps the breakaway group
    the smaller and more obviously wrong of the two — a reviewer should be
    shown the few rows that look out of place, not handed a coin flip."""
    distinct = sorted(set(scores))
    best_gap = -1.0
    best_pair = (distinct[0], distinct[1])
    for lower, upper in zip(distinct, distinct[1:]):
        gap = upper - lower
        if gap > best_gap:
            best_gap, best_pair = gap, (lower, upper)
    return best_pair


def _spellings(frame: pd.DataFrame, limit: int = PREVIEW_SPELLINGS) -> list[dict]:
    """One entry per distinct source spelling, most-repeated first."""
    groups: dict[str, dict] = {}
    for rec in frame.itertuples(index=False):
        raw = str(rec.Original_Description)
        key = " ".join(raw.split()).upper()
        entry = groups.get(key)
        if entry is None:
            entry = groups[key] = {
                "raw": raw[:160],
                "normalized": str(rec.Normalized_Description)[:160],
                "score": float(rec.Semantic_Similarity),
                "records": 0,
                "units": [],
                "ids": [],
            }
        entry["records"] += 1
        unit = str(rec.CPSE)
        if unit and unit not in entry["units"]:
            entry["units"].append(unit)
        code = str(rec.Original_Material_Code)
        if code and len(entry["ids"]) < 3:
            entry["ids"].append(code)
        entry["score"] = min(entry["score"], float(rec.Semantic_Similarity))
    ordered = sorted(groups.values(), key=lambda e: (-e["records"], e["score"]))
    return ordered[:limit]


def _golden_of(frame: pd.DataFrame) -> str:
    """Longest normalized description wins, exactly as the pipeline chooses."""
    texts = frame["Normalized_Description"].astype(str)
    return max(texts, key=len)


def split_frames(result, family_id: str):
    """(keep, breakaway, below, above) or a SplitError explaining why not."""
    members = _members(result, family_id)
    if len(members) < MIN_MEMBERS_TO_SPLIT:
        raise SplitError(
            "This family has a single record. There is nothing to separate it from."
        )

    scores = [float(s) for s in members["Semantic_Similarity"]]
    if len(set(scores)) < 2:
        raise SplitError(
            "Every record in this family matched the golden record at exactly "
            f"{scores[0]:.2f}. There is no seam to cut on — the matcher found no "
            "member less like the others than any other member."
        )

    below, above = _seam(scores)
    breakaway = members[members["Semantic_Similarity"] <= below]
    keep = members[members["Semantic_Similarity"] > below]
    return keep, breakaway, below, above


def plan_split(result, family_id: str) -> dict:
    """What a split would do, in enough detail to agree or refuse."""
    row = result.clusters[result.clusters["Family_ID"] == family_id]
    if row.empty:
        raise SplitError(f"Family {family_id} is not in this run.")
    row = row.iloc[0]

    keep, breakaway, below, above = split_frames(result, family_id)
    proposed = _golden_of(breakaway)

    floor = threshold_manifest()["auto_merge"]
    gap = above - below
    breakaway_max = max(float(s) for s in breakaway["Semantic_Similarity"])
    wide_gap = gap >= MIN_MEANINGFUL_GAP
    below_floor = breakaway_max < floor

    if wide_gap:
        why = ("The scores fall off sharply here — a gap of %.2f, wide enough to be a "
               "real boundary rather than noise." % gap)
    elif below_floor:
        why = ("Every record on the left matches below the %.2f auto-merge floor, so the "
               "matcher was never confident about them." % floor)
    else:
        why = ("This looks like one part, not two. The widest gap in the family is only "
               "%.2f, and the least similar record on the left still matches at %.2f — "
               "above the %.2f auto-merge floor. Splitting here separates spellings of "
               "the same thing." % (gap, breakaway_max, floor))

    return {
        "weak": not (wide_gap or below_floor),
        "why": why,
        "signals": {
            "gap": round(gap, 4), "wide_gap": bool(wide_gap),
            "min_gap": MIN_MEANINGFUL_GAP,
            "breakaway_max": round(breakaway_max, 4),
            "below_floor": bool(below_floor), "floor": floor,
        },
        "family_id": family_id,
        "code": str(row.Canonical_Code),
        "desc": str(row.Golden_Record),
        "members": int(row.Members),
        "cut": {
            "below": round(below, 4),
            "above": round(above, 4),
            "gap": round(above - below, 4),
        },
        "breakaway": {
            "records": int(len(breakaway)),
            "units": sorted({str(u) for u in breakaway["CPSE"]}),
            "proposed_desc": proposed,
            "spellings": _spellings(breakaway),
            "more": max(0, len(set(
                " ".join(str(d).split()).upper() for d in breakaway["Original_Description"]
            )) - PREVIEW_SPELLINGS),
        },
        "keep": {
            "records": int(len(keep)),
            "desc": str(row.Golden_Record),
            "spellings": _spellings(keep),
            "more": max(0, len(set(
                " ".join(str(d).split()).upper() for d in keep["Original_Description"]
            )) - PREVIEW_SPELLINGS),
        },
    }


# --------------------------------------------------------------------------
# Applying it


def _confidence_for(score: float) -> str:
    if score >= HIGH_CONFIDENCE:
        return "HIGH"
    if score >= MEDIUM_CONFIDENCE:
        return "MEDIUM"
    return "LOW"


# A family a human pulled apart is not "auto-merged" afterwards, whatever the
# arithmetic says about the remainder. Both halves are a decision waiting to be
# made, so neither is allowed to rank above the review band.
_NEVER_BETTER_THAN = "review"
_BAND_ORDER = ["resolved", "accepted", "review", "conflict"]


def _capped(status: str) -> str:
    return status if _BAND_ORDER.index(status) >= _BAND_ORDER.index(_NEVER_BETTER_THAN) \
        else _NEVER_BETTER_THAN


def _price_block(frame: pd.DataFrame) -> dict:
    """The same arithmetic the pipeline does, over one family's rows.

    Duplicated deliberately rather than imported: the pipeline computes this
    from the cleaned upload while looping clusters, and reaching back into that
    loop from here would mean keeping the whole cleaned frame alive for the
    lifetime of every run."""
    out = {
        "Priced_Records": 0, "Spend": None, "Price_Low": None, "Price_High": None,
        "Price_Median": None, "Price_Spread": None, "Savings": None, "Excess": None,
        "Unit_Medians": "", "Outlier_Unit": None, "Outlier_Ratio": None,
        "Outlier_Excess": None, "Basis_UOM": None, "Other_UOM_Records": 0,
        "UOM_Forms": 0, "UOM_List": "",
    }
    if CANONICAL_PRICE not in frame.columns:
        return out

    # "Unit Price" has a space in it, so itertuples renames it to a positional
    # attribute. Read the column directly rather than guess what it was called.
    prices = pd.to_numeric(frame[CANONICAL_PRICE], errors="coerce")
    units = frame["CPSE"].astype(str)
    measures = (frame[CANONICAL_UOM].astype(str) if CANONICAL_UOM in frame.columns
                else pd.Series([uom.UNSPECIFIED] * len(frame), index=frame.index))

    # Same rule as the pipeline: figures are computed inside ONE unit of
    # measure, because a price per piece and a price per kilo are not two
    # points on one scale, and this application does not invent a conversion.
    by_measure: dict[str, list] = {}
    total_priced = 0
    for price, unit_name, measure in zip(prices, units, measures):
        if price is None or pd.isna(price) or float(price) <= 0:
            continue
        total_priced += 1
        by_measure.setdefault(str(measure), []).append((unit_name, float(price)))

    if not by_measure:
        return out

    basis = max(by_measure, key=lambda m: len(by_measure[m]))
    unit_prices = by_measure[basis]
    values = [v for _, v in unit_prices]
    out["Basis_UOM"] = basis
    out["Other_UOM_Records"] = total_priced - len(values)
    out["UOM_Forms"] = len(by_measure)
    out["UOM_List"] = "|".join(
        f"{m}:{len(v)}" for m, v in sorted(by_measure.items(), key=lambda kv: -len(kv[1]))
    )

    values.sort()
    priced = len(values)
    spend = sum(values)
    low, high = values[0], values[-1]
    mid = priced // 2
    median = values[mid] if priced % 2 else (values[mid - 1] + values[mid]) / 2

    by_unit: dict[str, list[float]] = {}
    for unit, price in unit_prices:
        by_unit.setdefault(unit, []).append(price)

    unit_medians, worst_unit, worst_ratio, worst_excess = [], None, 1.0, 0.0
    for unit, unit_values in by_unit.items():
        unit_values.sort()
        k = len(unit_values)
        if k < MIN_ROWS_FOR_UNIT_OUTLIER:
            continue
        half = k // 2
        unit_median = unit_values[half] if k % 2 else (unit_values[half - 1] + unit_values[half]) / 2
        ratio = unit_median / median if median else 1.0
        unit_medians.append((unit, round(unit_median, 2), k, round(ratio, 3)))
        if ratio > worst_ratio:
            worst_ratio, worst_unit = ratio, unit
            worst_excess = sum(v - median for v in unit_values if v > median)
    unit_medians.sort(key=lambda r: -r[3])

    out.update({
        "Priced_Records": priced,
        "Spend": round(spend, 2),
        "Price_Low": round(low, 4),
        "Price_High": round(high, 4),
        "Price_Median": round(median, 4),
        "Price_Spread": round(high / low, 2) if low else None,
        "Savings": round(spend - low * priced, 2),
        "Excess": round(sum(v - median for v in values if v > median), 2),
        "Unit_Medians": "|".join(f"{n}:{v}:{c}:{r}" for n, v, c, r in unit_medians),
        "Outlier_Unit": worst_unit,
        "Outlier_Ratio": round(worst_ratio, 3) if worst_unit else None,
        "Outlier_Excess": round(worst_excess, 2) if worst_unit else None,
    })
    return out


def _cluster_row(frame: pd.DataFrame, family_id: str, code: str, golden: str,
                 split_from: str) -> dict:
    scores = [float(s) for s in frame["Semantic_Similarity"]]
    mean_score = sum(scores) / len(scores)

    counts: dict[str, int] = {}
    for unit in frame["CPSE"].astype(str):
        if unit:
            counts[unit] = counts.get(unit, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    row = {
        "Family_ID": family_id,
        "Canonical_Code": code,
        "Golden_Record": golden,
        "Members": int(len(frame)),
        "Units": len(ordered),
        "Unit_List": "|".join(name for name, _ in ordered),
        "Unit_Counts": "|".join(f"{name}:{count}" for name, count in ordered),
        "Mean_Similarity": round(mean_score, 4),
        "Min_Similarity": round(min(scores), 4),
        "Status": _capped(status_for(mean_score)),
        "Split_From": split_from,
    }
    row.update(_price_block(frame))
    return row


def _next_family_number(clusters: pd.DataFrame) -> int:
    highest = 0
    for value in clusters["Family_ID"].astype(str):
        tail = value.rsplit("_", 1)[-1]
        if tail.isdigit():
            highest = max(highest, int(tail))
    return highest + 1


def apply_split(result, family_id: str, encode, force: bool = False,
                by: str = "") -> dict:
    """Cut the family at its widest seam. Mutates `result` in place.

    `encode` takes a list of strings and returns L2-normalized vectors, so the
    breakaway rows are scored against their OWN new golden record rather than
    inheriting a number that was measured against a record they no longer
    belong to. Carrying the old similarity over would have been the same class
    of dishonesty as the toast this replaces."""
    plan = plan_split(result, family_id)
    if plan["weak"] and not force:
        raise SplitError(plan["why"])

    keep_frame, break_frame, below, above = split_frames(result, family_id)
    keep_idx = list(keep_frame.index)
    break_idx = list(break_frame.index)

    records = result.records
    clusters = result.clusters
    if "Split_From" not in clusters.columns:
        clusters["Split_From"] = ""

    # Captured verbatim before anything moves. Rebuilding this row from the
    # frames afterwards would be a re-derivation that has to agree exactly with
    # what the run already computed; keeping the row itself cannot disagree.
    origin_row_before = {k: v for k, v in
                         clusters[clusters["Family_ID"] == family_id].iloc[0].items()}

    number = _next_family_number(clusters)
    new_family = f"FAM_{number:05d}"
    new_code = f"MTL-{number:07d}"
    new_golden = _golden_of(break_frame)

    # ---- rescore the breakaway rows against their new golden record -------
    texts = list(dict.fromkeys(
        [new_golden] + [str(t) for t in break_frame["Normalized_Description"]]
    ))
    vectors = encode(texts)
    lookup = {text: index for index, text in enumerate(texts)}

    def _cos(text: str) -> float:
        a, b = vectors[lookup[text]], vectors[lookup[new_golden]]
        dot = sum(float(x) * float(y) for x, y in zip(a, b))
        na = math.sqrt(sum(float(x) * float(x) for x in a)) or 1.0
        nb = math.sqrt(sum(float(y) * float(y) for y in b)) or 1.0
        return max(-1.0, min(1.0, dot / (na * nb)))

    new_scores = [round(_cos(str(t)), 4) for t in break_frame["Normalized_Description"]]

    records.loc[break_idx, "Family_ID"] = new_family
    records.loc[break_idx, "Canonical_Code"] = new_code
    records.loc[break_idx, "Golden_Record"] = new_golden
    records.loc[break_idx, "Semantic_Similarity"] = new_scores
    records.loc[break_idx, "Confidence"] = [_confidence_for(s) for s in new_scores]

    # ---- rebuild both cluster rows ---------------------------------------
    origin_code = str(clusters.loc[clusters["Family_ID"] == family_id, "Canonical_Code"].iloc[0])
    kept_golden = str(clusters.loc[clusters["Family_ID"] == family_id, "Golden_Record"].iloc[0])

    kept_row = _cluster_row(
        records.loc[keep_idx], family_id, origin_code, kept_golden, split_from=""
    )
    new_row = _cluster_row(
        records.loc[break_idx], new_family, new_code, new_golden, split_from=origin_code
    )

    columns = list(clusters.columns)
    position = clusters.index[clusters["Family_ID"] == family_id][0]
    for key, value in kept_row.items():
        if key in columns:
            clusters.at[position, key] = value

    result.clusters = pd.concat(
        [clusters, pd.DataFrame([{c: new_row.get(c) for c in columns}])],
        ignore_index=True,
    ).sort_values("Family_ID").reset_index(drop=True)

    result.records = records.sort_values(
        by=["Family_ID", "Semantic_Similarity"], ascending=[True, False]
    )

    _refresh_stats(result)

    # The log is written here, after the frames have actually moved, so a
    # decision cannot appear in the audit trail without having happened.
    decision_log.record(
        result, "split", by=by,
        family_id=family_id, code=origin_code,
        related_id=new_family, related_code=new_code,
        affected=len(break_idx),
        summary=(f"{len(break_idx)} of {len(break_idx) + len(keep_idx)} records "
                 f"separated from {origin_code} into {new_code}; cut at the gap "
                 f"between {above:.2f} and {below:.2f}"),
        restore={
            "indices": [int(i) for i in break_idx],
            "family_id": family_id,
            "code": origin_code,
            "golden": kept_golden,
            "scores": [float(s) for s in break_frame["Semantic_Similarity"]],
            "created_family_id": new_family,
            "origin_row": origin_row_before,
        },
        detail={"new_family_id": new_family, "new_code": new_code,
                "kept_records": len(keep_idx), "moved_records": len(break_idx),
                "cut_above": round(above, 4), "cut_below": round(below, 4)},
    )

    return {
        "kept": {"family_id": family_id, "code": origin_code,
                 "records": kept_row["Members"], "status": kept_row["Status"],
                 "score": kept_row["Mean_Similarity"], "desc": kept_golden},
        "created": {"family_id": new_family, "code": new_code,
                    "records": new_row["Members"], "status": new_row["Status"],
                    "score": new_row["Mean_Similarity"], "desc": new_golden},
        "cut": {"below": round(below, 4), "above": round(above, 4)},
    }


def approve_family(result, family_id: str, by: str = "") -> dict:
    """Record that a reviewer confirmed this family's mapping.

    Approval is a human overlay, not a rescoring. The similarity band stays
    exactly what the matcher measured — inflating a 0.71 family to "resolved"
    because someone clicked a button would corrupt the one number on the page
    that is supposed to be the model's own opinion. What changes is that the
    family is no longer WAITING on anybody, so it leaves the review queue and
    is labelled as approved wherever its status is shown.

    That is also what makes the queue a queue: without a way to take something
    out of it, it is a list that only ever grows."""
    clusters = result.clusters
    if "Approved" not in clusters.columns:
        clusters["Approved"] = False
    match = clusters.index[clusters["Family_ID"] == family_id]
    if not len(match):
        raise SplitError(f"Family {family_id} is not in this run.")
    position = match[0]
    if bool(clusters.at[position, "Approved"]):
        raise SplitError(
            f"{clusters.at[position, 'Canonical_Code']} was already approved in this session."
        )
    clusters.at[position, "Approved"] = True
    _refresh_stats(result)
    decision_log.record(
        result, "approve", by=by,
        family_id=family_id,
        code=str(clusters.at[position, "Canonical_Code"]),
        affected=int(clusters.at[position, "Members"]),
        summary=(f"{clusters.at[position, 'Canonical_Code']} confirmed by a "
                 f"reviewer; {int(clusters.at[position, 'Members'])} records "
                 f"signed off at similarity "
                 f"{float(clusters.at[position, 'Mean_Similarity']):.2f}"),
        # Approval moves no rows. It is reversible by clearing one flag, which
        # needs no per-row payload.
        restore={"flag": "Approved", "family_id": family_id, "was": False},
        detail={"score": round(float(clusters.at[position, "Mean_Similarity"]), 4),
                "status": str(clusters.at[position, "Status"])},
    )
    return {
        "family_id": family_id,
        "code": str(clusters.at[position, "Canonical_Code"]),
        "status": str(clusters.at[position, "Status"]),
        "members": int(clusters.at[position, "Members"]),
    }


def plan_merge(result, left_id: str, right_id: str) -> dict:
    """What merging two canonical records would do, before it does it.

    The matcher can only group what looks alike. Two records that describe one
    part in different vocabularies — GSKT SPRL WND 4IN and SPIRAL WOUND GASKET
    100NB — never come together, because 4 inch and 100NB share no characters
    even though they are the same bore. Nothing in the run can see that; a
    person can, and now the AI pass can propose it.

    So this is the third verb, and it is the only one that makes the taxonomy
    SMALLER. It is also the most destructive: two codes become one, and the
    absorbed code stops existing. Which is exactly why it is shown in full
    first — both families, both record counts, which code survives, and what
    the surviving description will be."""
    clusters = result.clusters
    left = clusters[clusters["Family_ID"] == left_id]
    right = clusters[clusters["Family_ID"] == right_id]
    if left_id == right_id:
        raise SplitError("A family cannot be merged with itself.")
    if left.empty or right.empty:
        raise SplitError("One of those families is no longer in this run.")
    left, right = left.iloc[0], right.iloc[0]

    records = result.records
    left_rows = records[records["Family_ID"] == left_id]
    right_rows = records[records["Family_ID"] == right_id]

    # The larger family keeps its code. Retiring the code that more records
    # already point at would mean rewriting more history than necessary.
    if int(right.Members) > int(left.Members):
        left, right = right, left
        left_id, right_id = right_id, left_id
        left_rows, right_rows = right_rows, left_rows

    combined = pd.concat([left_rows, right_rows])
    golden = _golden_of(combined)

    units: dict[str, int] = {}
    for unit in combined["CPSE"].astype(str):
        if unit:
            units[unit] = units.get(unit, 0) + 1

    return {
        "keeps": {
            "family_id": str(left.Family_ID), "code": str(left.Canonical_Code),
            "golden": str(left.Golden_Record), "records": int(left.Members),
            "score": float(left.Mean_Similarity),
            "spellings": _spellings(left_rows),
        },
        "absorbed": {
            "family_id": str(right.Family_ID), "code": str(right.Canonical_Code),
            "golden": str(right.Golden_Record), "records": int(right.Members),
            "score": float(right.Mean_Similarity),
            "spellings": _spellings(right_rows),
        },
        "result": {
            "code": str(left.Canonical_Code),
            "golden": golden,
            # The pipeline's own rule picks the longest normalized description,
            # so the surviving record can come from the family being absorbed.
            # That reads like a mistake unless it is said out loud, because the
            # code beside it belongs to the other family.
            "golden_from": (
                "keeps" if golden == str(left.Golden_Record)
                else "absorbed" if golden == str(right.Golden_Record)
                else "members"
            ),
            "records": int(len(combined)),
            "units": len(units),
            "spellings": int(len(_spellings(combined, limit=10_000))),
        },
        "retires": str(right.Canonical_Code),
    }


def apply_merge(result, left_id: str, right_id: str, encode, by: str = "") -> dict:
    """Fold one canonical record into another. Mutates `result` in place.

    The absorbed rows are re-scored against the surviving golden record rather
    than keeping a number measured against a record that no longer exists —
    the same rule the split follows, for the same reason."""
    plan = plan_merge(result, left_id, right_id)
    keeps, absorbed = plan["keeps"], plan["absorbed"]

    records, clusters = result.records, result.clusters
    if "Split_From" not in clusters.columns:
        clusters["Split_From"] = ""
    if "Approved" not in clusters.columns:
        clusters["Approved"] = False

    keep_id, drop_id = keeps["family_id"], absorbed["family_id"]
    moving = list(records.index[records["Family_ID"] == drop_id])
    golden = plan["result"]["golden"]

    # Captured BEFORE anything moves. The merge rewrites the family, the code,
    # the golden record and the score of every surviving member, so a log that
    # wanted to describe — or one day reverse — this decision has to read those
    # values while they still exist.
    surviving_before = list(records.index[records["Family_ID"] == keep_id])
    prior = {
        "moved_indices": [int(i) for i in moving],
        "kept_indices": [int(i) for i in surviving_before],
        "moved_family_id": drop_id,
        "moved_code": absorbed["code"],
        "moved_golden": absorbed["golden"],
        "kept_golden": keeps["golden"],
        "moved_scores": [float(v) for v in
                         records.loc[moving, "Semantic_Similarity"]],
        "kept_scores": [float(v) for v in
                        records.loc[surviving_before, "Semantic_Similarity"]],
        "cluster_row": {k: v for k, v in
                        clusters[clusters["Family_ID"] == drop_id].iloc[0].items()},
        "kept_row": {k: v for k, v in
                     clusters[clusters["Family_ID"] == keep_id].iloc[0].items()},
    }

    records.loc[moving, "Family_ID"] = keep_id
    records.loc[moving, "Canonical_Code"] = keeps["code"]

    # Every row in the surviving family is re-scored against the new golden
    # record: the merge may have changed which spelling is longest.
    members = records[records["Family_ID"] == keep_id]
    texts = list(dict.fromkeys([golden] + [str(t) for t in members["Normalized_Description"]]))
    vectors = encode(texts)
    lookup = {text: index for index, text in enumerate(texts)}

    def cos(text: str) -> float:
        a, b = vectors[lookup[text]], vectors[lookup[golden]]
        dot = sum(float(x) * float(y) for x, y in zip(a, b))
        na = math.sqrt(sum(float(x) * float(x) for x in a)) or 1.0
        nb = math.sqrt(sum(float(y) * float(y) for y in b)) or 1.0
        return max(-1.0, min(1.0, dot / (na * nb)))

    scores = [round(cos(str(t)), 4) for t in members["Normalized_Description"]]
    idx = list(members.index)
    records.loc[idx, "Golden_Record"] = golden
    records.loc[idx, "Semantic_Similarity"] = scores
    records.loc[idx, "Confidence"] = [_confidence_for(s) for s in scores]

    row = _cluster_row(records.loc[idx], keep_id, keeps["code"], golden,
                       split_from=str(clusters.loc[
                           clusters["Family_ID"] == keep_id, "Split_From"].iloc[0] or ""))

    columns = list(clusters.columns)
    position = clusters.index[clusters["Family_ID"] == keep_id][0]
    for key, value in row.items():
        if key in columns:
            clusters.at[position, key] = value
    # A merged family is a decision waiting to be confirmed, like a split one.
    clusters.at[position, "Approved"] = False

    result.clusters = clusters[clusters["Family_ID"] != drop_id].reset_index(drop=True)
    result.records = records.sort_values(
        by=["Family_ID", "Semantic_Similarity"], ascending=[True, False])

    _refresh_stats(result)

    decision_log.record(
        result, "merge", by=by,
        family_id=keep_id, code=keeps["code"],
        related_id=drop_id, related_code=absorbed["code"],
        affected=len(moving),
        summary=(f"{absorbed['code']} merged into {keeps['code']} and retired; "
                 f"{len(moving)} records re-pointed, {row['Members']} now under "
                 f"{keeps['code']}"),
        restore={
            "indices": prior["moved_indices"] + prior["kept_indices"],
            **prior,
        },
        detail={"retired_code": absorbed["code"],
                "retired_family_id": drop_id,
                "records_moved": len(moving),
                "records_after": int(row["Members"]),
                "golden_from": plan["result"].get("golden_from", "")},
    )

    return {
        "kept": {"family_id": keep_id, "code": keeps["code"], "golden": golden,
                 "records": row["Members"], "status": row["Status"],
                 "score": row["Mean_Similarity"]},
        "retired": {"family_id": drop_id, "code": absorbed["code"],
                    "records": absorbed["records"]},
    }


def plan_approve(result, family_id: str) -> dict:
    """What confirming this family commits to.

    Approve used to act with no preview at all, which made it the one action
    in the app that asked for trust without showing its working — the opposite
    of everything around it. It is not destructive, but it IS a signature: it
    says these records are one part and takes the family out of the queue."""
    row = result.clusters[result.clusters["Family_ID"] == family_id]
    if row.empty:
        raise SplitError(f"Family {family_id} is not in this run.")
    row = row.iloc[0]
    if bool(getattr(row, "Approved", False)):
        raise SplitError(f"{row.Canonical_Code} was already approved in this session.")

    members = result.records[result.records["Family_ID"] == family_id]
    spellings = _spellings(members)
    floor = threshold_manifest()["auto_merge"]
    score = float(row.Mean_Similarity)

    concerns = []
    if score < floor:
        concerns.append(
            "This family scores %.2f, below the %.2f auto-merge floor — the matcher "
            "was not confident these are the same part." % (score, floor))
    if len(spellings) > 6:
        concerns.append(
            "%d distinct spellings resolve to this record. Worth scanning them before "
            "signing them off as one item." % len(spellings))

    return {
        "family_id": family_id,
        "code": str(row.Canonical_Code),
        "golden": str(row.Golden_Record),
        "records": int(row.Members),
        "score": score,
        "status": str(row.Status),
        "units": len(_unit_names(members)),
        "spellings": spellings,
        "more": max(0, len(_spellings(members, limit=10_000)) - len(spellings)),
        "concerns": concerns,
    }


def _unit_names(frame) -> list[str]:
    return sorted({str(u) for u in frame["CPSE"] if str(u)})


def _refresh_stats(result) -> None:
    """Every counter the page shows, recomputed from the tables themselves.

    The KPI strip reads `stats`, the distribution bar reads `clusters`, and the
    review rail reads a third thing. Leaving any of them stale is how a split
    ends up "in the review queue" on one part of the screen and absent from
    another — which is the exact bug this feature was reported for."""
    clusters = result.clusters
    records = result.records
    families = int(len(clusters))
    result.stats["families"] = families
    result.stats["records"] = int(len(records))
    result.stats["collapsed"] = int(len(records) - families)
    result.stats["fan_in"] = round(len(records) / families, 2) if families else 0.0
    result.stats["status_counts"] = clusters["Status"].value_counts().to_dict()
    result.stats["review_families"] = int(pending(clusters).sum())


def pending(clusters) -> "pd.Series":
    """Families still waiting on a person: in a review band and not approved.

    One definition, read by the queue, the rail count and the KPI, so the three
    of them cannot tell the user different numbers."""
    waiting = clusters["Status"].isin(["review", "conflict"])
    if "Approved" in clusters.columns:
        waiting = waiting & ~clusters["Approved"].fillna(False).astype(bool)
    return waiting


# ==========================================================================
# UNDO
#
# Merge retires a code. Split invents one. Approve signs something off. All
# three were, until now, one-way: the only route back from a wrong click was
# to re-run the whole file, which on fifty thousand rows is a minute of
# waiting and the loss of every other decision made since.
#
# That is the wrong shape for a review tool. A person checking a machine's
# work has to be able to try something and take it back, or they will not try
# anything — and a reviewer who will not experiment reads the score and clicks
# approve, which is the outcome this whole application exists to avoid.
#
# Undo reads the decision log and nothing else. Each entry already carries the
# rows it moved and the values they held, captured inside the verb before it
# changed anything, so reversal is a restore rather than a recomputation.
# Nothing is re-derived, so nothing can disagree with what the run had before.
#
# Two rules, both deliberate:
#
#   Last in, first out. Decisions compound — a split creates a family a later
#   merge can absorb — so reversing out of order would restore rows into a
#   family that no longer exists. Undo takes back the last thing you did,
#   which is what undo means everywhere else.
#
#   An undone decision stays in the log, marked. Deleting it would make the
#   audit trail lie by omission: "merged, then unmerged" and "never touched"
#   are different histories and only one of them happened.
# ==========================================================================

def _restore_cluster_row(clusters: pd.DataFrame, row: dict) -> pd.DataFrame:
    """Put a whole cluster row back, whether or not it is still there.

    Values are coerced to the column's current dtype before being written.
    A row captured before `Approved` existed carries no value for it, and
    pandas refuses to put NaN into a bool column — which surfaced as
    "Invalid value 'nan' for dtype 'bool'" from an undo that had otherwise
    worked. A missing flag means False, which is what it meant before the
    column existed."""
    columns = list(clusters.columns)
    values = {}
    for column in columns:
        value = row.get(column)
        if str(clusters[column].dtype) == "bool":
            value = bool(value) if value == value and value is not None else False
        values[column] = value
    match = clusters.index[clusters["Family_ID"] == row["Family_ID"]]
    if len(match):
        for key, value in values.items():
            clusters.at[match[0], key] = value
        return clusters
    return pd.concat([clusters, pd.DataFrame([values])], ignore_index=True)


def _restore_rows(records: pd.DataFrame, indices, *, family_id: str, code: str,
                  golden: str, scores) -> None:
    """Put a set of source rows back where they were, at the scores they had."""
    idx = [i for i in indices if i in records.index]
    if not idx:
        return
    records.loc[idx, "Family_ID"] = family_id
    records.loc[idx, "Canonical_Code"] = code
    records.loc[idx, "Golden_Record"] = golden
    if scores is not None and len(scores) == len(indices):
        keep = [s for i, s in zip(indices, scores) if i in records.index]
        records.loc[idx, "Semantic_Similarity"] = keep
        records.loc[idx, "Confidence"] = [_confidence_for(float(s)) for s in keep]


def undo_last(result, by: str = "") -> dict:
    """Reverse the most recent decision. Mutates `result` in place."""
    entry = decision_log.undoable(result)
    if entry is None:
        log = decision_log.log_of(result)
        live = [e for e in log if not e.get("undone")]
        if not live:
            raise SplitError(
                "There is nothing to undo — no decision in this run is still standing."
            )
        raise SplitError(
            f"Decision {live[-1]['seq']} moved more rows than this session keeps "
            "restore data for, so it cannot be reversed. Re-run the file to start "
            "from the matcher's own grouping."
        )

    verb = entry["verb"]
    restore = entry["restore"]
    records, clusters = result.records, result.clusters

    if verb == "approve":
        match = clusters.index[clusters["Family_ID"] == restore["family_id"]]
        if not len(match):
            raise SplitError(
                f"{entry['code']} is no longer in this run, so its approval "
                "cannot be taken back."
            )
        clusters.at[match[0], "Approved"] = bool(restore.get("was", False))
        undone = {"restored": [entry["code"]], "removed": []}

    elif verb == "split":
        created = restore["created_family_id"]
        # Put the breakaway rows back under the family they came from, at the
        # scores they were measured at against ITS golden record.
        _restore_rows(records, restore["indices"],
                      family_id=restore["family_id"], code=restore["code"],
                      golden=restore["golden"], scores=restore.get("scores"))
        result.clusters = _restore_cluster_row(clusters, restore["origin_row"])
        result.clusters = result.clusters[
            result.clusters["Family_ID"] != created].reset_index(drop=True)
        undone = {"restored": [entry["code"]], "removed": [entry["related_code"]]}

    elif verb == "merge":
        # The absorbed rows go back to the family that was retired, and the
        # rows that never moved go back to the score they held against the
        # golden record they had before the merge changed it.
        _restore_rows(records, restore["moved_indices"],
                      family_id=restore["moved_family_id"],
                      code=restore["moved_code"],
                      golden=restore["moved_golden"],
                      scores=restore.get("moved_scores"))
        _restore_rows(records, restore["kept_indices"],
                      family_id=entry["family_id"], code=entry["code"],
                      golden=restore["kept_golden"],
                      scores=restore.get("kept_scores"))
        clusters = _restore_cluster_row(clusters, restore["kept_row"])
        result.clusters = _restore_cluster_row(clusters, restore["cluster_row"])
        undone = {"restored": [entry["related_code"]], "removed": []}

    else:                                    # pragma: no cover - guarded above
        raise SplitError(f"Cannot reverse a {verb}.")

    result.clusters = result.clusters.sort_values("Family_ID").reset_index(drop=True)
    result.records = result.records.sort_values(
        by=["Family_ID", "Semantic_Similarity"], ascending=[True, False])
    _refresh_stats(result)
    decision_log.mark_undone(entry, by=by)

    return {
        "seq": entry["seq"],
        "verb": verb,
        "code": entry["code"],
        "summary": entry["summary"],
        "families": int(len(result.clusters)),
        "review_families": int(result.stats.get("review_families", 0)),
        **undone,
    }
