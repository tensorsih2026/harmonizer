"""Measured accuracy, for whatever file you put in.

THE CONSTRAINT, STATED FIRST
----------------------------
Accuracy cannot be computed without labels. "Mean similarity 0.94" is the
embedding's confidence in its own output — the same vectors produced both the
grouping and the score, so it cannot answer "how often is this right?". Any
figure derived from the model alone would be the same circularity wearing a
different name, and this module refuses to produce one.

What it does instead is find labels wherever a file actually has them, and
score against those. Three sources, in order of strength:

  1. A LABEL COLUMN in the upload. Many real material masters already carry
     one — a verified code, a reviewed group, a de-duplication key from an
     earlier cleanup. The column is found by content and name the same way the
     description and unit columns are, so it works whatever the exporting
     system happened to call it.

  2. THE REVIEWER'S OWN DECISIONS. Every Approve says "this family is right";
     every Split says "these rows do not belong together". Those are labels,
     made by a person, on this file. Accumulated over a session they score the
     matcher on exactly the data the officer cares about.

  3. THE BUNDLED BENCHMARK, a labelled reference set shipped with the app for
     when neither of the above exists. It is one dataset, it is ours, and it is
     reported as such — never as a claim about the user's data.

If none of the three is present the answer is "not measured", in those words.

THE METRIC
----------
Pairwise precision, recall and F1 — the standard for record linkage. Every
pair of rows is either together or apart in the truth, and either together or
apart in the output:

    TP  same item, matcher grouped them        (a duplicate correctly caught)
    FP  different items, matcher grouped them  (a WRONG merge)
    FN  same item, matcher kept them apart     (a duplicate MISSED)

Precision = TP/(TP+FP): of the merges made, how many were right.
Recall    = TP/(TP+FN): of the duplicates present, how many were found.

Both are reported, never just one, because they trade against each other and
either alone can be made to look excellent by a system that is useless.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import pandas as pd

# Names that suggest a column holds the answer rather than the question.
LABEL_TOKENS = {
    "gold": 6.0, "ground truth": 6.0, "groundtruth": 6.0, "truth": 5.0,
    "label": 5.0, "verified": 4.5, "confirmed": 4.0, "approved": 3.5,
    "true group": 6.0, "true_group": 6.0, "correct": 4.0, "canonical": 3.0,
    "master code": 3.5, "parent": 2.5, "cluster": 3.0, "group": 2.5,
    "dedup": 4.5, "duplicate of": 5.0, "same as": 4.5,
}

# A label column groups rows. If nearly every value is unique it is an id, not
# a label; if there is only one value it groups nothing.
MIN_GROUPS = 2
MAX_DISTINCT_RATIO = 0.92
MIN_COVERAGE = 0.60          # share of rows that must carry a label


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(name).strip().lower()).strip()


def find_label_column(df: pd.DataFrame, exclude: set[str] | None = None) -> tuple[str | None, str]:
    """The column holding ground truth, if the file has one.

    Scored on two independent signals, like every other column this app infers:
    what it is CALLED and what it CONTAINS. Name alone would match a column
    called "Group" that holds plant codes; content alone cannot tell a label
    from any other repeated categorical. Both must agree.

    Returns (column, reason) so the UI can say why it chose — or why it found
    nothing, which is the more important message.
    """
    exclude = exclude or set()
    rows = len(df)
    if not rows:
        return None, "The file has no rows."

    best, best_score, best_reason = None, 0.0, ""
    considered = []

    for column in df.columns:
        if column in exclude:
            continue
        name = _norm(column)
        name_score = 0.0
        for token, weight in LABEL_TOKENS.items():
            if token in name:
                name_score = max(name_score, weight)
        if name_score <= 0:
            continue

        series = df[column].astype(str).str.strip()
        filled = series[(series != "") & (series.str.lower() != "nan")]
        coverage = len(filled) / rows
        if coverage < MIN_COVERAGE:
            considered.append(f"{column}: only {coverage:.0%} of rows carry a value")
            continue

        groups = filled.nunique()
        distinct_ratio = groups / max(1, len(filled))
        if groups < MIN_GROUPS:
            considered.append(f"{column}: every row has the same value, so it groups nothing")
            continue
        if distinct_ratio > MAX_DISTINCT_RATIO:
            considered.append(
                f"{column}: {distinct_ratio:.0%} of values are unique — that is an "
                f"identifier, not a grouping"
            )
            continue

        # A label that groups a few rows each is what truth looks like here.
        shape = 1.0 - abs(0.35 - distinct_ratio)
        score = name_score + shape * 2.0 + coverage
        considered.append(f"{column}: candidate, score {score:.1f}")
        if score > best_score:
            best, best_score, best_reason = column, score, (
                f"{column} holds {groups:,} distinct values over {len(filled):,} rows "
                f"({coverage:.0%} coverage), which is the shape of a grouping label."
            )

    if best is None:
        return None, (
            "No label column found. To measure accuracy on this file, add a column "
            "naming the true item each row belongs to — call it gold_group, "
            "true_group, verified_code or anything containing those words — and "
            "harmonize again."
            + (" Columns examined: " + "; ".join(considered[:4]) if considered else "")
        )
    return best, best_reason


# --------------------------------------------------------------------------
# Pairwise scoring


def _pairs(counts) -> int:
    """Pairs within groups of these sizes: sum of n(n-1)/2."""
    return sum(n * (n - 1) // 2 for n in counts)


def pairwise(truth: list, predicted: list) -> dict:
    """Precision / recall / F1 over every pair of rows.

    Counted from the contingency table rather than by enumerating pairs: a
    50,000-row file has 1.2 billion pairs, and the closed form gives the same
    answer in one pass."""
    n = len(truth)
    if n != len(predicted):
        raise ValueError("truth and predicted must be the same length")
    if n < 2:
        return {"pairs": 0, "tp": 0, "fp": 0, "fn": 0,
                "precision": None, "recall": None, "f1": None}

    joint = Counter(zip(truth, predicted))
    tp = _pairs(joint.values())
    same_truth = _pairs(Counter(truth).values())
    same_pred = _pairs(Counter(predicted).values())

    fp = same_pred - tp
    fn = same_truth - tp
    total = n * (n - 1) // 2
    tn = total - tp - fp - fn

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and (precision + recall) else
          (0.0 if precision is not None and recall is not None else None))

    return {
        "pairs": total, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "rand_index": (tp + tn) / total if total else None,
        "adjusted_rand": _adjusted_rand(joint, Counter(truth), Counter(predicted), n),
    }


def _adjusted_rand(joint, truth_counts, pred_counts, n: int):
    """Rand index corrected for chance. Zero means no better than random."""
    if n < 2:
        return None
    total = n * (n - 1) / 2
    index = sum(v * (v - 1) / 2 for v in joint.values())
    a = sum(v * (v - 1) / 2 for v in truth_counts.values())
    b = sum(v * (v - 1) / 2 for v in pred_counts.values())
    expected = a * b / total if total else 0
    maximum = (a + b) / 2
    denom = maximum - expected
    return (index - expected) / denom if denom else None


def _examples(frame: pd.DataFrame, truth_col: str, pred_col: str, desc_col: str, limit: int = 6):
    """The mistakes themselves. A metric without examples is unfalsifiable."""
    wrong_merges, missed = [], []

    for pred, group in frame.groupby(pred_col):
        labels = group[truth_col].astype(str)
        if labels.nunique() > 1:
            counts = labels.value_counts()
            wrong_merges.append({
                "predicted": str(pred),
                "true_items": int(counts.size),
                "records": int(len(group)),
                "samples": [
                    {"label": str(lbl), "text": str(txt)[:110]}
                    for lbl, txt in list(zip(labels, group[desc_col].astype(str)))[:4]
                ],
            })

    for label, group in frame.groupby(truth_col):
        preds = group[pred_col].astype(str)
        if preds.nunique() > 1:
            missed.append({
                "true_item": str(label),
                "split_into": int(preds.nunique()),
                "records": int(len(group)),
                "samples": [
                    {"predicted": str(pr), "text": str(txt)[:110]}
                    for pr, txt in list(zip(preds, group[desc_col].astype(str)))[:4]
                ],
            })

    wrong_merges.sort(key=lambda e: -e["records"])
    missed.sort(key=lambda e: -e["records"])
    return wrong_merges[:limit], missed[:limit]


def evaluate(result, label_column: str = "Gold_Label") -> dict:
    """Score a finished run against whatever labels its records carry."""
    records = result.records
    if label_column not in records.columns:
        return {"available": False,
                "reason": "This run carries no ground-truth labels."}

    frame = records[[label_column, "Family_ID", "Original_Description"]].copy()
    frame[label_column] = frame[label_column].astype(str).str.strip()
    frame = frame[(frame[label_column] != "") & (frame[label_column].str.lower() != "nan")]

    if len(frame) < 2:
        return {"available": False,
                "reason": "Fewer than two labelled rows — nothing to compare."}

    truth = frame[label_column].tolist()
    predicted = frame["Family_ID"].astype(str).tolist()
    scores = pairwise(truth, predicted)

    wrong_merges, missed = _examples(frame, label_column, "Family_ID", "Original_Description")

    true_groups = len(set(truth))
    pred_groups = len(set(predicted))

    return {
        "available": True,
        "label_column": label_column,
        "labelled_records": int(len(frame)),
        "unlabelled_records": int(len(records) - len(frame)),
        "true_items": true_groups,
        "predicted_items": pred_groups,
        "scores": {
            "precision": _round(scores["precision"]),
            "recall": _round(scores["recall"]),
            "f1": _round(scores["f1"]),
            "adjusted_rand": _round(scores["adjusted_rand"]),
        },
        "counts": {
            "pairs": scores["pairs"], "tp": scores["tp"],
            "fp": scores["fp"], "fn": scores["fn"],
        },
        "reading": _reading(scores, true_groups, pred_groups),
        "wrong_merges": wrong_merges,
        "missed_merges": missed,
    }


def _round(value):
    return None if value is None else round(float(value), 4)


def _reading(scores, true_groups, pred_groups) -> str:
    """One sentence a person can repeat out loud without misleading anyone."""
    p, r = scores["precision"], scores["recall"]
    if p is None or r is None:
        return "Too few labelled pairs to score."

    parts = [
        "Of the merges the matcher made, %.1f%% joined rows that really are the "
        "same item. Of the duplicate pairs actually present, it found %.1f%%."
        % (p * 100, r * 100)
    ]
    if pred_groups > true_groups:
        parts.append(
            "It produced %d items where the labels say %d, so it is splitting "
            "more than it should — the misses cost more than the wrong merges here."
            % (pred_groups, true_groups)
        )
    elif pred_groups < true_groups:
        parts.append(
            "It produced %d items where the labels say %d, so it is merging things "
            "the labels keep apart — check the wrong merges below before trusting "
            "any spend figure computed across them."
            % (pred_groups, true_groups)
        )
    else:
        parts.append("It produced exactly as many items as the labels describe (%d)."
                     % true_groups)
    return " ".join(parts)
