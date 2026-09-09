"""Run one description through the pipeline, step by step, and say what happens.

Every other view in this app shows a finished result. This one shows the
machine working, on a string the viewer chose — including a string that was
never in the upload, which is the only way to demonstrate that the matcher is
reasoning rather than replaying a lookup table.

Nothing here is a re-implementation. It calls the same `normalize_text`, the
same `NORMALIZATION_RULES`, the same model and the same `SIMILARITY_THRESHOLD`
the pipeline used to build the run being looked at. If this disagrees with the
result, the demo is lying, so it is wired to the originals on purpose.

Each family is represented by its golden record's vector. Those are encoded
once per run and cached, because a viewer typing in a box should not pay for
80 encodings per keystroke.
"""

from __future__ import annotations

import math
import re

from harmonizer import (
    NORMALIZATION_RULES,
    SIMILARITY_THRESHOLD,
    HIGH_CONFIDENCE,
    MEDIUM_CONFIDENCE,
    normalize_text,
    status_for,
)

MAX_INPUT = 200
TOP_N = 6

# The tidy-up the pipeline applies after the abbreviation rules, kept here in
# the same order so the trace matches what actually ran.
FINAL_RULES = [
    (r"[,;:/]+", " ", "separators folded to spaces"),
    (r"\s+", " ", "runs of whitespace collapsed"),
]

RULE_NOTES = {
    r"\bS\.?S\.?\b": "stainless steel, however it was abbreviated",
    r"\bC\.?S\.?\b": "carbon steel",
    r"\bM\.?S\.?\b": "mild steel",
    r"\bDIA\b": "diameter",
    r"\bAPPX\b": "approximate",
}


def trace_normalization(text: str) -> tuple[str, list[dict]]:
    """The exact sequence the pipeline performs, with what each step changed."""
    steps: list[dict] = []

    raw = str(text)
    upper = normalize_text(raw)
    if upper != raw:
        steps.append({
            "rule": "case and whitespace",
            "note": "upper-cased and trimmed, so spelling differences in case stop mattering",
            "before": raw, "after": upper, "changed": True,
        })

    current = upper
    for pattern, replacement in NORMALIZATION_RULES.items():
        after = re.sub(pattern, replacement, current)
        if after != current:
            steps.append({
                "rule": replacement,
                "note": RULE_NOTES.get(pattern, "expanded to its full form"),
                "before": current, "after": after, "changed": True,
            })
        current = after

    for pattern, replacement, note in FINAL_RULES:
        after = re.sub(pattern, replacement, current)
        if after != current:
            steps.append({
                "rule": note, "note": "so punctuation does not split two identical parts",
                "before": current, "after": after, "changed": True,
            })
        current = after

    current = current.strip()
    return current, steps


def _cos(a, b) -> float:
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        x = float(x); y = float(y)
        dot += x * y; na += x * x; nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return max(-1.0, min(1.0, dot / (math.sqrt(na) * math.sqrt(nb))))


def family_vectors(result, encode):
    """Golden-record vectors for the run, encoded once and kept on the result."""
    cache = getattr(result, "_sim_vectors", None)
    if cache is not None and cache.get("n") == len(result.clusters):
        return cache

    rows = result.clusters
    texts = [str(t) for t in rows["Golden_Record"]]
    vectors = encode(texts) if texts else []
    cache = {
        "n": len(rows),
        "vectors": vectors,
        "meta": [
            {"family_id": str(r.Family_ID), "code": str(r.Canonical_Code),
             "golden": str(r.Golden_Record), "members": int(r.Members),
             "status": str(r.Status)}
            for r in rows.itertuples(index=False)
        ],
    }
    try:
        result._sim_vectors = cache
    except Exception:
        pass
    return cache


def simulate(result, text: str, encode) -> dict:
    """Four stages, the same four the pipeline reports while it runs."""
    raw = str(text or "").strip()[:MAX_INPUT]
    if not raw:
        return {"ok": False, "reason": "Type a material description to run it through the pipeline."}

    normalized, steps = trace_normalization(raw)
    if not normalized:
        return {"ok": False,
                "reason": "Nothing survived normalization — this reads as a placeholder, "
                          "and the pipeline drops those before matching."}

    cache = family_vectors(result, encode)
    if not cache["meta"]:
        return {"ok": False, "reason": "This run has no families to match against."}

    vector = encode([normalized])[0]
    scored = []
    for meta, fam_vector in zip(cache["meta"], cache["vectors"]):
        scored.append((_cos(vector, fam_vector), meta))
    scored.sort(key=lambda pair: -pair[0])

    top = scored[:TOP_N]
    best_score, best = top[0]
    merges = best_score >= SIMILARITY_THRESHOLD

    # Was this exact string already in the upload? Saying so matters: a viewer
    # who types a description they can see on screen should be told the match
    # is a lookup, not a leap.
    seen = result.records["Normalized_Description"].astype(str)
    exact = int((seen == normalized).sum())

    confidence = ("HIGH" if best_score >= HIGH_CONFIDENCE
                  else "MEDIUM" if best_score >= MEDIUM_CONFIDENCE else "LOW")

    return {
        "ok": True,
        "input": raw,
        "normalized": normalized,
        "steps": steps,
        "unchanged": not steps,
        "vector_dims": len(vector),
        # A handful of real components, so the embedding stage shows something
        # measured rather than an animation standing in for one.
        "vector_head": [round(float(v), 4) for v in list(vector)[:16]],
        "candidates": [
            {"score": round(float(s), 4), "code": m["code"], "golden": m["golden"],
             "family_id": m["family_id"], "members": m["members"], "status": m["status"]}
            for s, m in top
        ],
        "decision": {
            "merges": bool(merges),
            "threshold": SIMILARITY_THRESHOLD,
            "score": round(float(best_score), 4),
            "confidence": confidence,
            "band": status_for(float(best_score)),
            "family_id": best["family_id"] if merges else None,
            "code": best["code"] if merges else None,
            "golden": best["golden"] if merges else None,
            "members": best["members"] if merges else None,
            "already_present": exact,
            "verdict": (
                "This description already appears in the run %d times — it resolves to the "
                "same canonical record." % exact if merges and exact else
                "Above the %.2f clustering threshold, so it would be absorbed into this "
                "family rather than creating a new code." % SIMILARITY_THRESHOLD if merges else
                "Below the %.2f clustering threshold. Nothing in this run is close enough, so "
                "it would open a new canonical family of its own." % SIMILARITY_THRESHOLD
            ),
        },
    }
