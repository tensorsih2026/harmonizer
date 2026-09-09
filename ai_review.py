"""Where a language model catches what cosine similarity cannot.

The matcher makes two kinds of mistake, and neither is visible to the thing
that made them:

  FALSE MERGE — rows grouped that are different parts.
      BEARING 6205 2RS and BEARING 6205 ZZ differ by two characters and score
      about 0.97. One is rubber-sealed, the other metal-shielded; they are
      different products with different applications. No amount of embedding
      quality fixes this, because the strings really are nearly identical. It
      needs someone who knows what 2RS means.

  FALSE SPLIT — the same part sitting in two families.
      GSKT SPRL WND 4IN and SPIRAL WOUND GASKET 100NB are one gasket: 4 inch
      IS 100NB. They share almost no tokens, so the embedding never brought
      them together and nothing in the run says they belong.

Both are asked as questions about TEXT ONLY. The model never sees prices, unit
names, legacy codes or the similarity scores — it is not being asked to agree
with the pipeline, and telling it what the pipeline decided would only invite
it to.

WHAT COMES OUT IS A FLAG, NEVER A CHANGE
----------------------------------------
Nothing here edits a family. Every finding lands in the review queue next to
the Approve and Split buttons a person already uses, carrying the model's own
reason in its own words. AI proposes; the officer disposes. That is the only
arrangement defensible for a material master, and it is also the honest one:
this pass is a second opinion from a system that is confidently wrong
sometimes, which is exactly what a review queue is for.
"""

from __future__ import annotations

from gemini import AIUnavailable, generate_json

FAMILIES_PER_CALL = 6
PAIRS_PER_CALL = 12

# How far below the clustering threshold to look for pairs that should have
# merged. Above 0.75 they already did; below 0.55 they share nothing and the
# question is not worth asking.
NEAR_MISS_LOW = 0.55
NEAR_MISS_HIGH = 0.75

MERGE_PROMPT = """You are auditing grouped material master records for Indian public sector oil and gas companies.

Each group below was assembled automatically by text similarity. For each group, decide whether all the descriptions denote THE SAME physical item that a storekeeper would issue from one bin.

Pay particular attention to differences that change the part while barely changing the text:
- seal or shield codes (2RS, ZZ, RS, open)
- alloy grades (SS304 vs SS316, A105 vs F11)
- pressure classes and schedules (150# vs 300#, SCH 40 vs SCH 80)
- sizes and ratings (M10 vs M12, 3Cx240 vs 3Cx120, 50 HP vs 75 HP)
- valve or fitting types (ball vs gate vs globe)
- fire extinguisher agents (CO2 vs DCP vs foam)

Rules:
- Answer ONLY from the text shown. Do not guess at anything not written.
- "same": true if every description is the same item. false if the group holds more than one distinct item.
- When false, put the descriptions into "groups" — a list of lists of the given line numbers, one list per distinct item.
- "reason" must be one sentence naming the specific attribute that differs, in plain words an officer can act on.
- "confidence" 0.0 to 1.0.

Return ONLY a JSON array, one object per group, in order:
{{"id": "<the id given>", "same": true, "groups": [], "reason": "", "confidence": 0.0}}

Groups:
{groups}"""

SPLIT_PROMPT = """You are auditing an Indian public sector material master for duplicate items that were missed.

Each pair below is two SEPARATE canonical records. Decide whether they are in fact the same physical item written two ways.

Remember that these conventions describe the same thing:
- inch and nominal bore (4 IN = 100 NB = DN100)
- abbreviated and expanded words (GSKT = GASKET, SPRL WND = SPIRAL WOUND)
- HP and kW for the same machine, if both are stated for the same rating
- reordered attributes (VALVE BALL 2IN and 2 INCH BALL VALVE)

But these are NOT the same item, however similar the words:
- different sizes, ratings, classes, schedules or alloy grades
- different types of the same family of equipment

Rules:
- Answer ONLY from the text shown.
- "same": true only if you are confident they are one item.
- "reason": one sentence saying why, naming the convention or the difference.
- "confidence" 0.0 to 1.0.

Return ONLY a JSON array, one object per pair, in order:
{{"id": "<the id given>", "same": false, "reason": "", "confidence": 0.0}}

Pairs:
{pairs}"""


def _merge_candidates(result, limit: int) -> list[dict]:
    """Families worth asking about: more than one distinct spelling.

    A family written one way cannot be a false merge of two things, so asking
    about it spends a call to be told what is already known."""
    records = result.records
    clusters = result.clusters.sort_values("Members", ascending=False)

    out = []
    for row in clusters.itertuples(index=False):
        fid = str(row.Family_ID)
        members = records[records["Family_ID"] == fid]
        seen: list[str] = []
        for text in members["Original_Description"].astype(str):
            collapsed = " ".join(text.split())
            if collapsed and collapsed not in seen:
                seen.append(collapsed)
            if len(seen) >= 8:
                break
        if len(seen) < 2:
            continue
        out.append({
            "family_id": fid,
            "code": str(row.Canonical_Code),
            "golden": str(row.Golden_Record),
            "lines": seen,
        })
        if len(out) >= limit:
            break
    return out


def _pair_candidates(result, encode, limit: int) -> list[dict]:
    """Canonical records that ALMOST merged.

    Everything above the threshold is already one family, so the interesting
    band is just below it — pairs the matcher looked at and rejected."""
    clusters = result.clusters
    if len(clusters) < 2:
        return []

    goldens = [str(g) for g in clusters["Golden_Record"]]
    codes = [str(c) for c in clusters["Canonical_Code"]]
    ids = [str(f) for f in clusters["Family_ID"]]
    vectors = encode(goldens)

    def cosine(a, b) -> float:
        dot = na = nb = 0.0
        for x, y in zip(a, b):
            x = float(x); y = float(y)
            dot += x * y; na += x * x; nb += y * y
        if na <= 0 or nb <= 0:
            return 0.0
        return dot / ((na ** 0.5) * (nb ** 0.5))

    scored = []
    n = len(goldens)
    for i in range(n):
        for j in range(i + 1, n):
            score = cosine(vectors[i], vectors[j])
            if NEAR_MISS_LOW <= score < NEAR_MISS_HIGH:
                scored.append((score, i, j))
    scored.sort(reverse=True)

    return [
        {
            "left_id": ids[i], "right_id": ids[j],
            "left_code": codes[i], "right_code": codes[j],
            "left": goldens[i], "right": goldens[j],
            "score": round(score, 4),
        }
        for score, i, j in scored[:limit]
    ]


def review(result, encode, transport=None, report=None,
           family_limit: int = 60, pair_limit: int = 36) -> dict:
    """Both directions of error, as flags. Never raises, never edits."""
    findings: dict[str, list[dict]] = {}
    merges: list[dict] = []
    errors: list[str] = []
    calls = 0

    def add(family_id: str, flag: dict) -> None:
        findings.setdefault(str(family_id), []).append(flag)

    # ---- false merges ----------------------------------------------------
    candidates = _merge_candidates(result, family_limit)
    batches = [candidates[i : i + FAMILIES_PER_CALL]
               for i in range(0, len(candidates), FAMILIES_PER_CALL)]

    for index, batch in enumerate(batches):
        if report is not None:
            report(0.6 * index / max(1, len(batches)),
                   f"Checking {len(candidates):,} families for wrong merges")
        blocks = []
        for entry in batch:
            lines = "\n".join(f"    {k + 1}. {text}" for k, text in enumerate(entry["lines"]))
            blocks.append(f'  id: {entry["family_id"]}\n{lines}')
        try:
            answer = generate_json(MERGE_PROMPT.format(groups="\n\n".join(blocks)),
                                   transport=transport)
            calls += 1
        except AIUnavailable as exc:
            errors.append(str(exc))
            break

        if not isinstance(answer, list):
            continue
        by_id = {str(r.get("id")): r for r in answer if isinstance(r, dict)}
        for entry in batch:
            row = by_id.get(entry["family_id"])
            if not row or row.get("same", True):
                continue
            groups = row.get("groups") or []
            add(entry["family_id"], {
                "kind": "wrong merge",
                "note": str(row.get("reason", ""))[:400],
                "confidence": _confidence(row),
                "suggested_groups": [
                    [entry["lines"][i - 1] for i in group
                     if isinstance(i, int) and 1 <= i <= len(entry["lines"])]
                    for group in groups if isinstance(group, list)
                ],
                "source": "gemini",
            })

    # ---- false splits ----------------------------------------------------
    pairs = _pair_candidates(result, encode, pair_limit)
    pair_batches = [pairs[i : i + PAIRS_PER_CALL]
                    for i in range(0, len(pairs), PAIRS_PER_CALL)]

    for index, batch in enumerate(pair_batches):
        if report is not None:
            report(0.6 + 0.4 * index / max(1, len(pair_batches)),
                   f"Checking {len(pairs):,} near-miss pairs for missed duplicates")
        blocks = []
        for k, pair in enumerate(batch):
            pid = f'{pair["left_id"]}|{pair["right_id"]}'
            blocks.append(f'  id: {pid}\n    A. {pair["left"]}\n    B. {pair["right"]}')
        try:
            answer = generate_json(SPLIT_PROMPT.format(pairs="\n\n".join(blocks)),
                                   transport=transport)
            calls += 1
        except AIUnavailable as exc:
            errors.append(str(exc))
            break

        if not isinstance(answer, list):
            continue
        by_id = {str(r.get("id")): r for r in answer if isinstance(r, dict)}
        for pair in batch:
            pid = f'{pair["left_id"]}|{pair["right_id"]}'
            row = by_id.get(pid)
            if not row or not row.get("same", False):
                continue
            note = str(row.get("reason", ""))[:400]
            confidence = _confidence(row)
            merges.append({
                "left": pair["left_code"], "right": pair["right_code"],
                "left_id": pair["left_id"], "right_id": pair["right_id"],
                "left_text": pair["left"], "right_text": pair["right"],
                "score": pair["score"], "note": note, "confidence": confidence,
            })
            for side, other in ((pair["left_id"], pair["right_code"]),
                                (pair["right_id"], pair["left_code"])):
                add(side, {
                    "kind": "possible duplicate",
                    "note": f"Looks like the same item as {other}. {note}",
                    "confidence": confidence,
                    "suggested_groups": [],
                    "source": "gemini",
                })

    if report is not None:
        report(1.0, f"{sum(len(v) for v in findings.values()):,} flags raised")

    return {
        "flags": findings,
        "merge_suggestions": merges,
        "checked_families": len(candidates),
        "checked_pairs": len(pairs),
        "errors": errors,
        "calls": calls,
    }


def _confidence(row: dict) -> float:
    try:
        return max(0.0, min(1.0, float(row.get("confidence", 0.0))))
    except (TypeError, ValueError):
        return 0.0
