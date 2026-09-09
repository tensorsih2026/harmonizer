"""Standardization — the half of the problem statement the pipeline did not do.

PS26099 asks for "Standardization AND Harmonization". Everything built so far
is the second word: fragmented records resolved into families. The first word
was missing, and the gap is visible in the output. The golden record is chosen
by `max(cluster, key=length)` — the LONGEST normalized description in the
family. That is a representative, not a standard:

    STAINLESS STEEL. PIPE 6" SCH 40

It won because it is long. It carries a stray full stop, an inch mark, and no
consistent field order. Every family in the run is spelled a different way,
because each one inherited whichever member happened to be wordiest.

A standard is different. It has NAMED FIELDS in a FIXED ORDER, so two people
in two offices write the same part identically, and so a system can filter on
"everything in SS316" without reading prose:

    item      PIPE
    material  STAINLESS STEEL
    size      6 IN
    schedule  SCH 40
    -> PIPE, STAINLESS STEEL, 6 IN, SCH 40

Extracting those fields is what a language model is genuinely better at than
cosine similarity, because it requires knowing that SCH 40 is a wall thickness
and 6" is a bore — domain knowledge, not string distance.

WHAT THIS MODULE WILL NOT DO
----------------------------
It does not touch the harmonization. Families, similarity scores and the
mapping are computed locally and are not shown to the model. This reads a
golden record and its spellings, and returns fields. If it is switched off,
every existing number is unchanged.

It also never overwrites. The extracted standard is carried ALONGSIDE the
original golden record, both are shown, and the export keeps the original
until a reviewer accepts. An LLM rewriting a material master unattended is
exactly the thing a Ministry should refuse, and so does this.
"""

from __future__ import annotations

import hashlib
import json

from gemini import AIUnavailable, generate_json

# Families per request. One call each is 80 round trips on a small run and
# thousands on a real one; ten at a time keeps the prompt readable and the
# latency sane without asking the model to hold too much at once.
FAMILIES_PER_CALL = 10

# The fields a material master actually needs to filter and procure on. Kept
# deliberately short: a schema nobody fills in is worse than no schema.
FIELDS = ["item", "type", "material", "size", "rating", "standard"]

PROMPT = """You are standardising material master descriptions for Indian public sector oil and gas companies (ONGC, IOCL, BPCL, HPCL, GAIL and similar).

For each item below you are given the canonical description chosen by a clustering pipeline, plus the different ways the units actually wrote it. Extract structured attributes.

Rules:
- Use ONLY information present in the given text. Never infer, complete or guess a value that is not written. If a field is not stated, use null.
- Normalise units of measure to a consistent form (6", 6 IN and 150NB are the same bore — write it as the most explicit form present).
- "item" is the noun: PIPE, VALVE, BOLT, BEARING, GASKET, PUMP, CABLE.
- "type" is the qualifying variant: BALL, GATE, HEXAGONAL, DEEP GROOVE, SPIRAL WOUND, CENTRIFUGAL.
- "material" is the material of construction: STAINLESS STEEL SS316, CARBON STEEL, MILD STEEL.
- "size" is the principal dimension including its unit: 6 IN, M10 X 50 MM, 240 SQ MM.
- "rating" is a pressure/electrical/capacity rating if stated: CLASS 150, 11 KV, 50 HP, SCH 40.
- "standard" is a named specification if stated: IS 2062, API 5L, ASTM A106, OISD.
- Write every value in UPPER CASE.
- "confidence" is your confidence that these attributes are correct and complete for this item, 0.0 to 1.0.

Return ONLY a JSON array, one object per item, in the same order, shaped:
{{"id": "<the id given>", "item": null, "type": null, "material": null, "size": null, "rating": null, "standard": null, "confidence": 0.0}}

Items:
{items}"""


def _fingerprint(golden: str, spellings: list[str]) -> str:
    """Cache key. Same family text, same answer — no second call, and no
    variation between runs of the same file."""
    blob = json.dumps([golden, sorted(spellings)], ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _describe(entry: dict) -> str:
    lines = [f'  id: {entry["family_id"]}', f'  canonical: {entry["golden"]}']
    if entry["spellings"]:
        lines.append("  written as: " + " | ".join(entry["spellings"][:6]))
    return "\n".join(lines)


def canonical_description(attributes: dict) -> str:
    """The standard, assembled in one fixed order.

    This is what makes it a standard rather than a description: the same fields
    in the same order every time, so two offices writing the same part produce
    the same string, and so the result can be filtered on."""
    parts = [attributes.get(field) for field in FIELDS]
    return ", ".join(str(p).strip() for p in parts if p and str(p).strip())


def as_rows(answer) -> list[dict]:
    """The model's reply as a list of objects, whatever shape it arrived in.

    Asking for a JSON array usually gets a JSON array. Usually. It also gets
    {"items": [...]}, {"results": [...]}, a single bare object when the batch
    had one entry, and occasionally a dict keyed by id. None of those are worth
    losing an entire pass over, and the first version of this lost ALL of them
    — it checked `isinstance(answer, list)`, logged a note nobody displayed,
    and reported zero standardised with no explanation on screen."""
    if isinstance(answer, list):
        return [row for row in answer if isinstance(row, dict)]
    if isinstance(answer, dict):
        for value in answer.values():
            if isinstance(value, list) and any(isinstance(v, dict) for v in value):
                return [row for row in value if isinstance(row, dict)]
        # {"FAM_00001": {...}, "FAM_00002": {...}}
        nested = [dict(v, id=v.get("id", k)) for k, v in answer.items() if isinstance(v, dict)]
        if nested:
            return nested
        if any(field in answer for field in FIELDS):
            return [answer]
    return []


def match_rows(batch: list[dict], rows: list[dict]) -> dict[str, dict]:
    """Pair each family with the row the model meant for it.

    Ids are echoed back correctly most of the time. When they are not — a
    lower-cased id, the canonical code instead of the family id, or the id
    dropped entirely — the honest fallback is position, but ONLY when the model
    returned exactly as many rows as were asked about and in order. Guessing
    beyond that would attach one part's attributes to another part, which in a
    material master is worse than returning nothing."""
    by_key: dict[str, dict] = {}
    for row in rows:
        key = str(row.get("id", "")).strip()
        if key:
            by_key[key] = row
            by_key[key.upper()] = row

    matched: dict[str, dict] = {}
    for entry in batch:
        row = (by_key.get(entry["family_id"])
               or by_key.get(entry["family_id"].upper())
               or by_key.get(entry.get("code", ""))
               or by_key.get(str(entry.get("code", "")).upper()))
        if row is not None:
            matched[entry["family_id"]] = row

    # Position is evidence only when there is an ordered correspondence to
    # observe. With a single item and an unrecognised id there is none — the
    # model may simply have answered about something else — so that case is
    # refused rather than paired on hope.
    if not matched and len(rows) == len(batch) >= 2:
        for entry, row in zip(batch, rows):
            matched[entry["family_id"]] = row

    return matched


def extract(families: list[dict], transport=None, report=None) -> dict:
    """Attributes for each family. Never raises — a failed pass is explained.

    `families` is [{family_id, code, golden, spellings: [...]}]. Nothing else is
    sent: no prices, no unit names, no legacy codes.

    Every way this can produce nothing is counted, because "0 standardised"
    with no reason on screen is the least useful possible outcome and is
    exactly what the first version did.
    """
    out: dict[str, dict] = {}
    errors: list[str] = []
    notes: list[str] = []
    calls = 0
    stats = {"asked": len(families), "rows_returned": 0, "matched": 0,
             "empty_attributes": 0, "unmatched": 0}
    sample = None

    batches = [families[i : i + FAMILIES_PER_CALL]
               for i in range(0, len(families), FAMILIES_PER_CALL)]

    for index, batch in enumerate(batches):
        if report is not None:
            report(index / max(1, len(batches)),
                   f"Extracting attributes, {index * FAMILIES_PER_CALL:,} of {len(families):,}")
        items = "\n\n".join(_describe(entry) for entry in batch)
        try:
            answer = generate_json(PROMPT.format(items=items), transport=transport)
            calls += 1
        except AIUnavailable as exc:
            errors.append(str(exc))
            break

        rows = as_rows(answer)
        stats["rows_returned"] += len(rows)
        if sample is None and rows:
            # One real row, kept for the diagnostics panel. If the shape is
            # wrong this is the thing that says how.
            sample = {k: rows[0].get(k) for k in (["id"] + FIELDS + ["confidence"])}
        if not rows:
            notes.append("A batch came back as %s with no usable rows."
                         % type(answer).__name__)
            continue

        matched = match_rows(batch, rows)
        stats["matched"] += len(matched)
        stats["unmatched"] += len(batch) - len(matched)

        for entry in batch:
            row = matched.get(entry["family_id"])
            if row is None:
                continue
            attributes = {}
            for field in FIELDS:
                value = row.get(field)
                if value is None or str(value).strip() == "" or str(value).strip().lower() in ("null", "none", "n/a"):
                    attributes[field] = None
                else:
                    attributes[field] = str(value).strip().upper()[:80]
            standard = canonical_description(attributes)
            if not standard:
                stats["empty_attributes"] += 1
                continue
            try:
                confidence = float(row.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            out[entry["family_id"]] = {
                "attributes": attributes,
                "standard": standard,
                "confidence": max(0.0, min(1.0, confidence)),
                "fingerprint": entry["fingerprint"],
                # Stated plainly wherever this is shown. The pipeline's own
                # output is arithmetic; this is a model's reading of text.
                "source": "gemini",
            }

    if stats["unmatched"] and not out:
        notes.append(
            "The model returned %d row(s) but none could be matched to a family. "
            "It is answering with different identifiers than it was given."
            % stats["rows_returned"])
    if stats["empty_attributes"] and not out:
        notes.append(
            "%d row(s) matched but every attribute came back empty, so there was "
            "nothing to assemble into a standard." % stats["empty_attributes"])

    if report is not None:
        report(1.0, f"{len(out):,} of {len(families):,} standardised")

    return {"attributes": out, "errors": errors, "notes": notes,
            "calls": calls, "stats": stats, "sample": sample}


def families_from(result, limit: int | None = None) -> list[dict]:
    """The minimum a model needs to name a part: the canonical text and the
    ways it was written. Prices, units and codes are deliberately absent."""
    clusters = result.clusters
    records = result.records
    if limit:
        clusters = clusters.sort_values("Members", ascending=False).head(limit)

    wanted = set(clusters["Family_ID"])
    spellings: dict[str, list[str]] = {}
    for rec in records[records["Family_ID"].isin(wanted)].itertuples(index=False):
        bucket = spellings.setdefault(str(rec.Family_ID), [])
        text = str(rec.Original_Description).strip()
        if text and text not in bucket and len(bucket) < 6:
            bucket.append(text)

    out = []
    for row in clusters.itertuples(index=False):
        fid = str(row.Family_ID)
        variants = spellings.get(fid, [])
        out.append({
            "family_id": fid,
            "code": str(row.Canonical_Code),
            "golden": str(row.Golden_Record),
            "spellings": variants,
            "fingerprint": _fingerprint(str(row.Golden_Record), variants),
        })
    return out
