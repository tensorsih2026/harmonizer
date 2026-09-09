"""
CPSE material harmonization pipeline.

This is the logic from the original CLI script, refactored so a web server can
call it: no input() prompts, no disk paths, no module-level work. The model is
loaded once by the caller and passed in.

The normalization rules and the clustering parameters are unchanged from the
original script.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import pandas as pd
import torch
# util.community_detection is deliberately NOT imported: it is the thing this
# module had to replace to survive a 50,000-row file. See detect_communities.
from sentence_transformers import SentenceTransformer

import codes
import uom
from evaluate import find_label_column


# --------------------------------------------------------------------------
# Column detection. The original script hardcoded "Description" and probed a
# list of candidates for CPSE and material code; we keep that behaviour and
# widen the description candidates so real exports load without renaming.
# --------------------------------------------------------------------------

DESCRIPTION_CANDIDATES = [
    "Description", "description", "DESCRIPTION",
    "Material Description", "Material_Description", "MaterialDescription",
    "Item Description", "Item_Description",
    "Long Description", "Short Text", "Material Text",
]

CPSE_CANDIDATES = [
    # Deliberately excludes a bare "Unit": in material masters that column
    # almost always holds the unit of measure (nos, Pcs, EA), not the owning
    # organisation. Matching it would file every row under "NOS".
    "CPSE", "CPSE Name", "CPSE_Name", "Organization", "Organisation",
    "Company", "Business Unit", "Owning Unit", "Source", "Plant", "Refinery",
]

CODE_CANDIDATES = [
    "Material Code", "Material_Code", "MaterialCode", "Code",
    "Item Code", "Item_Code", "Material Number", "Material_Number",
]

# Unchanged from the original script.
NORMALIZATION_RULES = {
    r"\bS\.?S\.?\b": "STAINLESS STEEL",
    r"\bC\.?S\.?\b": "CARBON STEEL",
    r"\bM\.?S\.?\b": "MILD STEEL",
    r"\bDIA\b": "DIAMETER",
    r"\bAPPX\b": "APPROXIMATE",
}

# Values that mean "nothing here" in an export, however they were typed.
PLACEHOLDERS = {
    "", "-", "--", ".", "?", "??", "NA", "N/A", "NAN", "NONE", "NULL", "NIL",
    "UNKNOWN", "UNSPECIFIED", "ERROR", "#N/A", "#NA", "#REF!", "#VALUE!", "TBD",
    "XX", "XXX", "0",
}
UNSPECIFIED = "UNSPECIFIED"


def clean_unit(value, fallback: str = UNSPECIFIED) -> str:
    """
    Fold a source-unit value to one canonical form.

    A real export writes the same unit as 'A', ' A', 'a' and 'A  ', and mixes in
    'UNKNOWN', 'N/A' and blanks. Left alone those become eleven separate units
    in the workspace, which makes cross-unit duplication impossible to read.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return fallback
    text = re.sub(r"\s+", " ", str(value)).strip().upper()
    return fallback if text in PLACEHOLDERS else text


SIMILARITY_THRESHOLD = 0.75
HIGH_CONFIDENCE = 0.90
MEDIUM_CONFIDENCE = 0.75

# Family status is decided by mean similarity falling into one of these bands,
# highest first; anything below the last one is a conflict. Defined once here
# and shipped to the frontend with every run, so no number describing the
# pipeline is ever typed a second time into a template. Change a figure here
# and every label in the UI moves with it.
# A unit needs at least this many priced rows in a family before its median is
# compared against the family's. Below that, one emergency purchase at a bad
# rate would flag an entire office.
MIN_ROWS_FOR_UNIT_OUTLIER = 5

# How far above the family median a unit has to sit before it is worth an
# officer's attention at all.
UNIT_OUTLIER_RATIO = 1.25

STATUS_BANDS: list[tuple[str, float]] = [
    ("resolved", 0.95),
    ("accepted", 0.82),
    ("review", MEDIUM_CONFIDENCE),
]
FALLBACK_STATUS = "conflict"


def status_for(mean_score: float) -> str:
    for name, floor in STATUS_BANDS:
        if mean_score >= floor:
            return name
    return FALLBACK_STATUS


def threshold_manifest() -> dict:
    """Everything the UI needs to describe its own numbers truthfully."""
    bands = []
    previous = 1.0
    for name, floor in STATUS_BANDS:
        bands.append({"name": name, "floor": floor, "ceiling": previous})
        previous = floor
    bands.append({"name": FALLBACK_STATUS, "floor": 0.0, "ceiling": previous})
    return {
        "cluster": SIMILARITY_THRESHOLD,
        "high_confidence": HIGH_CONFIDENCE,
        "medium_confidence": MEDIUM_CONFIDENCE,
        "auto_merge": next(f for n, f in STATUS_BANDS if n == "accepted"),
        "bands": bands,
    }

MODEL_NAME = "all-MiniLM-L6-v2"

# The four stages the UI shows. Index order matters — the frontend drives its
# progress rows off stage_index.
STAGES = [
    "Reading and normalizing records",
    "Encoding descriptions",
    "Detecting communities",
    "Building golden records",
]

ProgressFn = Callable[[int, float, str], None]


class PipelineError(Exception):
    """Raised for input problems the user can fix, e.g. a missing column."""


@dataclass
class HarmonizationResult:
    records: pd.DataFrame          # one row per source record
    clusters: pd.DataFrame         # one row per family
    stats: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    # Every reviewer decision taken against this run, in order. Written by
    # decisions.record() from inside the verbs themselves, so a decision cannot
    # be in the run without being in the log. See decisions.py.
    decisions: list = field(default_factory=list)


def pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """First candidate present in df, matched case-insensitively."""
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        hit = lowered.get(candidate.strip().lower())
        if hit is not None:
            return hit
    return None


# --------------------------------------------------------------------------
# Column inference
#
# An exact-name allowlist is the wrong shape for this problem. Every CPSE
# exports from a different system with a different naming convention, and the
# whole premise of the product is that those systems do not agree with each
# other. A file whose description column is called "Messy_Description" or
# "Item_Text" or "MATNR_DESC" is not a malformed file — it is the normal case,
# and rejecting it tells the user their data is wrong when in fact the matcher
# was too narrow.
#
# So the allowlist is kept only as a fast path for exact hits, and everything
# else falls through to scoring. Each column is judged on two independent
# signals: what it is CALLED, and what it CONTAINS. Name evidence alone is not
# enough ("Source" could be a system name or a supplier); content evidence
# alone is not enough (two free-text columns look alike). Requiring both to
# agree is what keeps the inference from confidently picking the wrong column.
# --------------------------------------------------------------------------

# Tokens that suggest a role, and what each is worth. Substring matches on the
# normalized column name, so "Messy_Description" and "MATERIALDESC" both hit.
DESC_TOKENS = {
    "description": 6.0, "desc": 5.0, "material text": 5.0, "short text": 5.0,
    "long text": 4.0, "item text": 4.0, "text": 2.0, "nomenclature": 5.0,
    "particular": 4.0, "narration": 3.5, "itemname": 4.0, "item name": 4.0,
    "material": 2.0, "item": 1.5, "name": 1.0, "title": 1.0, "spec": 1.5,
    # SAP field names. Not a special case for one user's file — SAP MM is what
    # most of these organisations actually run, and MAKTX is as much a real
    # column name in this domain as "Description" is.
    "maktx": 6.0, "maktg": 5.0, "txz01": 4.5,
}
UNIT_TOKENS = {
    "cpse": 6.0, "organisation": 5.0, "organization": 5.0, "company": 4.5,
    "source_system": 5.0, "source system": 5.0, "sourcesystem": 5.0,
    "business unit": 4.5, "owning unit": 4.5, "refinery": 4.0, "plant": 3.5,
    "location": 3.0, "entity": 3.5, "division": 3.0, "site": 3.0,
    "source": 3.0, "system": 2.5, "org": 2.5, "unit": 1.0, "depot": 3.0,
    "werks": 5.0, "bukrs": 5.0,          # SAP plant / company code
}
PRICE_TOKENS = {
    "unit price": 6.0, "unit_price": 6.0, "unitprice": 6.0, "rate": 5.0,
    "price": 5.5, "amount": 4.0, "value": 3.0, "cost": 5.0, "landed": 4.0,
    "net price": 5.5, "po value": 4.5, "inr": 2.0, "rupees": 2.5, "usd": 1.5,
    "netpr": 4.5, "peinh": 2.0,          # SAP net price / price unit
}

CODE_TOKENS = {
    "material code": 6.0, "material number": 6.0, "matnr": 5.5,
    "item code": 5.5, "item_code": 5.5, "material_code": 6.0,
    "part number": 5.0, "sku": 4.5, "code": 3.5, "number": 2.0,
    "no.": 1.5, "id": 1.5, "ref": 1.5,
    "matnr": 6.0, "bismt": 4.0,          # SAP material / old material number
}

UOM_TOKENS = {
    "uom": 6.0, "unit of measure": 6.0, "unitofmeasure": 6.0, "u/m": 5.0,
    "base unit": 5.0, "baseunit": 5.0, "order unit": 4.5, "issue unit": 4.5,
    "stock unit": 4.5, "measure": 4.0, "uom code": 5.0,
    "meins": 6.0, "bstme": 4.5,          # SAP base / order unit of measure
    "unit": 1.5,
}

# Values that mark a column as a unit OF MEASURE rather than an owning unit.
# Detecting this by content rather than by column name is what stops a column
# innocently called "Unit" or "UOM" from filing every row under "NOS".
UOM_VALUES = {
    "NOS", "NO", "NOS.", "PCS", "PC", "PIECE", "PIECES", "EA", "EACH", "SET",
    "SETS", "PAIR", "BOX", "PKT", "PACK", "KG", "KGS", "GM", "GRAM", "TON",
    "MT", "LTR", "LTRS", "LITRE", "LITER", "ML", "MTR", "MTRS", "METER",
    "METRE", "CM", "MM", "INCH", "FT", "SQM", "CUM", "ROLL", "COIL", "DRUM",
    "BAG", "BTL", "BOTTLE", "CAN", "TUBE", "REAM", "DOZ", "DOZEN", "UNIT",
}


def _name_score(column: str, tokens: dict[str, float]) -> float:
    text = re.sub(r"[_\-]+", " ", str(column).strip().lower())
    tight = text.replace(" ", "")
    best = 0.0
    for token, weight in tokens.items():
        if token in text or token.replace(" ", "") in tight:
            best = max(best, weight)
    return best


def _profile(series: pd.Series, sample: int = 4000) -> dict:
    """Cheap content statistics, computed on a sample so a 500k-row file costs
    the same as a 5k-row one."""
    values = series.dropna()
    if len(values) > sample:
        values = values.sample(sample, random_state=0)
    text = values.astype(str).str.strip()
    text = text[text != ""]
    n = len(text)
    if n == 0:
        return {"n": 0, "mean_len": 0.0, "distinct": 0.0, "numeric": 1.0,
                "spaces": 0.0, "uom": 0.0, "positive": 0.0, "fractional": 0.0}

    lengths = text.str.len()
    as_number = pd.to_numeric(text.str.replace(",", "", regex=False), errors="coerce")
    numeric = as_number.notna().mean()
    upper = text.str.upper()
    clean = as_number.dropna()
    return {
        "n": n,
        "mean_len": float(lengths.mean()),
        "distinct": float(text.nunique() / n),
        "numeric": float(numeric),
        "spaces": float(text.str.contains(" ").mean()),
        "uom": float(upper.isin(UOM_VALUES).mean()),
        # A money column is numeric, positive, and usually not whole numbers.
        "positive": float((clean > 0).mean()) if len(clean) else 0.0,
        "fractional": float((clean % 1 != 0).mean()) if len(clean) else 0.0,
    }


def _describe_choice(column: str, how: str) -> str:
    return f"{column} ({how})"


def infer_columns(df: pd.DataFrame) -> dict:
    """Work out which column holds the description, the owning unit and the
    legacy code. Returns the choices plus a plain-English reason for each, so
    the UI can show the user what was assumed rather than silently guessing."""
    columns = list(df.columns)
    profiles = {c: _profile(df[c]) for c in columns}
    reasons: dict[str, str] = {}

    # ---- description -----------------------------------------------------
    desc = pick_column(df, DESCRIPTION_CANDIDATES)
    if desc is not None:
        reasons["description"] = _describe_choice(str(desc), "matched a known column name")
    else:
        best, best_score = None, 0.0
        for c in columns:
            p = profiles[c]
            if p["n"] == 0:
                continue
            name = _name_score(c, DESC_TOKENS)
            # Free text is long, varied, wordy and not a number.
            content = (
                min(p["mean_len"] / 18.0, 2.0) * 2.2
                + p["distinct"] * 2.4
                + p["spaces"] * 2.0
                - p["numeric"] * 5.0
                - p["uom"] * 6.0
            )
            score = name + content
            # Both signals must be present: a column with a description-ish
            # name but numeric content is a code, and a long free-text column
            # named "Remarks" is not the material description.
            if name < 1.0 or content < 1.2:
                continue
            if score > best_score:
                best, best_score = c, score
        if best is None:
            # Nothing had a recognisable name. An export can legitimately use
            # column names that mean nothing outside the system that produced
            # them, so fall back to content alone — but only when a single
            # column is unambiguously the free-text one. If two columns both
            # look like prose there is no safe guess, and refusing is the
            # honest outcome.
            scored = []
            for c in columns:
                p = profiles[c]
                if p["n"] == 0:
                    continue
                if p["mean_len"] < 8 or p["numeric"] > 0.2 or p["uom"] > 0.2:
                    continue
                if p["spaces"] < 0.2 or p["distinct"] < 0.005:
                    continue
                scored.append((
                    min(p["mean_len"] / 18.0, 2.0) * 2.2 + p["distinct"] * 2.4 + p["spaces"] * 2.0,
                    c,
                ))
            scored.sort(reverse=True)
            if scored and (len(scored) == 1 or scored[0][0] > scored[1][0] * 1.5):
                best = scored[0][1]
                reasons["description"] = _describe_choice(
                    str(best),
                    f"inferred from content alone — the only free-text column, "
                    f"{profiles[best]['mean_len']:.0f} characters on average"
                )
                desc = best

        elif best is not None:
            desc = best
            reasons["description"] = _describe_choice(
                str(best),
                f"inferred from its name and content — "
                f"{profiles[best]['mean_len']:.0f} characters on average, "
                f"{profiles[best]['distinct'] * 100:.0f}% distinct"
            )

    # ---- owning unit -----------------------------------------------------
    unit = pick_column(df, CPSE_CANDIDATES)
    if unit is not None and profiles[unit]["uom"] > 0.5:
        unit = None                       # named like a unit, holds UOM values
    if unit is not None:
        reasons["unit"] = _describe_choice(str(unit), "matched a known column name")
    else:
        best, best_score = None, 0.0
        for c in columns:
            if c == desc:
                continue
            p = profiles[c]
            if p["n"] == 0 or p["uom"] > 0.35:
                continue
            name = _name_score(c, UNIT_TOKENS)
            if name < 2.0:
                continue
            # An owning unit repeats: a handful of values across many rows.
            content = (
                (2.5 if p["distinct"] < 0.05 else 1.2 if p["distinct"] < 0.2 else -1.5)
                + (1.0 if p["mean_len"] <= 24 else -1.0)
                - p["numeric"] * 2.5
            )
            score = name + content
            if content <= 0:
                continue
            if score > best_score:
                best, best_score = c, score
        if best is not None:
            unit = best
            reasons["unit"] = _describe_choice(
                str(best),
                f"inferred — {int(round(profiles[best]['distinct'] * profiles[best]['n']))} "
                f"distinct values repeating across the file"
            )

    # ---- legacy code -----------------------------------------------------
    code = pick_column(df, CODE_CANDIDATES)
    if code is not None:
        reasons["code"] = _describe_choice(str(code), "matched a known column name")
    else:
        best, best_score = None, 0.0
        for c in columns:
            if c in (desc, unit):
                continue
            p = profiles[c]
            if p["n"] == 0:
                continue
            name = _name_score(c, CODE_TOKENS)
            if name < 1.5:
                continue
            # An identifier is near-unique, short and rarely has spaces.
            content = (
                p["distinct"] * 3.0
                + (1.2 if p["mean_len"] <= 24 else -1.5)
                - p["spaces"] * 1.5
            )
            score = name + content
            if p["distinct"] < 0.4:
                continue
            if score > best_score:
                best, best_score = c, score
        if best is not None:
            code = best
            reasons["code"] = _describe_choice(str(best), "inferred from its name and near-unique values")

    # ---- unit price ------------------------------------------------------
    # Optional. Without it the savings analysis simply does not run — it is
    # never approximated, because a made-up rupee figure in a procurement tool
    # is worse than no figure at all.
    price = None
    best, best_score = None, 0.0
    for c in columns:
        if c in (desc, unit, code):
            continue
        p = profiles[c]
        if p["n"] == 0:
            continue
        name = _name_score(c, PRICE_TOKENS)
        if name < 2.0:
            continue
        # Money is numeric and positive; quantities are too, so the tie-break
        # is that prices are rarely whole numbers and rarely repeat.
        content = (
            p["numeric"] * 3.0
            + p["positive"] * 2.0
            + p["fractional"] * 1.5
            + (0.8 if p["distinct"] > 0.3 else 0.0)
        )
        if p["numeric"] < 0.8 or p["positive"] < 0.8:
            continue
        score = name + content
        if score > best_score:
            best, best_score = c, score
    if best is not None:
        price = best
        reasons["price"] = _describe_choice(
            str(best),
            "matched a known column name" if _name_score(best, PRICE_TOKENS) >= 5.0
            else "inferred — numeric, positive, and priced to the paisa"
        )

    # ---- unit of measure -------------------------------------------------
    # Optional, and found by CONTENT first. A column called "Unit" might hold
    # CPSE names; a column called "Base" might hold NOS. What settles it is
    # whether the values themselves are units of measure, which is a test the
    # profiler already runs for a different reason.
    measure = None
    best, best_score = None, 0.0
    for c in columns:
        if c in (desc, unit, code, price):
            continue
        p = profiles[c]
        if p["n"] == 0 or p["uom"] < 0.45:
            continue
        # Name is a tie-break, never the deciding vote.
        score = p["uom"] * 6.0 + _name_score(c, UOM_TOKENS)
        # A unit of measure column is short and highly repetitive.
        if p["mean_len"] > 12 or p["distinct"] > 0.25:
            continue
        if score > best_score:
            best, best_score = c, score
    if best is not None:
        measure = best
        reasons["measure"] = _describe_choice(
            str(best),
            "matched a known column name" if _name_score(best, UOM_TOKENS) >= 4.0
            else "inferred \u2014 short, repetitive, and holding unit-of-measure values"
        )

    return {"description": desc, "unit": unit, "code": code, "price": price,
            "measure": measure, "reasons": reasons}


# How many neighbour references the candidate pass may hold before it stops
# materialising them. Sized so the fallback engages long before memory does:
# 20 million references is a few hundred MB of Python integers.
MAX_NEIGHBOUR_REFS = 20_000_000


def detect_communities(embeddings, threshold: float, report=None,
                       block: int = 512) -> list[list[int]]:
    """Group L2-normalized vectors into communities above a cosine threshold.

    This replaces sentence_transformers.util.community_detection, and the
    reason is worth recording because nothing at the call site reveals it.

    That helper, called with min_community_size=1, appends a Python list of
    EVERY neighbour above the threshold FOR EVERY POINT. Its memory is the sum
    of the squares of the community sizes, not the number of points — so a
    50,000-row master with eighty real families builds roughly 31 million
    Python integers across 50,000 lists, sorts them, and walks them against a
    growing set. On a laptop that is a MemoryError during the encoding stage,
    and what the user sees is the word "failed" with nothing on screen that
    explains it.

    The behaviour here is the same and the failure mode is not:

      * Similarity is produced one BLOCK of rows at a time, so peak memory is
        block x N floats regardless of how similar the data turns out to be.
      * Candidate communities are collected, then taken largest-first with
        already-claimed points removed. That is what the original does, and it
        matters for quality: seeding from the densest region first is what
        stops a point that bridges two families from merging them.
      * The candidate pass is CAPPED. If a pathologically dense file would
        exceed the cap, it degrades to a streaming pass that seeds by
        neighbour count and never materialises the lists — a slightly coarser
        grouping instead of a dead process.

    Callers run this over DISTINCT descriptions rather than rows, which is
    what keeps the cap comfortable: repetition is collapsed before it gets
    here, and identical strings were always going to land together anyway.

    Deterministic: ties are broken by index, so a file always yields the same
    families.
    """
    n = int(embeddings.shape[0])
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    candidates: list[list[int]] = []
    refs = 0
    overflowed = False

    for start in range(0, n, block):
        stop = min(n, start + block)
        sims = torch.matmul(embeddings[start:stop], embeddings.T)
        hits = (sims >= threshold)
        for row in range(stop - start):
            near = hits[row].nonzero().flatten().tolist()
            refs += len(near)
            if refs > MAX_NEIGHBOUR_REFS:
                overflowed = True
                break
            candidates.append(near)
        del sims, hits
        if overflowed:
            break
        if report is not None:
            report(2, 0.6 * stop / n, f"Comparing {n:,} descriptions")

    if overflowed:
        return _stream_communities(embeddings, threshold, n, block, report)

    # Largest first; index order within a tie so the result is reproducible.
    order = sorted(range(len(candidates)), key=lambda i: (-len(candidates[i]), i))

    claimed = [False] * n
    communities: list[list[int]] = []
    for position, i in enumerate(order):
        members = [idx for idx in candidates[i] if not claimed[idx]]
        if not members:
            continue
        for idx in members:
            claimed[idx] = True
        communities.append(members)
        if report is not None and position % 500 == 0:
            report(2, 0.6 + 0.4 * position / max(1, len(order)),
                   f"{len(communities):,} families so far")

    for idx in range(n):
        if not claimed[idx]:
            communities.append([idx])

    communities.sort(key=len, reverse=True)
    return communities


def _stream_communities(embeddings, threshold: float, n: int, block: int,
                        report=None) -> list[list[int]]:
    """Memory-bounded fallback for a file too dense to hold candidates for.

    Counts neighbours without keeping them, seeds from the densest point down,
    and claims each seed's unclaimed neighbourhood. Coarser than the main path
    — a bridging point can pull two families together — but it completes in
    bounded memory on any input, which is the only property that matters once
    the alternative is a crash."""
    degrees = [0] * n
    for start in range(0, n, block):
        stop = min(n, start + block)
        sims = torch.matmul(embeddings[start:stop], embeddings.T)
        hits = (sims >= threshold)
        for row in range(stop - start):
            degrees[start + row] = len(hits[row].nonzero().flatten().tolist())
        del sims, hits
        if report is not None:
            report(2, 0.5 * stop / n, f"Comparing {n:,} descriptions (streaming)")

    order = sorted(range(n), key=lambda i: (-degrees[i], i))
    claimed = [False] * n
    communities: list[list[int]] = []
    for seed in order:
        if claimed[seed]:
            continue
        sims = torch.matmul(embeddings, embeddings[seed])
        near = (sims >= threshold).nonzero().flatten().tolist()
        del sims
        members = [idx for idx in near if not claimed[idx]] or [seed]
        for idx in members:
            claimed[idx] = True
        communities.append(members)
    communities.sort(key=len, reverse=True)
    return communities


def normalize_text(text: str) -> str:
    text = str(text).strip().upper()
    return re.sub(r"\s+", " ", text)


def load_model(device: Optional[str] = None) -> SentenceTransformer:
    """Load the embedding model once, at server startup."""
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    return SentenceTransformer(MODEL_NAME, device=device)


# Bytes that mark a file as definitely-not-a-table. Checked before parsing so a
# PNG produces an instant, readable error instead of a job that grinds through
# binary garbage for ten minutes.
BINARY_SIGNATURES = {
    b"\x89PNG": "a PNG image",
    b"\xff\xd8\xff": "a JPEG image",
    b"GIF8": "a GIF image",
    b"%PDF": "a PDF",
    b"PK\x03\x04": "a ZIP or Office file (.xlsx, .docx)",
    b"\xd0\xcf\x11\xe0": "a legacy Office file (.xls, .doc)",
    b"\x1f\x8b": "a gzip archive",
    b"Rar!": "a RAR archive",
    b"\x7fELF": "a binary executable",
    b"BM": "a bitmap image",
}

MAX_HEADER_SCAN = 20          # how many leading junk lines to look past
MIN_TABLE_COLUMNS = 2


EXCEL_SUFFIXES = (".xlsx", ".xlsm", ".xltx", ".xls")


def looks_like_excel(name: str, raw: bytes) -> bool:
    """An Office container whose filename claims a workbook."""
    if not name.lower().endswith(EXCEL_SUFFIXES):
        return False
    return raw.startswith(b"PK\x03\x04") or raw.startswith(b"\xd0\xcf\x11\xe0")


def sniff_binary(name: str, raw: bytes) -> Optional[str]:
    """Return a human description if the bytes are clearly not text, else None."""
    if looks_like_excel(name, raw):
        return None
    head = raw[:4096]
    for signature, label in BINARY_SIGNATURES.items():
        if raw.startswith(signature):
            return label
    if b"\x00" in head:
        return "a binary file"
    return None


def read_excel(name: str, raw: bytes) -> pd.DataFrame:
    """
    Read a workbook, looking past banner rows exactly as the CSV path does.

    Only the first sheet is read: a material master export is one table, and
    silently concatenating other sheets would invent rows.
    """
    engine = "xlrd" if name.lower().endswith(".xls") else "openpyxl"
    try:
        probe = pd.read_excel(io.BytesIO(raw), sheet_name=0, header=None, dtype=str, engine=engine)
    except ImportError as exc:
        raise PipelineError(
            f"Cannot read {name}: the Excel reader is not installed "
            f"({exc}). Save the file as CSV and upload that."
        ) from exc
    except Exception as exc:
        raise PipelineError(f"Could not read {name} as a workbook: {exc}") from exc

    if probe.empty:
        raise PipelineError(f"{name} has no rows in its first sheet.")

    best_any: Optional[pd.DataFrame] = None
    for skip in range(min(MAX_HEADER_SCAN, len(probe))):
        header = probe.iloc[skip]
        body = probe.iloc[skip + 1:]
        if body.empty:
            break
        frame = body.copy()
        frame.columns = [str(c).strip() if pd.notna(c) else f"col_{i}"
                         for i, c in enumerate(header)]
        frame = frame.reset_index(drop=True)
        frame = frame.dropna(axis=1, how="all")
        if frame.shape[1] < MIN_TABLE_COLUMNS:
            continue
        if pick_column(frame, DESCRIPTION_CANDIDATES) is not None:
            return frame
        if best_any is None:
            best_any = frame

    if best_any is not None:
        return best_any
    raise PipelineError(f"Could not find a header row in {name}.")


def decode_text(name: str, raw: bytes) -> str:
    """Decode upload bytes, tolerating the encodings ERP exports arrive in."""
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    # latin-1 accepts any byte sequence, so reaching here means something is
    # very wrong with the input rather than with the encoding guess.
    raise PipelineError(f"Could not decode {name} as text.")


def parse_table(name: str, raw: bytes) -> pd.DataFrame:
    """
    Parse one uploaded file into a DataFrame.

    Real exports often carry banner rows above the real header ("MATERIAL
    MASTER EXTRACT", a timestamp, a blank line), so the header is searched for
    rather than assumed to be line 1. The delimiter is sniffed too, which picks
    up tab- and semicolon-separated files without the user having to say so.
    """
    binary = sniff_binary(name, raw)
    if binary is not None:
        raise PipelineError(
            f"{name} looks like {binary}, not a data table. "
            "Upload a CSV, TSV, Excel or text export of your material master."
        )

    if looks_like_excel(name, raw):
        return read_excel(name, raw)

    text = decode_text(name, raw)
    if not text.strip():
        raise PipelineError(f"{name} is empty.")

    best_any: Optional[pd.DataFrame] = None
    last_error: Optional[str] = None

    for skip in range(MAX_HEADER_SCAN):
        for reader in (
            dict(engine="c", sep=","),
            dict(engine="python", sep=None),      # sniffs tab / semicolon / pipe
        ):
            try:
                frame = pd.read_csv(
                    io.StringIO(text),
                    skiprows=skip,
                    dtype=str,
                    keep_default_na=False,
                    na_values=[""],
                    on_bad_lines="skip",
                    **reader,
                )
            except Exception as exc:              # noqa: BLE001 - reported below
                last_error = str(exc).splitlines()[0]
                continue

            if frame.shape[1] < MIN_TABLE_COLUMNS or frame.empty:
                continue

            # A header row we can actually use ends the search immediately.
            if pick_column(frame, DESCRIPTION_CANDIDATES) is not None:
                return frame

            if best_any is None:
                best_any = frame

    if best_any is not None:
        # Parsed as a table, but no column we recognise as a description. The
        # caller raises a message naming the columns we did find.
        return best_any

    raise PipelineError(
        f"Could not read {name} as a table"
        + (f" ({last_error})" if last_error else "")
        + ". Expected a CSV or TSV with a header row."
    )


CANONICAL_DESC = "Description"
CANONICAL_UNIT = "CPSE"
CANONICAL_CODE = "Material Code"
CANONICAL_PRICE = "Unit Price"
CANONICAL_UOM = "Unit of Measure"
# Ground truth, when a file carries it. Never exported; used only to score.
CANONICAL_LABEL = "Gold_Label"
CANONICAL_FILE = "Source_File"


def read_tables(files: list[tuple[str, bytes]]) -> pd.DataFrame:
    """
    Parse and combine one or more uploaded exports into one canonical table.

    Each file is normalised to the same four columns BEFORE concatenation.
    Without this, two exports that name the description column differently —
    "Description" in one, "Material Description" in another — end up in
    separate columns, and every row from the second file is silently dropped as
    blank. Real material masters from different units rarely share a schema, so
    this is the common case, not the edge case.

    A file with no unit column takes its unit name from the filename, which is
    what keeps provenance intact when several unit exports are merged.
    """
    frames = []
    notes: list[str] = []
    label_notes: list[str] = []
    uom_notes: list[dict] = []
    mapped: dict[str, list[str]] = {}
    source_columns: dict[str, list[str]] = {}

    for name, raw in files:
        # One unusable file should not sink a batch of ten good ones. The
        # reason is kept and surfaced as a warning on the result.
        try:
            frame = parse_table(name, raw)
        except PipelineError as exc:
            notes.append(str(exc))
            continue

        stem = re.sub(r"\.(csv|tsv|txt|dat|psv|xlsx|xlsm|xltx|xls)$", "", name, flags=re.I)

        picked = infer_columns(frame)
        desc_col = picked["description"]
        if desc_col is None:
            notes.append(
                f"{name} was skipped — could not identify a description column. "
                f"Looked for a column named like a description, or one holding "
                f"varied free text. Found: "
                f"{', '.join(str(c) for c in frame.columns[:8])}."
            )
            continue

        unit_col = picked["unit"]
        code_col = picked["code"]
        price_col = picked["price"]
        measure_col = picked.get("measure")

        # Say what was assumed. A wrong guess the user can see is recoverable;
        # a wrong guess made silently is not.
        for role in ("description", "unit", "code", "price", "measure"):
            if role in picked["reasons"] and "inferred" in picked["reasons"][role]:
                notes.append(f"{name}: {role} column = {picked['reasons'][role]}.")

        for role, chosen in (("description", desc_col), ("unit", unit_col),
                             ("code", code_col), ("price", price_col),
                             ("measure", measure_col)):
            if chosen is not None:
                source_columns.setdefault(role, []).append(str(chosen))

        if unit_col is not None:
            # The file names its own units, so a placeholder value means the
            # unit is genuinely unknown for that row. Borrowing the filename
            # here would invent provenance.
            units = frame[unit_col].map(clean_unit)
        else:
            # No unit column at all: the filename is the only provenance there
            # is, and one file per unit is the documented upload shape.
            units = stem

        if price_col is not None:
            # Strip thousands separators and currency symbols before coercing;
            # anything that will not parse becomes NaN and is simply excluded
            # from the savings analysis rather than guessed at.
            prices = pd.to_numeric(
                frame[price_col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
                errors="coerce",
            )
        else:
            prices = pd.Series([float("nan")] * len(frame), index=frame.index)

        # Unit of measure, folded the same way the descriptions are. EA, NOS
        # and NO are one unit; a price per piece and a price per kilo are not
        # the same number and never become one. See uom.py.
        if measure_col is not None:
            measures = frame[measure_col].map(lambda v: uom.normalize(v)[0])
            uom_notes.append(uom.summarize(frame[measure_col]))
        else:
            measures = pd.Series([uom.UNSPECIFIED] * len(frame), index=frame.index)

        # A ground-truth column, if this file happens to carry one. Real
        # material masters often do — a verified code, a reviewed group, a
        # de-duplication key from an earlier cleanup — and it is the only thing
        # that can turn "mean similarity 0.94" into a measured accuracy.
        # Detected by name AND content like every other column here, so it
        # works whatever the exporting system called it, and carried through
        # under one canonical name so files that disagree still line up.
        label_col, label_reason = find_label_column(
            frame, exclude={desc_col, unit_col, code_col, price_col, measure_col}
        )
        label_notes.append(f"{name}: {label_reason}")
        if label_col is not None:
            labels = frame[label_col].fillna("").astype(str).str.strip()
            source_columns.setdefault("label", []).append(str(label_col))
        else:
            labels = pd.Series([""] * len(frame), index=frame.index)

        canonical = pd.DataFrame({
            CANONICAL_DESC: frame[desc_col],
            CANONICAL_UNIT: units,
            CANONICAL_CODE: frame[code_col].fillna("").astype(str).str.strip() if code_col else "",
            CANONICAL_PRICE: prices,
            CANONICAL_UOM: measures,
            CANONICAL_LABEL: labels,
            CANONICAL_FILE: stem,
        })

        if len(canonical):
            frames.append(canonical)
            if desc_col != CANONICAL_DESC:
                mapped.setdefault(str(desc_col), []).append(name)
        else:
            notes.append(f"{name} was skipped — no rows.")

    if not frames:
        raise PipelineError(
            "None of the uploaded files could be used. "
            + (" ".join(notes) if notes else "")
        )

    if mapped:
        # One line for the whole batch. Ten files with three different header
        # spellings should read as one sentence, not ten.
        parts = [
            f"{col} ({len(names)} file{'s' if len(names) != 1 else ''})"
            for col, names in sorted(mapped.items(), key=lambda kv: -len(kv[1]))
        ]
        notes.insert(0, "Descriptions were read from: " + ", ".join(parts) + ".")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.attrs["notes"] = notes
    # Kept so the evaluation panel can say WHY it found no labels, per file,
    # rather than only that it found none.
    combined.attrs["label_notes"] = label_notes
    combined.attrs["uom_notes"] = uom_notes
    # The columns are renamed to canonical names above, so by the time
    # harmonize() runs it can no longer tell what the user's file actually
    # called them. Carry the originals through, deduplicated in first-seen
    # order, so the UI can report the real header rather than our internal one.
    combined.attrs["source_columns"] = {
        role: list(dict.fromkeys(names)) for role, names in source_columns.items()
    }
    return combined


def harmonize(
    df: pd.DataFrame,
    model: SentenceTransformer,
    progress: Optional[ProgressFn] = None,
    encode_batch_size: int = 256,
) -> HarmonizationResult:
    """Run the full pipeline. `progress(stage_index, fraction, message)`."""

    def report(stage: int, fraction: float, message: str) -> None:
        if progress is not None:
            progress(stage, max(0.0, min(1.0, fraction)), message)

    warnings: list[str] = list(df.attrs.get("notes", []))

    # ---------------------------------------------------------------- 0 ----
    report(0, 0.05, "Locating the description column")

    picked = infer_columns(df)
    desc_col = picked["description"]
    if desc_col is None:
        raise PipelineError(
            "No description column found. Looked for a column named like a "
            "description (e.g. "
            + ", ".join(DESCRIPTION_CANDIDATES[:4])
            + ") and, failing that, for one holding varied free text. Found: "
            + ", ".join(str(c) for c in df.columns[:12])
        )

    cpse_col = picked["unit"]
    code_col = picked["code"]
    column_reasons = picked["reasons"]

    rows_in = len(df)
    cleaned = df.copy()
    cleaned["Original_Description"] = cleaned[desc_col]
    cleaned = cleaned.dropna(subset=[desc_col]).copy()
    dropped_blank = rows_in - len(cleaned)

    # The original script called drop_duplicates() with no subset, which drops
    # rows identical across every column. That is kept, but the counts are
    # reported so a silent collapse is visible rather than invisible.
    before_dedupe = len(cleaned)
    cleaned = cleaned.drop_duplicates(keep="first").copy()
    dropped_duplicate = before_dedupe - len(cleaned)
    cleaned = cleaned.reset_index(drop=True)

    # A description of "#N/A" or "?" is not a material. Left in, it becomes its
    # own canonical family and pollutes the counts.
    probe = cleaned["Original_Description"].astype(str).str.strip().str.upper()
    junk_mask = probe.isin(PLACEHOLDERS)
    dropped_junk = int(junk_mask.sum())
    if dropped_junk:
        cleaned = cleaned[~junk_mask].copy().reset_index(drop=True)

    if cleaned.empty:
        raise PipelineError("Every row was blank, duplicate or a placeholder — nothing left to harmonize.")

    report(0, 0.4, f"{len(cleaned):,} records after cleaning")

    cleaned["Normalized_Description"] = cleaned["Original_Description"].apply(normalize_text)
    for pattern, replacement in NORMALIZATION_RULES.items():
        cleaned["Normalized_Description"] = cleaned["Normalized_Description"].str.replace(
            pattern, replacement, regex=True
        )
    cleaned["Normalized_Description"] = (
        cleaned["Normalized_Description"]
        .str.replace(r"[,;:/]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    report(0, 1.0, "Normalization complete")

    # ---------------------------------------------------------------- 1 ----
    #
    # Encode DISTINCT strings, not rows.
    #
    # A material master is mostly repetition: fifty thousand rows across the
    # CPSEs might hold only a few thousand different ways of writing a part,
    # and in a single-unit export far fewer. Encoding every row separately
    # spends the entire budget re-deriving vectors that are bit-for-bit equal,
    # and then asks the clustering to compare them against each other.
    #
    # Two identical normalized strings have cosine similarity 1.0 and were
    # always going to land in the same family, so collapsing them first cannot
    # change the answer — it only removes work. On a 50,000-row file this is
    # the difference between the run finishing and the process dying.
    all_texts = cleaned["Normalized_Description"].tolist()
    total = len(all_texts)

    unique_texts: list[str] = []
    text_index: dict[str, int] = {}
    inverse: list[int] = []
    for text in all_texts:
        position = text_index.get(text)
        if position is None:
            position = text_index[text] = len(unique_texts)
            unique_texts.append(text)
        inverse.append(position)

    n_unique = len(unique_texts)
    if n_unique < total:
        report(1, 0.0, f"{total:,} records hold {n_unique:,} distinct descriptions")
    else:
        report(1, 0.0, f"Encoding {n_unique:,} descriptions")

    chunks = []
    done = 0
    for start in range(0, n_unique, encode_batch_size):
        chunk = unique_texts[start : start + encode_batch_size]
        chunks.append(
            model.encode(
                chunk,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=encode_batch_size,
            )
        )
        done += len(chunk)
        report(1, done / max(1, n_unique), f"Encoded {done:,} of {n_unique:,}")

    embeddings = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
    del chunks

    # ---------------------------------------------------------------- 2 ----
    unique_clusters = detect_communities(
        embeddings, SIMILARITY_THRESHOLD, report=report
    )

    # Back to rows. Each distinct string carries every record that wrote it.
    rows_for_unique: list[list[int]] = [[] for _ in range(n_unique)]
    for row, position in enumerate(inverse):
        rows_for_unique[position].append(row)

    clusters = [
        [row for position in community for row in rows_for_unique[position]]
        for community in unique_clusters
    ]
    report(2, 1.0, f"{len(clusters):,} families detected")

    # ---------------------------------------------------------------- 3 ----
    report(3, 0.0, "Selecting golden records")

    uom_loc = (
        cleaned.columns.get_loc(CANONICAL_UOM)
        if CANONICAL_UOM in cleaned.columns else None
    )
    price_loc = (
        cleaned.columns.get_loc(CANONICAL_PRICE)
        if CANONICAL_PRICE in cleaned.columns and cleaned[CANONICAL_PRICE].notna().any()
        else None
    )

    lengths = cleaned["Normalized_Description"].str.len().to_numpy()
    family_ids: list[str] = []
    canonical_codes: list[str] = []
    goldens: list[str] = []
    similarities: list[float] = []
    confidences: list[str] = []
    row_indices: list[int] = []

    cluster_rows = []
    n_clusters = max(len(clusters), 1)

    for family_number, cluster in enumerate(clusters, start=1):
        # Longest normalized description wins, as in the original script.
        golden_index = max(cluster, key=lambda index: lengths[index])
        golden_record = cleaned.iat[golden_index, cleaned.columns.get_loc("Normalized_Description")]

        # Vectorized: embeddings are already L2-normalized, so a dot product
        # against the golden vector is the cosine similarity for the whole
        # family at once. The original looped util.cos_sim per record, which
        # costs tens of thousands of separate calls on a large master.
        # Rows index the cleaned frame; embeddings index distinct strings, so
        # the lookup goes through `inverse`. Identical strings therefore score
        # identically, which is the point.
        member_positions = [inverse[member] for member in cluster]
        member_idx = torch.as_tensor(member_positions, dtype=torch.long, device=embeddings.device)
        sims = torch.matmul(embeddings.index_select(0, member_idx), embeddings[inverse[golden_index]])
        sims = sims.clamp(-1.0, 1.0).tolist()

        family_id = f"FAM_{family_number:05d}"
        canonical_code = f"MTL-{family_number:07d}"

        for member, similarity in zip(cluster, sims):
            similarity = round(float(similarity), 4)
            if similarity >= HIGH_CONFIDENCE:
                confidence = "HIGH"
            elif similarity >= MEDIUM_CONFIDENCE:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"

            row_indices.append(member)
            family_ids.append(family_id)
            canonical_codes.append(canonical_code)
            goldens.append(golden_record)
            similarities.append(similarity)
            confidences.append(confidence)

        member_scores = [round(float(s), 4) for s in sims]
        mean_score = sum(member_scores) / len(member_scores)
        min_score = min(member_scores)

        status = status_for(mean_score)

        # Units ordered by how much they contributed, not by whichever row the
        # loop happened to reach first. Iteration order made the same four
        # units appear shuffled differently on every row, which reads as a
        # rendering bug; the counts are what actually varies between families.
        unit_counts: dict[str, int] = {}
        if cpse_col is not None:
            unit_loc = cleaned.columns.get_loc(cpse_col)
            for member in cluster:
                unit = str(cleaned.iat[member, unit_loc])
                if unit:
                    unit_counts[unit] = unit_counts.get(unit, 0) + 1
        ordered_units = sorted(unit_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        units = [name for name, _ in ordered_units]

        # ---- price spread ------------------------------------------------
        # The whole point of harmonizing: once these rows are known to be the
        # same part, what each unit paid becomes comparable for the first
        # time. Everything here is arithmetic over the user's own figures —
        # nothing is modelled, estimated or filled in.
        spend = low = high = median_price = savings = excess = None
        priced = 0
        basis_uom, other_priced, uom_forms = None, 0, 0
        by_measure: dict[str, list] = {}
        unit_prices: list[tuple[str, float]] = []
        unit_medians: list[tuple[str, float, int, float]] = []
        worst_unit, worst_ratio, worst_excess = None, 1.0, 0.0
        if price_loc is not None:
            # Prices are collected WITH their unit of measure, because a
            # price per piece and a price per kilogram are not two points on
            # one scale. Comparing them and calling the gap "excess spend" is
            # a category error, and it was in every family on the sample file.
            rows = []
            for member in cluster:
                value = cleaned.iat[member, price_loc]
                if value is None or pd.isna(value) or float(value) <= 0:
                    continue
                value = float(value)
                measure = (str(cleaned.iat[member, uom_loc]) if uom_loc is not None
                           else uom.UNSPECIFIED)
                owner = (str(cleaned.iat[member, cleaned.columns.get_loc(cpse_col)])
                         if cpse_col is not None else "")
                rows.append((measure, owner, value))

            # The basis is the unit most of the family's priced rows are in.
            # Everything below is computed inside it; the rest are counted and
            # reported, never converted.
            by_measure: dict[str, list] = {}
            for measure, owner, value in rows:
                by_measure.setdefault(measure, []).append((owner, value))
            if by_measure:
                basis_uom = max(by_measure, key=lambda m: len(by_measure[m]))
                basis_rows = by_measure[basis_uom]
                other_priced = len(rows) - len(basis_rows)
                uom_forms = len(by_measure)
            else:
                basis_uom, basis_rows, other_priced, uom_forms = None, [], 0, 0

            values = [v for _, v in basis_rows]
            unit_prices = [(o, v) for o, v in basis_rows if o]
            priced = len(values)
            if priced:
                values.sort()
                spend = sum(values)
                low, high = values[0], values[-1]
                mid = priced // 2
                median_price = values[mid] if priced % 2 else (values[mid - 1] + values[mid]) / 2
                # Two figures, deliberately. Against the lowest price ever
                # paid is the theoretical ceiling — real but unreachable,
                # since one unit's bulk rate is not available to everyone.
                # Against the median is the defensible one: it counts only
                # what was paid ABOVE the typical price for the same part,
                # which is the number worth putting in front of an officer.
                savings = spend - low * priced
                excess = sum(v - median_price for v in values if v > median_price)

                # The actionable finding is not "this part varies in price" —
                # it is "THIS unit pays more than everyone else for it".
                # Compare each unit's own median against the family median;
                # a unit sitting well above it, on enough rows to not be a
                # one-off, is a specific question for a specific office.
                by_unit: dict[str, list[float]] = {}
                for unit_name, value in unit_prices:
                    by_unit.setdefault(unit_name, []).append(value)
                for unit_name, unit_values in by_unit.items():
                    unit_values.sort()
                    k = len(unit_values)
                    if k < MIN_ROWS_FOR_UNIT_OUTLIER:
                        continue
                    half = k // 2
                    unit_median = unit_values[half] if k % 2 else (unit_values[half - 1] + unit_values[half]) / 2
                    ratio = unit_median / median_price if median_price else 1.0
                    unit_medians.append((unit_name, round(unit_median, 2), k, round(ratio, 3)))
                    if ratio > worst_ratio:
                        worst_ratio, worst_unit = ratio, unit_name
                        # What that unit would not have spent at the family median.
                        worst_excess = sum(v - median_price for v in unit_values if v > median_price)
                unit_medians.sort(key=lambda row: -row[3])

        cluster_rows.append(
            {
                "Family_ID": family_id,
                "Canonical_Code": canonical_code,
                "Golden_Record": golden_record,
                "Members": len(cluster),
                "Units": len(units),
                "Unit_List": "|".join(units),
                "Unit_Counts": "|".join(f"{name}:{count}" for name, count in ordered_units),
                "Mean_Similarity": round(mean_score, 4),
                "Min_Similarity": round(min_score, 4),
                "Status": status,
                "Priced_Records": priced,
                "Spend": round(spend, 2) if spend is not None else None,
                "Price_Low": round(low, 4) if low is not None else None,
                "Price_High": round(high, 4) if high is not None else None,
                "Price_Median": round(median_price, 4) if median_price is not None else None,
                "Price_Spread": round(high / low, 2) if low else None,
                "Savings": round(savings, 2) if savings is not None else None,
                "Excess": round(excess, 2) if excess is not None else None,
                "Unit_Medians": "|".join(
                    f"{name}:{value}:{count}:{ratio}" for name, value, count, ratio in unit_medians
                ),
                "Outlier_Unit": worst_unit,
                "Outlier_Ratio": round(worst_ratio, 3) if worst_unit else None,
                "Outlier_Excess": round(worst_excess, 2) if worst_unit else None,
                # Which unit every figure above is measured in, and how much of
                # the family was left out of it. A spread figure without this
                # is a number whose units nobody can state.
                "Basis_UOM": basis_uom,
                "Other_UOM_Records": other_priced,
                "UOM_Forms": uom_forms,
                "UOM_List": "|".join(
                    f"{m}:{len(v)}" for m, v in
                    sorted(by_measure.items(), key=lambda kv: -len(kv[1]))
                ),
            }
        )

        if family_number % 500 == 0:
            report(3, family_number / n_clusters, f"{family_number:,} of {n_clusters:,} families")

    records = cleaned.iloc[row_indices].copy()
    records["Family_ID"] = family_ids
    records["Canonical_Code"] = canonical_codes
    records["Golden_Record"] = goldens
    records["Semantic_Similarity"] = similarities
    records["Confidence"] = confidences

    if cpse_col is not None:
        records["CPSE"] = records[cpse_col].astype(str)
    else:
        records["CPSE"] = "UNSPECIFIED"

    if code_col is not None:
        records["Original_Material_Code"] = records[code_col].astype(str)
    else:
        records["Original_Material_Code"] = ""

    keep = [
        "Family_ID", "Canonical_Code", "Original_Description",
        "Normalized_Description", "Golden_Record", "Semantic_Similarity",
        "Confidence", "CPSE", "Original_Material_Code",
    ]
    # Carried so that a family reorganized after the run — a reviewer splitting
    # one apart — can recompute its price spread from its own rows. The cleaned
    # upload is released when harmonize() returns, and without this the price
    # arithmetic would be unrepeatable for any family that changed shape. It is
    # excluded from the CSV export, which keeps the mapping table's published
    # shape exactly as it was.
    if CANONICAL_PRICE in records.columns:
        keep.append(CANONICAL_PRICE)

    # Ground truth rides through under one canonical name when any uploaded
    # file carried it. Excluded from the CSV export, exactly like the price:
    # it is what we score against, not part of the mapping we publish.
    if CANONICAL_LABEL in records.columns and (records[CANONICAL_LABEL].astype(str).str.strip() != "").any():
        keep.append(CANONICAL_LABEL)
    column_reasons["label"] = df.attrs.get("label_notes", [])

    records = records[keep].sort_values(
        by=["Family_ID", "Semantic_Similarity"], ascending=[True, False]
    ).reset_index(drop=True)

    clusters_df = pd.DataFrame(cluster_rows).sort_values("Family_ID").reset_index(drop=True)

    # A proposed code that says what the thing IS, beside the sequence code
    # that says only when it happened to be created. A proposal, exactly like
    # the standardized description: it never replaces Canonical_Code and
    # nothing in the run is keyed on it. See codes.py.
    clusters_df["Proposed_Code"] = codes.propose(
        [str(g) for g in clusters_df["Golden_Record"]]
    )

    # Coverage check. community_detection can leave points unassigned; without
    # this the output simply has fewer rows than the input and nothing says so.
    covered = len(records)
    if covered != len(cleaned):
        warnings.append(
            f"{len(cleaned) - covered:,} cleaned records were not assigned to any family "
            f"and are absent from the output."
        )

    if dropped_junk:
        warnings.append(
            f"{dropped_junk:,} row{'s' if dropped_junk != 1 else ''} "
            f"{'have' if dropped_junk != 1 else 'has'} a placeholder description "
            f"(#N/A, ?, blank) and {'were' if dropped_junk != 1 else 'was'} excluded."
        )

    unspecified = int((records["CPSE"] == UNSPECIFIED).sum())
    if unspecified:
        warnings.append(
            f"{unspecified:,} record{'s' if unspecified != 1 else ''} had no usable source unit."
        )

    report(3, 1.0, "Golden records built")

    collapsed = int(len(records) - len(clusters_df))
    stats = {
        "rows_in": int(rows_in),
        "rows_blank": int(dropped_blank),
        "rows_duplicate": int(dropped_duplicate),
        "rows_junk": int(dropped_junk),
        "records": int(covered),
        "families": int(len(clusters_df)),
        "collapsed": collapsed,
        "fan_in": round(covered / len(clusters_df), 2) if len(clusters_df) else 0.0,
        "units": sorted(records["CPSE"].dropna().unique().tolist()) if cpse_col else [],
        "has_unit_column": cpse_col is not None,
        "has_code_column": code_col is not None,
        "description_column": str(desc_col),
        "unit_column": str(cpse_col) if cpse_col is not None else None,
        "code_column": str(code_col) if code_col is not None else None,
        "column_reasons": column_reasons,
        # What the user's own files called these columns, before renaming.
        "source_columns": df.attrs.get("source_columns", {}),
        "has_price_column": price_loc is not None,
        "has_uom_column": uom_loc is not None,
        "uom": (df.attrs.get("uom_notes") or [{}])[0] if df.attrs.get("uom_notes") else {},
        "price_column": picked["price"] if picked["price"] is not None else None,
        "status_counts": clusters_df["Status"].value_counts().to_dict() if len(clusters_df) else {},
        "review_families": int((clusters_df["Status"].isin(["review", "conflict"])).sum())
        if len(clusters_df)
        else 0,
    }

    return HarmonizationResult(records=records, clusters=clusters_df, stats=stats, warnings=warnings)
