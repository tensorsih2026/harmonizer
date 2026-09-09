"""Unit of measure: the same disease, in the column nobody was reading.

A material master has two vocabularies, not one. The descriptions are the
obvious mess — HEX BOLT, BLT HEX S.S., Bolt Hexagonal — and the whole pipeline
exists to fold them. The unit of measure column is the quiet one: in a real
CPSE extract `EA`, `NOS`, `NO`, `PCS` and `EACH` are five spellings of *each*,
sitting untouched beside descriptions we spent a language model on.

Until now this file's contents were used once, defensively — to stop a column
innocently called "Unit" from being mistaken for the CPSE column. The values
themselves were never read.

That was not just a missing feature. It was a wrong number on screen.

The Savings view compares unit prices inside a family and reports the spread as
excess spend. With units unfolded, a family holds a price per piece from one
CPSE, a price per kilogram from another and a price per metre from a third, and
the "spread" between them is not a procurement finding — it is a category
error. On the sample master every single family carries more than one unit of
measure, so that figure was wrong everywhere it appeared.

What this module does, and deliberately does not do
---------------------------------------------------
It **folds spellings**: NOS, NO, PCS, EACH and EA all become one canonical unit,
by the same principle the descriptions are normalized by.

It **classifies dimension**: count, mass, length, volume, area, time. Two units
in different dimensions describe different purchases, and no amount of string
work makes them comparable.

It **never converts**. Kilograms do not become pieces. You cannot turn a price
per kilo into a price per piece without knowing what one piece weighs, and this
application does not know that and will not guess it. Where a family is priced
in two dimensions the answer is to compare within each and say so — not to
invent a factor and produce a number that looks authoritative and is fiction.
That restraint is the same one that keeps AI attributes out of the export.

Adaptability
------------
The table below is domain knowledge, not a fit to one sample file: it is the
vocabulary of Indian material masters. But no table is complete, so an unknown
token is not discarded. It is cleaned (case, punctuation, plurals, embedded
periods), matched against the known forms, and if it still does not match it
becomes its own canonical unit with dimension "unknown" — visible, counted, and
never silently folded into something it might not be.
"""

from __future__ import annotations

import re

# Dimension -> canonical unit -> the spellings that mean it.
#
# The canonical form is the one a person recognises, not the shortest: "EACH",
# not "EA". This string is shown on screen and written into an export.
FAMILIES: dict[str, dict[str, tuple[str, ...]]] = {
    "count": {
        "EACH": ("EA", "EACH", "EAC", "NO", "NOS", "NUMBER", "NUMBERS", "PC",
                 "PCS", "PIECE", "PIECES", "UNIT", "UNITS", "U", "N"),
        "SET": ("SET", "SETS", "ST", "STS"),
        "PAIR": ("PAIR", "PAIRS", "PR", "PRS"),
        "DOZEN": ("DOZ", "DOZEN", "DZ", "DZN"),
        "PACK": ("PACK", "PACKS", "PKT", "PKTS", "PACKET", "PACKETS", "PKG"),
        "BOX": ("BOX", "BOXES", "BX", "CTN", "CARTON", "CARTONS", "CASE"),
        "ROLL": ("ROLL", "ROLLS", "RL", "COIL", "COILS"),
        "DRUM": ("DRUM", "DRUMS", "DRM", "BARREL", "BARRELS", "BBL"),
        "BAG": ("BAG", "BAGS", "BG", "SACK", "SACKS"),
        "BOTTLE": ("BTL", "BOTTLE", "BOTTLES", "CAN", "CANS", "TIN", "TINS"),
        "TUBE": ("TUBE", "TUBES", "TB"),
        "REAM": ("REAM", "REAMS", "RM"),
    },
    "mass": {
        "KG": ("KG", "KGS", "KILO", "KILOS", "KILOGRAM", "KILOGRAMS", "KGM"),
        "G": ("G", "GM", "GMS", "GRAM", "GRAMS"),
        "TONNE": ("TON", "TONS", "TONNE", "TONNES", "MT", "T"),
        "QUINTAL": ("QTL", "QUINTAL", "QUINTALS"),
    },
    "length": {
        "M": ("M", "MTR", "MTRS", "MTS", "METER", "METERS", "METRE", "METRES"),
        "CM": ("CM", "CMS", "CENTIMETER", "CENTIMETRE"),
        "MM": ("MM", "MMS", "MILLIMETER", "MILLIMETRE"),
        "KM": ("KM", "KMS", "KILOMETER", "KILOMETRE"),
        "FT": ("FT", "FEET", "FOOT"),
        "INCH": ("IN", "INCH", "INCHES", '"'),
        "YARD": ("YD", "YARD", "YARDS", "YDS"),
    },
    "volume": {
        "L": ("L", "LT", "LTR", "LTRS", "LIT", "LITRE", "LITRES", "LITER", "LITERS"),
        "ML": ("ML", "MLS", "MILLILITRE", "MILLILITER"),
        "KL": ("KL", "KLTR", "KILOLITRE", "KILOLITER"),
        "M3": ("M3", "CUM", "CBM", "CUBICMETER", "CUBICMETRE"),
    },
    "area": {
        "M2": ("M2", "SQM", "SQMT", "SQUAREMETER", "SQUAREMETRE"),
        "FT2": ("FT2", "SQFT", "SQUAREFEET", "SQUAREFOOT"),
    },
    "time": {
        "HOUR": ("HR", "HRS", "HOUR", "HOURS"),
        "DAY": ("DAY", "DAYS", "DY"),
        "MONTH": ("MON", "MONTH", "MONTHS", "MTH"),
    },
    "service": {
        "JOB": ("JOB", "JOBS", "LOT", "LS", "LUMPSUM", "LUMPSUMP", "AU", "ACT"),
    },
}

# Flattened lookup, built once.
_LOOKUP: dict[str, tuple[str, str]] = {}
for _dimension, _units in FAMILIES.items():
    for _canonical, _spellings in _units.items():
        for _spelling in _spellings:
            _LOOKUP[_spelling] = (_canonical, _dimension)
        _LOOKUP.setdefault(_canonical, (_canonical, _dimension))

UNKNOWN_DIMENSION = "unknown"

# Values that mean "nobody filled this in". Folded together rather than left as
# four different units, for the same reason UNSPECIFIED exists for CPSE names.
BLANKS = {"", "-", "--", "NA", "NIL", "NONE", "NULL", "UNKNOWN", "?",
          "VALUE", "REF", "DIV0", "0", "XX", "TBD", "BLANK"}
UNSPECIFIED = "UNSPECIFIED"


def _clean(raw) -> str:
    """Upper, unpunctuated, unspaced — the form the lookup is keyed on."""
    if raw is None:
        return ""
    text = str(raw).strip().upper()
    if not text:
        return ""
    # "NOS." and "NO'S" and "NOS ." are all NOS.
    text = re.sub(r"[\s._\-/\\]+", "", text)
    text = text.replace("'", "").replace("(", "").replace(")", "")
    # Spreadsheet error values arrive as #N/A, #VALUE!, #REF! — the slash is
    # already gone by here, so the hash and bang are what is left to strip.
    text = text.strip("#!")
    return text


def normalize(raw) -> tuple[str, str]:
    """One raw cell -> (canonical unit, dimension).

    An unrecognised token keeps its own cleaned form and is marked with the
    unknown dimension. It is never quietly folded into a unit it might not be
    — a wrong fold here would silently make two different purchases look
    comparable, which is the exact failure this module exists to remove.
    """
    text = _clean(raw)
    if not text or text in BLANKS:
        return UNSPECIFIED, UNKNOWN_DIMENSION

    hit = _LOOKUP.get(text)
    if hit:
        return hit

    # Plural that the table does not list: NOSS, SETSS, BOXS.
    if text.endswith("S") and len(text) > 2:
        hit = _LOOKUP.get(text[:-1])
        if hit:
            return hit

    # A trailing period already stripped by _clean; try a leading article or a
    # stray numeric prefix ("1NOS", "10 KG" written as one token).
    stripped = re.sub(r"^\d+", "", text)
    if stripped and stripped != text:
        hit = _LOOKUP.get(stripped)
        if hit:
            return hit
        if stripped.endswith("S"):
            hit = _LOOKUP.get(stripped[:-1])
            if hit:
                return hit

    return text, UNKNOWN_DIMENSION


def summarize(values) -> dict:
    """What folding did to a whole column, for the ingestion note.

    Reported the way the run reports everything else: what came in, what it
    became, and how much of it the tool could not account for."""
    raw_seen: dict[str, int] = {}
    canon_seen: dict[str, int] = {}
    dimensions: dict[str, int] = {}
    unknown: dict[str, int] = {}

    for value in values:
        cleaned = _clean(value)
        raw_seen[cleaned] = raw_seen.get(cleaned, 0) + 1
        canonical, dimension = normalize(value)
        canon_seen[canonical] = canon_seen.get(canonical, 0) + 1
        dimensions[dimension] = dimensions.get(dimension, 0) + 1
        if dimension == UNKNOWN_DIMENSION and canonical != UNSPECIFIED:
            unknown[canonical] = unknown.get(canonical, 0) + 1

    return {
        "raw_forms": len(raw_seen),
        "canonical_forms": len(canon_seen),
        "collapsed": max(0, len(raw_seen) - len(canon_seen)),
        "counts": dict(sorted(canon_seen.items(), key=lambda kv: -kv[1])),
        "dimensions": dict(sorted(dimensions.items(), key=lambda kv: -kv[1])),
        "unrecognised": dict(sorted(unknown.items(), key=lambda kv: -kv[1])[:12]),
    }


def dimension_of(canonical: str) -> str:
    """Dimension for an already-canonical unit."""
    return _LOOKUP.get(str(canonical).upper(), (canonical, UNKNOWN_DIMENSION))[1]


def comparable(a: str, b: str) -> bool:
    """Can two prices in these units be put beside each other at all?

    Same canonical unit: yes. Anything else: no — INCLUDING two units of the
    same dimension, because this module does not convert. A price per metre and
    a price per foot are both lengths and still are not the same number.
    """
    return str(a).upper() == str(b).upper()
