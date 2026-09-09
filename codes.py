"""A code that says what the thing is.

`MTL-0000001` is a counter. It is assigned in whatever order the clustering
happened to produce families, and it carries no information: look at
`MTL-0004521` and you cannot tell whether it is a bolt, a gasket or a safety
helmet. Worse, it is not stable — add one more CPSE's file, re-run, and the
same bolt family may come out as `MTL-0000007`. A code that changes when you
re-run is not a code. It is a row number wearing a code's clothes.

The problem statement asks for standardization *and* harmonization. Folding a
thousand spellings into one canonical record is the harmonization half. Issuing
those records a coherent identifier is the other half, and this is it:

    MTL-BOLT-0001    MTL-BOLT-0002    MTL-NUT-0001    MTL-GSKT-0001

Sorted, like things sit together. A storekeeper reads the class before reading
the description. A new item slots into its own block instead of onto the end of
one long list.

How the category is chosen, without hard-coding a taxonomy
----------------------------------------------------------
The naive move is a keyword list — "if it contains BOLT, it is a bolt" — which
works on the file it was written against and nothing else. This uses the corpus
instead.

A category word is one that many DIFFERENT families share. `BOLT` appears in
hundreds of families; `M10X50` appears in one. So every family's tokens are
scored by how many families use them, and the winner is the family's category.
The vocabulary is therefore discovered from whatever was uploaded, in whatever
language or abbreviation style that organisation happens to use, and a file
full of instrumentation gets instrumentation categories without anybody
teaching it what a transmitter is.

Where the optional AI pass has run, its extracted `item` attribute overrides
the heuristic — it is a better answer to the same question, and it is the same
restraint as everywhere else: used when present, never required.

This is a PROPOSAL
------------------
Real CPSEs have prescribed coding schemes, and this is not one of them. The
proposed code sits beside the working code, exactly as the standardized
description sits beside the golden record. It is in the audited export and on
screen; it never replaces `Canonical_Code`, and nothing in the run is keyed on
it. Presenting an invented scheme as the organisation's own would be the same
class of dishonesty as an invented price.
"""

from __future__ import annotations

import re

PREFIX = "MTL"

# Words that are never a category: they describe, qualify or measure.
STOPWORDS = {
    "AND", "OR", "OF", "FOR", "WITH", "TO", "THE", "A", "AN", "IN", "ON",
    "TYPE", "SIZE", "GRADE", "CLASS", "MAKE", "MODEL", "SERIES", "SET",
    "NEW", "OLD", "OBSOLETE", "SPARE", "SPARES", "ASSY", "ASSEMBLY",
    "HEAVY", "LIGHT", "SMALL", "LARGE", "LONG", "SHORT", "STD", "STANDARD",
    "MM", "CM", "INCH", "NB", "OD", "ID", "DIA", "THK", "LG",
}

# Modifiers: what a thing is MADE OF and what SHAPE it is. These are the words
# that beat class nouns on raw frequency and are exactly the wrong answer.
#
# On the sample master STAINLESS and STEEL each appear in 22 families —
# bolts, nuts, studs, washers — so pure corpus frequency confidently files a
# nut under STEEL. A material is not a class: everything in a refinery is
# made of steel.
#
# This is a MODIFIER list, not a taxonomy. It says what a category is not; it
# never says what one is. The class nouns themselves — BOLT, GASKET,
# TRANSMITTER, whatever this particular organisation buys — are still
# discovered from the file, which is the part that has to adapt.
MODIFIERS = {
    # materials
    "STAINLESS", "STEEL", "SS", "SS304", "SS316", "MS", "CS", "GI", "CI",
    "BRASS", "BRONZE", "COPPER", "ALUMINIUM", "ALUMINUM", "ALLOY", "IRON",
    "RUBBER", "NITRILE", "NEOPRENE", "PTFE", "TEFLON", "NYLON", "PVC",
    "HDPE", "LDPE", "GRAPHITE", "CERAMIC", "GLASS", "PLASTIC", "CARBON",
    "GALVANISED", "GALVANIZED", "CHROME", "NICKEL", "ZINC", "TITANIUM",
    # form and finish
    "HEX", "HEXAGON", "HEXAGONAL", "HEXAGONL", "SQUARE", "ROUND", "FLAT",
    "SPIRAL", "SPRL", "WOUND", "WND", "SW", "THREADED", "PLAIN", "SLOTTED",
    "COUNTERSUNK", "CSK", "PAN", "CAP", "SOCKET", "HD", "HEAD",
    "FULL", "HALF", "DOUBLE", "SINGLE", "HEAVYDUTY", "WHITE", "BLACK",
    "YELLOW", "BLUE", "RED", "GREEN", "RATCHET",
}
STOPWORDS |= MODIFIERS

# A token has to appear in at least this many families before it can be a
# class. One family sharing a word with nobody is describing itself, not
# naming a category.
MIN_FAMILIES = 2

# A token that is mostly digits, or a dimension, is a size and not a class.
_SIZEISH = re.compile(r"^[0-9]|[0-9]{2,}|^M[0-9]|X[0-9]")

MIN_TOKEN = 2
MAX_TOKEN = 5          # how many characters the slug keeps
FALLBACK = "GEN"       # a family whose description yields nothing usable


def _tokens(text: str) -> list[str]:
    parts = re.split(r"[^A-Za-z0-9]+", str(text or "").upper())
    out = []
    for part in parts:
        if len(part) < MIN_TOKEN or part in STOPWORDS:
            continue
        if _SIZEISH.search(part):
            continue
        out.append(part)
    return out


def _slug(word: str) -> str:
    """A short, readable class token. Vowels go first when it must be cut."""
    word = re.sub(r"[^A-Z0-9]", "", str(word).upper())
    if not word:
        return FALLBACK
    if len(word) <= MAX_TOKEN:
        return word
    # BEARING -> BRNG, GASKET -> GSKT: how these are already abbreviated in
    # every material master anyone has ever seen.
    head, tail = word[0], re.sub(r"[AEIOU]", "", word[1:])
    squeezed = (head + tail)[:MAX_TOKEN]
    return squeezed if len(squeezed) >= MIN_TOKEN else word[:MAX_TOKEN]


def categories(goldens: list[str], items: dict[int, str] | None = None) -> list[str]:
    """One category per family, discovered from the corpus.

    `items` maps a family's position to the AI pass's extracted item name,
    where it ran. That is a better answer to the same question, so it wins."""
    items = items or {}
    token_lists = [_tokens(g) for g in goldens]

    # How many families use each token. A word shared across many families is
    # a class; a word unique to one is a specification.
    family_count: dict[str, int] = {}
    for tokens in token_lists:
        for token in set(tokens):
            family_count[token] = family_count.get(token, 0) + 1

    out = []
    for position, tokens in enumerate(token_lists):
        override = str(items.get(position, "") or "").strip()
        if override:
            out.append(_slug(override.split()[0]))
            continue
        if not tokens:
            out.append(FALLBACK)
            continue
        # The EARLIEST token that several families share. Position matters
        # because a material description leads with what the thing is; the
        # frequency floor is what stops a one-off spelling becoming a class of
        # its own. Raw frequency alone was tried and is wrong — it ranks
        # STAINLESS above NUT, because everything in a refinery is stainless.
        pick = next(
            (t for t in tokens if family_count.get(t, 0) >= MIN_FAMILIES),
            None,
        )
        out.append(_slug(pick if pick else tokens[0]))
    return out


def propose(goldens: list[str], items: dict[int, str] | None = None) -> list[str]:
    """Proposed codes for a whole run, in the order the families were given.

    Numbering runs inside a category and follows the description alphabetically
    rather than clustering order, so the same input produces the same codes
    whichever order the families happened to come out in. That is the whole
    point: a code that moves when you re-run is not an identifier."""
    cats = categories(goldens, items)

    grouped: dict[str, list[int]] = {}
    for position, category in enumerate(cats):
        grouped.setdefault(category, []).append(position)

    proposed = [""] * len(goldens)
    for category, positions in grouped.items():
        ordered = sorted(positions, key=lambda i: (str(goldens[i]).upper(), i))
        for rank, position in enumerate(ordered, start=1):
            proposed[position] = f"{PREFIX}-{category}-{rank:04d}"
    return proposed


def summary(proposed: list[str]) -> dict:
    """What the scheme looks like as a whole, for the panel that explains it."""
    counts: dict[str, int] = {}
    for code in proposed:
        bits = str(code).split("-")
        if len(bits) == 3:
            counts[bits[1]] = counts.get(bits[1], 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "classes": len(counts),
        "largest": ordered[:12],
        "generic": counts.get(FALLBACK, 0),
        "coded": sum(1 for c in proposed if c),
    }
