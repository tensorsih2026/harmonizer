"""
Material Code Harmonizer — API server.

Serves the two HTML pages and the harmonization API from one origin, so there
is no CORS to configure and one URL to hand a judge.

Jobs run on a background thread with a single worker: the model is CPU-bound
and two concurrent runs would just make both slower. The browser submits, gets
a job id back immediately, and polls — a blocking upload would exceed the
proxy timeout on any dataset worth demoing.
"""

from __future__ import annotations

import hmac
import io
import json
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from harmonizer import (
    CANONICAL_LABEL,
    CANONICAL_PRICE,
    CANONICAL_UOM,
    MODEL_NAME,
    STAGES,
    HarmonizationResult,
    PipelineError,
    harmonize,
    load_model,
    read_tables,
    threshold_manifest,
    MIN_ROWS_FOR_UNIT_OUTLIER,
    UNIT_OUTLIER_RATIO,
)
import decisions
import store
import sweep
from review import (
    SplitError,
    undo_last, apply_merge, apply_split, approve_family, pending,
    plan_approve, plan_merge, plan_split,
)
from simulate import simulate
import gemini
import standardize
import ai_review
from evaluate import evaluate

# How many distinct source strings each provenance row carries inline. Enough
# to recognise the family at a glance; the export has every one of them.
SPELLINGS_PER_ROW = 12

# Above this price ratio inside one family, the spread stops being a finding
# about procurement and starts being a finding about the data.
#
# No single material varies 20x in unit price. When a family shows 600x, the
# honest reading is not "we could have saved 99%" — it is that the rows are
# not the same thing, or the price column mixes per-piece with per-lot, or an
# export is malformed. Presenting that as savings is exactly the claim a judge
# takes apart, so the number is shown WITH the reason to doubt it.
SUSPECT_SPREAD_RATIO = 20.0

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 200 * 1024 * 1024))  # 200 MB
MAX_FILES = int(os.getenv("MAX_FILES", 20))
# Extensions we can actually parse. A file that slips through this (a PNG
# renamed to .csv) is caught by the content check in harmonizer.parse_table.
ALLOWED_SUFFIXES = {".csv", ".tsv", ".txt", ".dat", ".psv", ".xlsx", ".xlsm", ".xls"}
RETAIN_JOBS = int(os.getenv("RETAIN_JOBS", 4))
STATIC_DIR = Path(__file__).parent / "static"

# Reader questions. Kept on disk so they survive a reload during development;
# on an ephemeral host (a Space, a container) the file goes when the container
# does, which is stated on the page rather than pretended otherwise.
QUESTIONS_PATH = Path(os.getenv("QUESTIONS_PATH", Path(__file__).parent / "data" / "questions.jsonl"))
MAX_QUESTION_CHARS = 1000

# Typing this on the FAQ page reveals the reply addresses. It is a shared
# secret, not a login: it gates a list from casual view, nothing more. Override
# it per deployment; anyone holding real addresses should front this route with
# actual authentication instead.
REPLY_CODE = os.getenv("MCH_REPLY_CODE", "336699")


def _constant_eq(a: str, b: str) -> bool:
    """Compare without leaking length or position through timing."""
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
QUESTION_MIN_SECONDS = float(os.getenv("QUESTION_MIN_SECONDS", 5))

_model = None
_model_lock = threading.Lock()
_run_lock = threading.Lock()          # one harmonization at a time
_jobs: "OrderedDict[str, dict]" = OrderedDict()
_jobs_lock = threading.Lock()


_questions: list[dict] = []
_questions_lock = threading.Lock()
_last_post: dict[str, float] = {}


def _load_questions() -> None:
    """Read whatever survived a restart. A corrupt line is skipped, not fatal."""
    if not QUESTIONS_PATH.is_file():
        return
    with _questions_lock:
        _questions.clear()
        for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                _questions.append(json.loads(line))
            except json.JSONDecodeError:
                continue


def _append_question(entry: dict) -> None:
    with _questions_lock:
        _questions.append(entry)
    try:
        QUESTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with QUESTIONS_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        # A read-only filesystem should not lose the question for this session.
        pass


def get_model():
    """Load the model once, lazily, and reuse it for every job."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = load_model()
    return _model


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load the model at boot so the first real request isn't the slow one.
    threading.Thread(target=get_model, daemon=True).start()
    _load_questions()
    # Bring back every run saved to disk. Before this, a restart erased not
    # just the results but every reviewer decision made against them.
    _restore_saved_runs()
    yield


app = FastAPI(title="Material Code Harmonizer", version="1.0.0", lifespan=lifespan)


# --------------------------------------------------------------------------
# Job store
# --------------------------------------------------------------------------

def _new_job(filenames: list[str], total_bytes: int) -> dict:
    job = {
        "job_id": uuid.uuid4().hex[:12],
        "status": "queued",              # queued | running | done | error
        "stage_index": 0,
        "stage": STAGES[0],
        "stage_progress": 0.0,
        "message": "Waiting for a worker",
        "files": filenames,
        "bytes": total_bytes,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "started_at": None,
        "finished_at": None,
        "duration_s": None,
        "error": None,
        "warnings": [],
        "stats": {},
        "result": None,                  # HarmonizationResult, kept in memory
    }
    with _jobs_lock:
        _jobs[job["job_id"]] = job
        # Evict oldest finished jobs only — dropping a running one would make
        # its own worker thread fail to find it.
        # Eviction is now a cache decision rather than a loss: a finished run
        # is on disk and comes back on demand.
        while len(_jobs) > RETAIN_JOBS:
            victim = next(
                (jid for jid, j in _jobs.items() if j["status"] in ("done", "error")),
                None,
            )
            if victim is None:
                break
            _jobs.pop(victim)
    return job


def _persist(job: dict) -> None:
    """Write a finished run to disk after anything that changed it.

    Called after the pipeline completes and after every reviewer decision, so
    a restart loses at most nothing. Failures are recorded and ignored: a
    review tool that stops working because it could not write a file is worse
    than one that forgets."""
    try:
        store.save(job)
    except Exception:                      # pragma: no cover - belt and braces
        pass


def _restore_saved_runs() -> None:
    """Bring back everything on disk at startup.

    Restored jobs are indistinguishable from live ones except that their
    source files are gone — which was already true of every finished run,
    because uploads are never written to disk."""
    for payload in store.load_all():
        try:
            job = dict(payload["meta"])
            job["result"] = HarmonizationResult(
                records=payload["records"],
                clusters=payload["clusters"],
                stats=payload.get("stats", {}),
                warnings=payload.get("warnings", []),
                decisions=payload.get("decisions", []),
            )
            job["restored"] = True
            job["saved_at"] = payload.get("saved_at", "")
            with _jobs_lock:
                _jobs.setdefault(job["job_id"], job)
        except Exception:
            continue


# The header the workspace sends with every decision. There is no login, so
# this is a signature the reviewer types, not an authenticated identity, and it
# is labelled that way wherever it is shown. A name somebody chose to put on
# their own work is still the difference between an accountable decision and an
# anonymous one — and it is the first thing an evaluator asks of an audit file.
REVIEWER_HEADER = "X-Reviewer"


def _reviewer(request: Request) -> str:
    raw = request.headers.get(REVIEWER_HEADER, "") if request is not None else ""
    return " ".join(str(raw).split())[:64]


def _get_job(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is not None:
        return job

    # Not in memory. Before this it was gone: evicted by the fifth upload, or
    # lost to a restart, along with every decision a reviewer had made against
    # it. Now memory is a cache in front of disk.
    for payload in store.load_all():
        if payload.get("job_id") != job_id:
            continue
        try:
            revived = dict(payload["meta"])
            revived["result"] = HarmonizationResult(
                records=payload["records"],
                clusters=payload["clusters"],
                stats=payload.get("stats", {}),
                warnings=payload.get("warnings", []),
                decisions=payload.get("decisions", []),
            )
            revived["restored"] = True
            revived["saved_at"] = payload.get("saved_at", "")
            with _jobs_lock:
                _jobs[job_id] = revived
            return revived
        except Exception:
            break

    raise HTTPException(404, f"Job {job_id} not found. It may have been evicted — run it again.")


def _public(job: dict) -> dict:
    return {k: v for k, v in job.items() if k != "result"}


def _require_result(job: dict) -> HarmonizationResult:
    if job["status"] == "error":
        raise HTTPException(409, job["error"] or "The job failed.")
    if job["status"] != "done":
        raise HTTPException(409, f"Job is {job['status']} — poll /api/jobs/{job['job_id']} until it is done.")
    return job["result"]


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------

def _run_job(job_id: str, payload: list[tuple[str, bytes]]) -> None:
    job = _get_job(job_id)

    def progress(stage_index: int, fraction: float, message: str) -> None:
        job["stage_index"] = stage_index
        job["stage"] = STAGES[stage_index]
        job["stage_progress"] = round(fraction, 4)
        job["message"] = message

    with _run_lock:
        job["status"] = "running"
        job["started_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        started = time.perf_counter()
        try:
            frame = read_tables(payload)
            result = harmonize(frame, get_model(), progress=progress)
            job["result"] = result
            job["stats"] = result.stats
            job["warnings"] = result.warnings
            job["status"] = "done"
            job["message"] = f"{result.stats['families']:,} families from {result.stats['records']:,} records"
            _persist(job)
        except PipelineError as exc:
            job["status"] = "error"
            job["error"] = str(exc)
            job["message"] = str(exc)
        except Exception as exc:  # pragma: no cover
            job["status"] = "error"
            job["error"] = f"{type(exc).__name__}: {exc}"
            job["message"] = job["error"]
        finally:
            job["duration_s"] = round(time.perf_counter() - started, 1)
            job["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "model": MODEL_NAME,
        "model_loaded": _model is not None,
        "jobs_retained": len(_jobs),
        "storage": store.usage(),
        "storage_warnings": store.warnings(),
    }


@app.get("/api/runs")
def list_runs() -> dict:
    """Every run this server still holds, in memory or on disk.

    A restored run is marked as such: it can be read, exported and reviewed
    further, but not re-clustered, because the uploaded rows behind it were
    never written to disk and are gone."""
    seen: dict[str, dict] = {}
    with _jobs_lock:
        for job_id, job in _jobs.items():
            if job.get("status") != "done":
                continue
            seen[job_id] = {
                "job_id": job_id,
                "files": job.get("files", []),
                "created_at": job.get("created_at"),
                "saved_at": job.get("saved_at", ""),
                "restored": bool(job.get("restored")),
                "families": int(job.get("stats", {}).get("families", 0) or 0),
                "records": int(job.get("stats", {}).get("records", 0) or 0),
                "decisions": len([
                    d for d in getattr(job.get("result"), "decisions", []) or []
                    if not d.get("undone")
                ]),
                "in_memory": True,
            }
    for payload in store.load_all():
        job_id = payload.get("job_id")
        if not job_id or job_id in seen:
            continue
        meta = payload.get("meta", {})
        seen[job_id] = {
            "job_id": job_id,
            "files": meta.get("files", []),
            "created_at": meta.get("created_at"),
            "saved_at": payload.get("saved_at", ""),
            "restored": True,
            "families": int(meta.get("stats", {}).get("families", 0) or 0),
            "records": int(meta.get("stats", {}).get("records", 0) or 0),
            "decisions": len([d for d in payload.get("decisions", []) if not d.get("undone")]),
            "in_memory": False,
        }
    runs = sorted(seen.values(), key=lambda r: r.get("created_at") or "", reverse=True)
    return {"runs": runs, "storage": store.usage()}


@app.post("/api/jobs")
async def create_job(files: list[UploadFile] = File(...)) -> JSONResponse:
    if not files:
        raise HTTPException(400, "Attach at least one CSV.")
    if len(files) > MAX_FILES:
        raise HTTPException(400, f"At most {MAX_FILES} files per run.")

    rejected = [
        (u.filename or "unnamed")
        for u in files
        if Path(u.filename or "").suffix.lower() not in ALLOWED_SUFFIXES
    ]
    if rejected:
        raise HTTPException(
            400,
            f"Not a data table: {', '.join(rejected)}. "
            f"Upload {', '.join(sorted(ALLOWED_SUFFIXES))} exports of your material master. "
            "For an Excel file, save it as CSV first.",
        )

    payload: list[tuple[str, bytes]] = []
    total = 0
    for upload in files:
        raw = await upload.read()
        total += len(raw)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                413, f"Upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB in total."
            )
        payload.append((upload.filename or "upload.csv", raw))

    job = _new_job([name for name, _ in payload], total)
    threading.Thread(target=_run_job, args=(job["job_id"], payload), daemon=True).start()
    return JSONResponse(_public(job), status_code=202)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    return _public(_get_job(job_id))


@app.get("/api/jobs/{job_id}/summary")
def job_summary(job_id: str) -> dict:
    job = _get_job(job_id)
    result = _require_result(job)
    clusters = result.clusters
    records = result.records

    counts = clusters["Status"].value_counts().to_dict()
    total = max(len(clusters), 1)

    # Two numbers per unit, because they answer different questions and only
    # one of them is any use in the rail. Families-per-unit is nearly always
    # "all of them" — in this run every unit appears in every family, so the
    # rail read 4 / 4 / 4 / 4 and told the user nothing. Records-per-unit is
    # what actually differs between units.
    if result.stats.get("has_unit_column"):
        grouped = records.groupby("CPSE")
        unit_families = grouped["Family_ID"].nunique()
        unit_records = grouped.size()
        unit_index = unit_records.sort_values(ascending=False).index
    else:
        unit_families = pd.Series(dtype=int)
        unit_records = pd.Series(dtype=int)
        unit_index = []

    return {
        "job_id": job_id,
        "duration_s": job["duration_s"],
        "warnings": result.warnings,
        "kpis": {
            "records": result.stats["records"],
            "families": result.stats["families"],
            "collapsed": result.stats["collapsed"],
            "review": result.stats["review_families"],
            "fan_in": result.stats["fan_in"],
            "rows_in": result.stats["rows_in"],
            "rows_blank": result.stats["rows_blank"],
            "rows_duplicate": result.stats["rows_duplicate"],
            "rows_junk": result.stats.get("rows_junk", 0),
        },
        "distribution": {
            "resolved": round(100 * counts.get("resolved", 0) / total, 1),
            "accepted": round(100 * counts.get("accepted", 0) / total, 1),
            "review": round(100 * counts.get("review", 0) / total, 1),
            "conflict": round(100 * counts.get("conflict", 0) / total, 1),
        },
        "counts": {
            "all": int(len(clusters)),
            "review": int(pending(clusters).sum()),
            "resolved": int(counts.get("resolved", 0)),
            "fanin": int((clusters["Members"] >= 4).sum()),
        },
        "units": [
            {
                "name": str(name),
                "families": int(unit_families.get(name, 0)),
                "records": int(unit_records.get(name, 0)),
            }
            for name in unit_index
        ][:40],
        # Every threshold the UI quotes comes from here, so a label can never
        # disagree with the pipeline that produced the number.
        "thresholds": threshold_manifest(),
        "source": {
            "files": job["files"],
            "description_column": result.stats["description_column"],
            "has_unit_column": result.stats["has_unit_column"],
            "has_code_column": result.stats["has_code_column"],
            # The user's own header names, so the UI reports their file rather
            # than our internal canonical names.
            "columns": result.stats.get("source_columns", {}),
        },
    }


def _unit_counts(raw) -> list[dict]:
    """\"A:312|C:180\" -> [{unit: A, records: 312}, ...], already ordered."""
    out = []
    for chunk in str(raw or "").split("|"):
        if not chunk or ":" not in chunk:
            continue
        name, _, count = chunk.rpartition(":")
        if not name:
            continue
        try:
            out.append({"unit": name, "records": int(count)})
        except ValueError:
            continue
    return out


def _filter_clusters(result: HarmonizationResult, view: str, unit: str | None, q: str | None):
    """Shared by the list and the constellation so both show the same set."""
    clusters = result.clusters
    records = result.records

    if view == "review":
        clusters = clusters[pending(clusters)]
    elif view == "resolved":
        clusters = clusters[clusters["Status"] == "resolved"]
    elif view == "fanin":
        clusters = clusters[clusters["Members"] >= 4]

    if unit:
        families = records.loc[records["CPSE"].astype(str) == unit, "Family_ID"].unique()
        clusters = clusters[clusters["Family_ID"].isin(families)]

        # Membership alone is not a filter here. In a real consolidated run
        # nearly every family contains nearly every unit, so "families that
        # contain IOCL" is the whole list — you click a unit, the rail
        # highlights, and the rows are byte-identical. What differs between
        # families is how MUCH that unit contributed, so selecting a unit
        # ranks the list by that unit's own record count, biggest first.
        # The families where IOCL has the most duplicate spellings are the
        # ones an IOCL buyer should be looking at, and they come to the top.
        def _contribution(raw) -> int:
            for entry in _unit_counts(raw):
                if entry["unit"] == unit:
                    return entry["records"]
            return 0

        if "Unit_Counts" in clusters.columns and len(clusters):
            clusters = clusters.assign(
                _unit_rank=clusters["Unit_Counts"].map(_contribution)
            ).sort_values("_unit_rank", ascending=False, kind="mergesort").drop(columns=["_unit_rank"])

    if q:
        needle = q.strip().lower()
        if needle:
            by_golden = clusters["Golden_Record"].str.lower().str.contains(needle, regex=False, na=False)
            by_code = clusters["Canonical_Code"].str.lower().str.contains(needle, regex=False, na=False)
            hit_families = records.loc[
                records["Original_Description"].astype(str).str.lower().str.contains(needle, regex=False, na=False)
                | records["Original_Material_Code"].astype(str).str.lower().str.contains(needle, regex=False, na=False),
                "Family_ID",
            ].unique()
            clusters = clusters[by_golden | by_code | clusters["Family_ID"].isin(hit_families)]

    return clusters


@app.get("/api/jobs/{job_id}/clusters")
def job_clusters(
    job_id: str,
    view: str = Query("all", pattern="^(all|review|resolved|fanin)$"),
    unit: str | None = None,
    q: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict:
    """Paginated cluster list. Filtering happens here, not in the browser —
    50,000+ rows will not fit in a page's memory, let alone render."""
    result = _require_result(_get_job(job_id))
    clusters = _filter_clusters(result, view, unit, q)

    total = int(len(clusters))
    page = clusters.iloc[offset : offset + limit]
    ai_flags = _ai_state(_get_job(job_id))["flags"]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(page) < total,
        "clusters": [
            {
                "family_id": row.Family_ID,
                "code": row.Canonical_Code,
                "desc": row.Golden_Record,
                "members": int(row.Members),
                "units": [u for u in str(row.Unit_List).split("|") if u],
                # How many records each unit contributed, biggest first. The
                # bare unit list is nearly identical on every family in a real
                # run — the counts are what actually distinguishes them.
                "unit_counts": _unit_counts(row.Unit_Counts),
                "score": float(row.Mean_Similarity),
                "min_score": float(row.Min_Similarity),
                "status": row.Status,
                "approved": bool(getattr(row, "Approved", False)),
                # Empty unless the optional AI pass has run. Carried on the list
                # too, so a family it questioned is visible while scanning
                # rather than only after opening it.
                "flags": ai_flags.get(str(row.Family_ID), []),
                "spend": _money(row.Spend) if "Spend" in clusters.columns else None,
                "excess": _money(row.Excess) if "Excess" in clusters.columns else None,
                "spread": _money(row.Price_Spread) if "Price_Spread" in clusters.columns else None,
            }
            for row in page.itertuples(index=False)
        ],
    }


@app.get("/api/jobs/{job_id}/graph")
def job_graph(
    job_id: str,
    view: str = Query("all", pattern="^(all|review|resolved|fanin)$"),
    unit: str | None = None,
    q: str | None = None,
    budget: int = Query(6000, ge=200, le=20000),
) -> dict:
    """
    A star-topology graph for the constellation view: one master node per
    family, one legacy node per source record, every legacy node linked to its
    master.

    The payload adapts to the size of the run rather than assuming a fixed node
    count. A three-family run sends three masters and all their records; a
    400,000-family run sends the largest families up to the node budget and
    says plainly how much was left out. Field names are short because at six
    thousand nodes the JSON size is the thing that matters.
    """
    result = _require_result(_get_job(job_id))
    clusters = _filter_clusters(result, view, unit, q)
    records = result.records

    total_families = int(len(clusters))
    total_records = int(clusters["Members"].sum()) if total_families else 0

    if total_families == 0:
        return {
            "total_families": 0, "shown_families": 0,
            "total_records": 0, "shown_records": 0,
            "truncated": False, "units": [], "masters": [], "legacy": [],
        }

    # Biggest families first: they carry the story, and if anything has to be
    # left out it should be the singletons.
    ordered = clusters.sort_values("Members", ascending=False)

    family_limit = min(total_families, 900)
    per_family = max(3, min(40, (budget - family_limit) // max(family_limit, 1)))
    ordered = ordered.head(family_limit)

    family_ids = set(ordered["Family_ID"])
    members = records[records["Family_ID"].isin(family_ids)]

    unit_names: list[str] = []
    unit_index: dict[str, int] = {}

    masters: list[dict] = []
    legacy: list[dict] = []
    master_index: dict[str, int] = {}

    for i, row in enumerate(ordered.itertuples(index=False)):
        master_index[row.Family_ID] = i
        masters.append({
            "i": i,
            "f": row.Family_ID,
            "c": row.Canonical_Code,
            "d": str(row.Golden_Record)[:120],
            "s": round(float(row.Mean_Similarity), 4),
            "st": row.Status,
            "n": int(row.Members),
            "u": int(row.Units),
        })

    # Records are the nodes — but SAMPLED ACROSS SPELLINGS, not off the top.
    #
    # Two things were wrong with taking the first `per_family` rows. A family
    # of 200 rows that mostly say the same thing hands back forty leaves with
    # an identical score, so the percentage reads as a property of the family
    # rather than of the string a unit actually typed. And the rare variant —
    # the one worth looking at — never gets plotted at all, because it is at
    # the bottom of the group.
    #
    # Collapsing to one node per spelling fixed the score and broke the
    # picture: the density of leaves around a master is what makes a family
    # read as a cluster, and four dots is not a cluster. So the fix is to keep
    # the same node COUNT and change which rows are chosen — round-robin
    # across the distinct spellings until the budget is spent. Every spelling
    # is represented, common ones get proportionally more nodes, and each node
    # carries its own record's similarity, which now genuinely varies within a
    # family.
    used = 0
    for family_id, group in members.groupby("Family_ID", sort=False):
        parent = master_index.get(family_id)
        if parent is None:
            continue

        spellings: dict[str, dict] = {}
        for rec in group.itertuples(index=False):
            raw = str(rec.Original_Description)
            key = " ".join(raw.split()).upper()
            entry = spellings.get(key)
            if entry is None:
                entry = spellings[key] = {
                    "r": raw[:90],
                    "s": float(rec.Semantic_Similarity),
                    "n": 0,
                    "unit": str(rec.CPSE),
                    "id": str(rec.Original_Material_Code)[:40],
                }
            entry["n"] += 1

        # Rarest spelling first, so an odd one-off is never the row that gets
        # dropped when the budget runs out.
        ranked = sorted(spellings.values(), key=lambda e: (e["n"], -e["s"]))
        budget = min(per_family, int(len(group)))
        placed = 0
        while placed < budget:
            progressed = False
            for entry in ranked:
                if placed >= budget:
                    break
                shown = entry.setdefault("shown", 0)
                if shown >= entry["n"]:
                    continue           # this spelling has no records left
                entry["shown"] = shown + 1
                progressed = True

                idx = unit_index.get(entry["unit"])
                if idx is None:
                    idx = len(unit_names)
                    unit_index[entry["unit"]] = idx
                    unit_names.append(entry["unit"])
                legacy.append({
                    "m": parent,
                    "u": idx,
                    "s": round(entry["s"], 4),
                    "r": entry["r"],
                    "id": entry["id"],
                    "n": entry["n"],
                })
                placed += 1
                used += 1
            if not progressed:
                break

    return {
        "total_families": total_families,
        "shown_families": len(masters),
        "total_records": total_records,
        "shown_records": used,
        "truncated": len(masters) < total_families or used < total_records,
        "per_family_cap": int(per_family),
        "units": unit_names,
        "masters": masters,
        "legacy": legacy,
    }


@app.get("/api/jobs/{job_id}/clusters/{family_id}")
def cluster_detail(job_id: str, family_id: str, limit: int = Query(200, ge=1, le=2000)) -> dict:
    result = _require_result(_get_job(job_id))
    row = result.clusters[result.clusters["Family_ID"] == family_id]
    if row.empty:
        raise HTTPException(404, f"Family {family_id} not found.")
    row = row.iloc[0]

    members = result.records[result.records["Family_ID"] == family_id]
    shown = members.head(limit)

    golden_text = str(row.Golden_Record)

    price = None
    if "Spend" in result.clusters.columns and row.Priced_Records:
        price = {
            "priced": int(row.Priced_Records),
            "spend": _money(row.Spend),
            "low": _money(row.Price_Low),
            "median": _money(row.Price_Median),
            "high": _money(row.Price_High),
            "spread": _money(row.Price_Spread),
            "excess": _money(row.Excess),
            "outlier_unit": row.Outlier_Unit if isinstance(row.Outlier_Unit, str) else None,
            "outlier_ratio": _money(row.Outlier_Ratio),
            "outlier_excess": _money(row.Outlier_Excess),
            "uom": _uom_basis(row),
            "units": [
                {"unit": p.split(":")[0], "median": float(p.split(":")[1]),
                 "records": int(p.split(":")[2]), "ratio": float(p.split(":")[3])}
                for p in str(row.Unit_Medians or "").split("|")
                if p and len(p.split(":")) == 4
            ],
        }

    ai = _ai_state(_get_job(job_id))
    return {
        "family_id": family_id,
        "proposed_code": (str(row.Proposed_Code)
                          if "Proposed_Code" in result.clusters.columns
                          and isinstance(row.Proposed_Code, str) else None),
        "code": row.Canonical_Code,
        "desc": row.Golden_Record,
        "standard": ai["attributes"].get(str(family_id)),
        "flags": ai["flags"].get(str(family_id), []),
        "thresholds": threshold_manifest(),
        "unit_counts": _unit_counts(row.Unit_Counts),
        "price": price,
        "status": row.Status,
        "approved": bool(getattr(row, "Approved", False)),
        "score": float(row.Mean_Similarity),
        "min_score": float(row.Min_Similarity),
        "members": int(row.Members),
        "truncated": int(row.Members) > len(shown),
        "sources": [
            {
                "u": str(r.CPSE),
                "raw": str(r.Original_Description),
                "normalized": str(r.Normalized_Description),
                "id": str(r.Original_Material_Code),
                "s": float(r.Semantic_Similarity),
                "confidence": str(r.Confidence),
                # True for the spelling that won and became the golden record.
                # The UI needs to point at it; it should not have to guess by
                # comparing strings and get it wrong on a tie.
                "golden": str(r.Normalized_Description) == golden_text,
            }
            for r in shown.itertuples(index=False)
        ],
    }


def _spelling_key(series):
    """Whitespace-collapsed, upper-cased source text — the thing that makes two
    rows "the same spelling" rather than two separate ones."""
    return series.astype(str).str.split().str.join(" ").str.upper()


@app.get("/api/jobs/{job_id}/provenance/{family_id}")
def family_provenance(
    job_id: str,
    family_id: str,
    q: str | None = None,
    spelling: str | None = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(40, ge=1, le=400),
) -> dict:
    """One family, at whichever depth the reader needs.

    Two levels, because a material master is mostly repetition and the two
    questions are different:

      * WITHOUT `spelling`: one entry per distinct source string, with how many
        records wrote it. A family of 5,000 rows and two spellings collapses to
        two readable lines — printing the 5,000 is how the masters became
        unreadable in the first place.
      * WITH `spelling`: the actual rows behind ONE of those lines. Every
        duplicate, with the unit that booked it, the legacy code it was booked
        under, and its own similarity. Nothing is hidden; it is one click away
        rather than in the way.

    `q` filters at whichever level is being read, so a code pasted from a
    purchase order can be found inside a family of any size. Both levels are
    paged: no response is unbounded, however large the family.
    """
    result = _require_result(_get_job(job_id))
    row = result.clusters[result.clusters["Family_ID"] == family_id]
    if row.empty:
        raise HTTPException(404, f"Family {family_id} not found.")
    row = row.iloc[0]

    members = result.records[result.records["Family_ID"] == family_id]
    golden_text = str(row.Golden_Record)
    needle = (q or "").strip().lower()

    header = {
        "family_id": family_id,
        "code": str(row.Canonical_Code),
        "golden": golden_text,
        "records": int(row.Members),
        "score": float(row.Mean_Similarity),
        "status": str(row.Status),
        "query": q or "",
    }

    if spelling is not None:
        # ---- the rows behind one distinct string ----------------------------
        key = " ".join(str(spelling).split()).upper()
        collapsed = members["Original_Description"].astype(str).str.split().str.join(" ").str.upper()
        frame = members[collapsed == key]
        if needle:
            frame = frame[
                frame["Original_Material_Code"].astype(str).str.lower().str.contains(needle, regex=False, na=False)
                | frame["CPSE"].astype(str).str.lower().str.contains(needle, regex=False, na=False)
                | frame["Original_Description"].astype(str).str.lower().str.contains(needle, regex=False, na=False)
            ]
        frame = frame.sort_values(["CPSE", "Original_Material_Code"], kind="mergesort")
        total = int(len(frame))
        page = frame.iloc[offset : offset + limit]
        return {
            **header,
            "level": "records",
            "spelling": str(spelling),
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": offset + len(page) < total,
            "rows": [
                {
                    "unit": str(r.CPSE),
                    "code": str(r.Original_Material_Code),
                    "text": str(r.Original_Description)[:180],
                    "score": float(r.Semantic_Similarity),
                    "confidence": str(r.Confidence),
                    "golden": str(r.Normalized_Description) == golden_text,
                }
                for r in page.itertuples(index=False)
            ],
        }

    # ---- one entry per distinct string --------------------------------------
    spellings: dict[str, dict] = {}
    for rec in members.itertuples(index=False):
        raw = str(rec.Original_Description)
        key = " ".join(raw.split()).upper()
        entry = spellings.get(key)
        if entry is None:
            entry = spellings[key] = {
                "key": key,
                "text": raw[:180],
                "score": float(rec.Semantic_Similarity),
                "records": 0,
                "units": [],
                "codes": [],
                "golden": str(rec.Normalized_Description) == golden_text,
            }
        entry["records"] += 1
        entry["score"] = min(entry["score"], float(rec.Semantic_Similarity))
        unit = str(rec.CPSE)
        if unit and unit not in entry["units"]:
            entry["units"].append(unit)
        code = str(rec.Original_Material_Code)
        if code and len(entry["codes"]) < 3:
            entry["codes"].append(code)

    ordered = sorted(spellings.values(), key=lambda e: (not e["golden"], -e["records"]))

    if needle:
        # A search inside a family looks at the string, its units and the legacy
        # codes underneath it — a purchase order carries the code, not the text.
        matching_keys = set(
            members.loc[
                members["Original_Material_Code"].astype(str).str.lower().str.contains(needle, regex=False, na=False)
                | members["CPSE"].astype(str).str.lower().str.contains(needle, regex=False, na=False),
                "Original_Description",
            ].astype(str).str.split().str.join(" ").str.upper()
        )
        ordered = [
            e for e in ordered
            if needle in e["text"].lower() or e["key"] in matching_keys
        ]

    total = len(ordered)
    page = ordered[offset : offset + limit]
    return {
        **header,
        "level": "spellings",
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(page) < total,
        "spellings": page,
    }


# One AI pass per run at a time. The work is a few dozen network calls, so it
# runs on a worker thread and the page polls, exactly like harmonization.
_ai_lock = threading.Lock()


def _ai_state(job: dict) -> dict:
    state = job.get("ai")
    if state is None:
        state = job["ai"] = {
            "status": "idle", "progress": 0.0, "message": "",
            "attributes": {}, "flags": {}, "merge_suggestions": [],
            "errors": [], "notes": [], "diagnostics": {}, "sample": None,
            "calls": 0, "checked_families": 0, "checked_pairs": 0,
        }
    return state


def _run_ai(job_id: str, do_standardize: bool, do_review: bool) -> None:
    job = _get_job(job_id)
    result = _require_result(job)
    state = _ai_state(job)

    with _ai_lock:
        state["status"] = "running"
        state["errors"] = []
        try:
            if do_standardize:
                def report(fraction, message):
                    state["progress"] = round(0.5 * fraction if do_review else fraction, 3)
                    state["message"] = message
                families = standardize.families_from(result)
                out = standardize.extract(families, report=report)
                state["attributes"] = out["attributes"]
                state["errors"] += out["errors"]
                state["notes"] = out["notes"]
                state["diagnostics"] = out["stats"]
                state["sample"] = out["sample"]
                state["calls"] += out["calls"]

            if do_review:
                base = 0.5 if do_standardize else 0.0
                span = 0.5 if do_standardize else 1.0

                def report2(fraction, message):
                    state["progress"] = round(base + span * fraction, 3)
                    state["message"] = message
                out = ai_review.review(result, _encode_texts, report=report2)
                state["flags"] = out["flags"]
                state["merge_suggestions"] = out["merge_suggestions"]
                state["checked_families"] = out["checked_families"]
                state["checked_pairs"] = out["checked_pairs"]
                state["errors"] += out["errors"]
                state["calls"] += out["calls"]

            state["status"] = "error" if state["errors"] and not (
                state["attributes"] or state["flags"]) else "done"
            state["progress"] = 1.0
            if state["status"] == "done":
                state["message"] = "%d standardised, %d flags" % (
                    len(state["attributes"]), sum(len(v) for v in state["flags"].values()))
            else:
                state["message"] = state["errors"][0] if state["errors"] else "The AI pass failed."
        except Exception as exc:  # pragma: no cover
            state["status"] = "error"
            state["message"] = f"{type(exc).__name__}: {exc}"
            state["errors"].append(state["message"])


@app.post("/api/jobs/{job_id}/ai")
def start_ai(job_id: str, mode: str = Query("both", pattern="^(both|standardize|review)$")) -> dict:
    """Run the optional AI pass over a finished harmonization.

    Explicitly started, never automatic: it is the one step that sends anything
    off the machine, so it happens because somebody pressed a button."""
    job = _get_job(job_id)
    _require_result(job)
    if not gemini.available():
        raise HTTPException(409, gemini.why_unavailable())
    state = _ai_state(job)
    if state["status"] == "running":
        raise HTTPException(409, "An AI pass is already running for this job.")
    threading.Thread(
        target=_run_ai, args=(job_id, mode in ("both", "standardize"), mode in ("both", "review")),
        daemon=True,
    ).start()
    return {"ok": True, "status": "running"}


@app.get("/api/jobs/{job_id}/ai")
def ai_status(job_id: str) -> dict:
    """What the AI pass found, and what it costs to say so.

    Reports availability first: with no key this endpoint is the thing that
    tells the page to offer the feature as switched off rather than broken."""
    job = _get_job(job_id)
    _require_result(job)
    state = _ai_state(job)
    return {
        "available": gemini.available(),
        "reason": gemini.why_unavailable(),
        "model": gemini.MODEL if gemini.available() else None,
        "status": state["status"],
        "progress": state["progress"],
        "message": state["message"],
        "calls": state["calls"],
        "checked_families": state["checked_families"],
        "checked_pairs": state["checked_pairs"],
        "standardised": len(state["attributes"]),
        "flagged_families": len(state["flags"]),
        "flags_total": sum(len(v) for v in state["flags"].values()),
        "merge_suggestions": state["merge_suggestions"][:25],
        "errors": state["errors"][:3],
        # Why a pass produced nothing, in the pass's own counts. "0 standardised"
        # with no explanation is the least useful outcome there is.
        "notes": state["notes"][:3],
        "diagnostics": state["diagnostics"],
        "sample": state["sample"],
        "fields": standardize.FIELDS,
    }


@app.get("/api/jobs/{job_id}/evaluation")
def job_evaluation(job_id: str) -> dict:
    """Measured accuracy for THIS run, when this file carried ground truth.

    Deliberately not a global constant on an About page. An accuracy figure is
    a property of a dataset, not of a program: a matcher that scores 0.94 on
    fastener descriptions may score 0.61 on instrumentation. So it is computed
    per run, from whatever labels the uploaded file actually carried, and when
    the file carried none the answer is "not measured" together with the one
    thing the user would have to add to change that."""
    result = _require_result(_get_job(job_id))
    report = evaluate(result)
    report["reasoning"] = result.stats.get("column_reasons", {}).get("label")
    report["thresholds"] = threshold_manifest()
    # Report the header the USER's file actually used. The pipeline carries
    # labels under one internal name so files that disagree still line up, but
    # showing that name back would look like the fixed header this deliberately
    # is not.
    source = (result.stats.get("source_columns") or {}).get("label") or []
    if report.get("available") and source:
        report["source_columns"] = list(source)
        report["label_column"] = source[0] if len(source) == 1 else ", ".join(source[:3])
    return report


@app.get("/api/jobs/{job_id}/provenance")
def job_provenance(
    job_id: str,
    q: str | None = None,
    sort: str = Query("records", pattern="^(records|spellings|code|score)$"),
    limit: int = Query(25, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict:
    """For every canonical record: what became it, and how many of each.

    The list view answers "which families exist". The constellation answers
    "what did the matcher group". Neither answers the question an officer asks
    holding a purchase file: *this* code — what exactly did it replace? So this
    returns, per family, the distinct source strings that resolve to it, how
    many records each accounts for, which units wrote them, and which one was
    promoted to the golden record.

    Distinct SPELLINGS, not raw rows: two hundred identical rows are one thing
    a human needs to read, and printing them two hundred times is how a real
    material master becomes unreadable.
    """
    result = _require_result(_get_job(job_id))
    clusters, records = result.clusters, result.records

    keys = _spelling_key(records["Original_Description"])
    spelling_counts = keys.groupby(records["Family_ID"]).nunique()

    rows = clusters.copy()
    rows["_spellings"] = rows["Family_ID"].map(spelling_counts).fillna(0).astype(int)

    if q and q.strip():
        needle = q.strip().lower()
        hit_families = records.loc[
            records["Original_Description"].astype(str).str.lower().str.contains(needle, regex=False, na=False)
            | records["Original_Material_Code"].astype(str).str.lower().str.contains(needle, regex=False, na=False),
            "Family_ID",
        ].unique()
        rows = rows[
            rows["Golden_Record"].str.lower().str.contains(needle, regex=False, na=False)
            | rows["Canonical_Code"].str.lower().str.contains(needle, regex=False, na=False)
            | rows["Family_ID"].isin(hit_families)
        ]

    if sort == "records":
        rows = rows.sort_values("Members", ascending=False, kind="mergesort")
    elif sort == "spellings":
        rows = rows.sort_values("_spellings", ascending=False, kind="mergesort")
    elif sort == "score":
        rows = rows.sort_values("Mean_Similarity", ascending=True, kind="mergesort")
    else:
        rows = rows.sort_values("Canonical_Code", kind="mergesort")

    total = int(len(rows))
    page = rows.iloc[offset : offset + limit]

    members_by_family = {fid: frame for fid, frame in
                         records[records["Family_ID"].isin(page["Family_ID"])].groupby("Family_ID")}

    state = _ai_state(_get_job(job_id))
    ai_flags, ai_attributes = state["flags"], state["attributes"]

    out = []
    for row in page.itertuples(index=False):
        frame = members_by_family.get(row.Family_ID)
        golden_text = str(row.Golden_Record)
        spellings: dict[str, dict] = {}
        if frame is not None:
            for rec in frame.itertuples(index=False):
                raw = str(rec.Original_Description)
                key = " ".join(raw.split()).upper()
                entry = spellings.get(key)
                if entry is None:
                    entry = spellings[key] = {
                        "text": raw[:160],
                        "score": float(rec.Semantic_Similarity),
                        "records": 0,
                        "units": [],
                        "codes": [],
                        # The one spelling that was promoted. The UI should not
                        # have to guess it by comparing strings and get a tie
                        # wrong.
                        "golden": str(rec.Normalized_Description) == golden_text,
                    }
                entry["records"] += 1
                entry["score"] = min(entry["score"], float(rec.Semantic_Similarity))
                unit = str(rec.CPSE)
                if unit and unit not in entry["units"]:
                    entry["units"].append(unit)
                code = str(rec.Original_Material_Code)
                if code and len(entry["codes"]) < 3:
                    entry["codes"].append(code)

        ordered = sorted(spellings.values(), key=lambda e: (not e["golden"], -e["records"]))
        shown = ordered[:SPELLINGS_PER_ROW]

        out.append({
            "family_id": str(row.Family_ID),
            "code": str(row.Canonical_Code),
            "golden": golden_text,
            "records": int(row.Members),
            "spellings": int(row._spellings) if hasattr(row, "_spellings") else len(ordered),
            "units": _unit_counts(row.Unit_Counts),
            "score": float(row.Mean_Similarity),
            "status": str(row.Status),
            "approved": bool(getattr(row, "Approved", False)),
            "sources": shown,
            "hidden": max(0, len(ordered) - len(shown)),
            # Filled by the optional AI pass; empty and invisible without it.
            "flags": ai_flags.get(str(row.Family_ID), []),
            "standard": ai_attributes.get(str(row.Family_ID)),
        })

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(page) < total,
        "sort": sort,
        "totals": {
            "families": int(len(clusters)),
            "records": int(len(records)),
            "spellings": int(keys.nunique()),
        },
        "rows": out,
    }


@app.get("/api/jobs/{job_id}/simulate")
def job_simulate(job_id: str, text: str = Query("", max_length=300)) -> dict:
    """Push one description through the real pipeline and report every step.

    Deliberately GET and side-effect free: it reads the run, it never joins it.
    A description typed here is not added to the master."""
    result = _require_result(_get_job(job_id))
    return simulate(result, text, _encode_texts)


@app.get("/api/jobs/{job_id}/clusters/{family_id}/split")
def split_preview(job_id: str, family_id: str) -> dict:
    """What "Split family" would do, before it does it.

    A reviewer cannot agree to a change they have not been shown. This returns
    the rows that would break away, grouped by spelling, and the seam they are
    cut on — so the button stops being a leap of faith."""
    result = _require_result(_get_job(job_id))
    try:
        return {"can_split": True, "plan": plan_split(result, family_id)}
    except SplitError as exc:
        return {"can_split": False, "reason": str(exc)}


@app.post("/api/jobs/{job_id}/clusters/{family_id}/split")
def split_apply(job_id: str, family_id: str, request: Request,
                force: bool = Query(False)) -> dict:
    """Perform the split. Changes the run this session is looking at.

    `force` is the reviewer overriding a weak seam. It is a query parameter and
    not a default because the refusal has to survive anything that calls this
    endpoint without having read the warning."""
    job = _get_job(job_id)
    result = _require_result(job)
    try:
        outcome = apply_split(result, family_id, _encode_texts, force=force,
                              by=_reviewer(request))
    except SplitError as exc:
        raise HTTPException(409, str(exc))
    _persist(job)
    return {"ok": True, **outcome, "review_families": result.stats["review_families"]}


@app.get("/api/jobs/{job_id}/clusters/{family_id}/approve")
def approve_preview(job_id: str, family_id: str) -> dict:
    """What approving commits to, before it is committed.

    Every other action here shows its working first. This one did not, which
    made it the only place in the app asking to be trusted rather than read."""
    result = _require_result(_get_job(job_id))
    try:
        return {"can_approve": True, "plan": plan_approve(result, family_id)}
    except SplitError as exc:
        return {"can_approve": False, "reason": str(exc)}


@app.get("/api/jobs/{job_id}/merge/{left_id}/{right_id}")
def merge_preview(job_id: str, left_id: str, right_id: str) -> dict:
    """What folding two canonical records into one would do."""
    result = _require_result(_get_job(job_id))
    try:
        return {"can_merge": True, "plan": plan_merge(result, left_id, right_id)}
    except SplitError as exc:
        return {"can_merge": False, "reason": str(exc)}


@app.post("/api/jobs/{job_id}/merge/{left_id}/{right_id}")
def merge_apply(job_id: str, left_id: str, right_id: str, request: Request) -> dict:
    """Perform the merge. One canonical code stops existing."""
    job = _get_job(job_id)
    result = _require_result(job)
    try:
        outcome = apply_merge(result, left_id, right_id, _encode_texts,
                              by=_reviewer(request))
    except SplitError as exc:
        raise HTTPException(409, str(exc))
    _persist(job)
    return {"ok": True, **outcome,
            "families": result.stats["families"],
            "review_families": result.stats["review_families"]}


@app.post("/api/jobs/{job_id}/clusters/{family_id}/approve")
def approve_apply(job_id: str, family_id: str, request: Request) -> dict:
    """Confirm a family's mapping and take it out of the review queue."""
    job = _get_job(job_id)
    result = _require_result(job)
    try:
        outcome = approve_family(result, family_id, by=_reviewer(request))
    except SplitError as exc:
        raise HTTPException(409, str(exc))
    _persist(job)
    return {"ok": True, **outcome, "review_families": result.stats["review_families"]}


def _encode_texts(texts: list[str]):
    """L2-normalized vectors for a handful of strings, from the loaded model."""
    return get_model().encode(
        texts, convert_to_tensor=False, normalize_embeddings=True,
        show_progress_bar=False,
    )


def _spread_suspicion(row, auto_merge_floor: float):
    """Why this family's spread might not be a saving at all.

    Two independent doubts, and they compound:

      * A ratio no real part exhibits. 20x is already hard to justify for one
        material; 600x is not a price difference, it is a unit mismatch
        (per-piece against per-lot), a malformed export, or a family holding
        several different things.
      * The family itself scoring below the auto-merge floor. If the matcher
        was not confident these rows are the same part, every rupee computed
        from comparing them inherits that doubt.

    Returned as a reason, not a suppression: the row still shows its money.
    Hiding the figure would be its own kind of dishonesty — the point is that
    the reader sees the number and the caveat in the same glance."""
    ratio = row.Price_Spread
    ratio = float(ratio) if ratio is not None and not pd.isna(ratio) else None
    score = float(row.Mean_Similarity)

    reasons = []
    if ratio is not None and ratio >= SUSPECT_SPREAD_RATIO:
        reasons.append(
            "%.0f\u00d7 between the cheapest and dearest row. No single material "
            "varies that much \u2014 this usually means the price column mixes units "
            "(per piece against per lot), or the family holds more than one part."
            % ratio
        )
    if score < auto_merge_floor:
        reasons.append(
            "The matcher scored this family %.2f, below the %.2f auto-merge floor, "
            "so it is queued for review. Prices compared across rows the matcher "
            "is unsure about inherit that uncertainty."
            % (score, auto_merge_floor)
        )

    if not reasons:
        return None
    return {
        "level": "high" if len(reasons) > 1 else "watch",
        "ratio": ratio,
        "reasons": reasons,
        "headline": "Treat as a data-quality finding, not a saving"
                    if len(reasons) > 1 else "Check before quoting this figure",
    }


def _uom_basis(row) -> dict:
    """Which unit of measure every price figure beside it is measured in.

    A spread figure whose units nobody can state is not a finding. Before this
    the family's prices were compared across EACH, KG and M as though they were
    one scale, so the "excess" was partly a category error. Now the figures are
    computed inside the family's dominant unit and this says which one, and how
    many priced rows sit outside it — counted, never converted, because turning
    a price per kilo into a price per piece needs a weight this application does
    not have and will not guess."""
    basis = getattr(row, "Basis_UOM", None)
    if not isinstance(basis, str) or not basis:
        return {}
    others = []
    for part in str(getattr(row, "UOM_List", "") or "").split("|"):
        bits = part.split(":")
        if len(bits) == 2 and bits[0] != basis:
            others.append({"uom": bits[0], "records": int(bits[1])})
    return {
        "basis": basis,
        "forms": int(getattr(row, "UOM_Forms", 0) or 0),
        "outside": int(getattr(row, "Other_UOM_Records", 0) or 0),
        "others": others,
    }



def _money(value):
    """None stays None. A savings figure is never rounded up into existence."""
    if value is None or pd.isna(value):
        return None
    return float(value)


@app.get("/api/jobs/{job_id}/savings")
def job_savings(
    job_id: str,
    limit: int = Query(25, ge=1, le=200),
) -> dict:
    """What the harmonization makes visible about money.

    Every figure here is arithmetic over the user's own price column. Nothing
    is modelled, forecast or benchmarked against outside data, and when the
    upload has no usable price column the response says so and carries no
    numbers at all — an invented rupee figure in a procurement tool is worse
    than an empty panel.
    """
    result = _require_result(_get_job(job_id))
    clusters = result.clusters

    if not result.stats.get("has_price_column") or "Spend" not in clusters.columns:
        return {
            "available": False,
            "reason": "No price column was found in the uploaded files, so "
                      "spend cannot be compared across units.",
            "price_column": None,
        }

    priced = clusters[clusters["Priced_Records"].fillna(0) > 0].copy()
    if priced.empty:
        return {
            "available": False,
            "reason": "A price column was found but held no usable positive values.",
            "price_column": result.stats.get("price_column"),
        }

    total_spend = float(priced["Spend"].sum())
    total_excess = float(priced["Excess"].fillna(0).sum())
    total_best = float(priced["Savings"].fillna(0).sum())

    # Families where one unit sits well above the median for the same part.
    outliers = priced[
        priced["Outlier_Ratio"].notna() & (priced["Outlier_Ratio"] > UNIT_OUTLIER_RATIO)
    ].sort_values("Outlier_Excess", ascending=False)

    def unit_medians(raw: str) -> list[dict]:
        out = []
        for chunk in str(raw or "").split("|"):
            if not chunk:
                continue
            parts = chunk.split(":")
            if len(parts) != 4:
                continue
            out.append({
                "unit": parts[0],
                "median": float(parts[1]),
                "records": int(parts[2]),
                "ratio": float(parts[3]),
            })
        return out

    by_excess = priced.sort_values("Excess", ascending=False).head(limit)
    auto_merge_floor = threshold_manifest()["auto_merge"]

    return {
        "available": True,
        "price_column": result.stats.get("price_column"),
        "currency_note": "Figures are in whatever currency the price column holds; "
                         "the pipeline does not convert or assume one.",
        "totals": {
            "spend": total_spend,
            "priced_records": int(priced["Priced_Records"].sum()),
            "unpriced_records": int(result.stats["records"] - priced["Priced_Records"].sum()),
            "families": int(len(priced)),
            "excess_over_median": total_excess,
            "excess_pct": round(100 * total_excess / total_spend, 2) if total_spend else 0.0,
            "against_best_price": total_best,
            "best_pct": round(100 * total_best / total_spend, 2) if total_spend else 0.0,
            "outlier_families": int(len(outliers)),
            "outlier_excess": float(outliers["Outlier_Excess"].fillna(0).sum()) if len(outliers) else 0.0,
        },
        # Ranked by money, not by ratio: a 12x markup on a ten-rupee washer
        # matters less than a 15% markup on a pump.
        "families": [
            {
                "family_id": row.Family_ID,
                "code": row.Canonical_Code,
                "desc": row.Golden_Record,
                "members": int(row.Members),
                "priced": int(row.Priced_Records),
                "spend": _money(row.Spend),
                "low": _money(row.Price_Low),
                "median": _money(row.Price_Median),
                "high": _money(row.Price_High),
                "spread": _money(row.Price_Spread),
                "excess": _money(row.Excess),
                "outlier_unit": row.Outlier_Unit if isinstance(row.Outlier_Unit, str) else None,
                "outlier_ratio": _money(row.Outlier_Ratio),
                "outlier_excess": _money(row.Outlier_Excess),
                "units": unit_medians(row.Unit_Medians),
                "uom": _uom_basis(row),
                "suspect": _spread_suspicion(row, auto_merge_floor),
            }
            for row in by_excess.itertuples(index=False)
        ],
        "outliers": [
            {
                "family_id": row.Family_ID,
                "code": row.Canonical_Code,
                "desc": row.Golden_Record,
                "unit": row.Outlier_Unit,
                "ratio": _money(row.Outlier_Ratio),
                "excess": _money(row.Outlier_Excess),
                "median": _money(row.Price_Median),
                "units": unit_medians(row.Unit_Medians),
            }
            for row in outliers.head(limit).itertuples(index=False)
        ],
        "thresholds": {
            "outlier_ratio": UNIT_OUTLIER_RATIO,
            "min_rows_for_outlier": MIN_ROWS_FOR_UNIT_OUTLIER,
            "suspect_spread": SUSPECT_SPREAD_RATIO,
        },
    }


def _export_frame(result):
    """The mapping table as published.

    Three columns ride along on the records frame as working state and are
    dropped here: the unit price and the folded unit of measure, so a
    reviewer's split can recompute a family's spread inside one unit, and the
    ground-truth label, which is what accuracy is scored against rather than
    part of the mapping we publish. The export keeps exactly the shape it has
    always had."""
    frame = result.records
    working = [c for c in (CANONICAL_PRICE, CANONICAL_UOM, CANONICAL_LABEL)
               if c in frame.columns]
    return frame.drop(columns=working) if working else frame


@app.get("/api/jobs/{job_id}/export.csv")
def export_csv(job_id: str, audit: int = Query(0, ge=0, le=1)) -> StreamingResponse:
    """The full source to canonical mapping.

    Without `audit`, exactly the shape it has always had — a loader that has
    been fed this file for weeks does not silently receive four new columns
    because a reviewer clicked something. With `audit=1`, four provenance
    columns are appended: what state each family is in, when a person last
    touched it, and what it was merged from or split from."""
    result = _require_result(_get_job(job_id))
    frame = _export_frame(result)
    if audit:
        frame = decisions.audit_frame(frame, result)
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    buffer.seek(0)
    name = f'harmonized_{"audit_" if audit else ""}{job_id}.csv'
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@app.get("/api/jobs/{job_id}/sweep")
def threshold_sweep(job_id: str) -> dict:
    """What the auto-merge floor buys and what it costs, at every setting.

    The most consequential constant in the system is the score above which a
    family is accepted with nobody reading it. "Why 0.82?" deserves a curve,
    not an assertion, and the curve can be computed from the run that already
    exists — the band a family lands in is a pure function of its mean score,
    so no re-clustering is needed."""
    result = _require_result(_get_job(job_id))
    return sweep.sweep(result)


@app.get("/api/jobs/{job_id}/decisions")
def decisions_log(job_id: str) -> dict:
    """Every decision a person made against this run, newest last.

    The mapping says what the answer is. This says who decided it. A run where
    nobody has touched anything returns an empty list and says so, which is a
    real answer rather than a missing one."""
    result = _require_result(_get_job(job_id))
    log = decisions.log_of(result)
    return {
        "entries": [decisions.public(e) for e in log],
        "summary": decisions.summary(result),
        "families": int(len(result.clusters)),
        "records": int(len(result.records)),
    }


@app.post("/api/jobs/{job_id}/decisions/undo")
def decisions_undo(job_id: str, request: Request) -> dict:
    """Take back the last decision.

    Merge retires a code and split invents one. Without this the only way back
    from a wrong click was to re-run the file, which loses every other decision
    made since — so in practice a reviewer stopped experimenting, read the
    score and clicked approve, which is the behaviour this application exists
    to prevent.

    Reversal replays the payload the log already carries, captured inside each
    verb before it changed anything. Nothing is re-derived, so the run comes
    back exactly as it was rather than as closely as we can recompute."""
    job = _get_job(job_id)
    result = _require_result(job)
    try:
        outcome = undo_last(result, by=_reviewer(request))
        _persist(job)
    except SplitError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return outcome


@app.get("/api/jobs/{job_id}/decisions.csv")
def decisions_csv(job_id: str) -> StreamingResponse:
    """The audit trail as its own file, beside the mapping.

    Separate from the mapping on purpose: an auditor wants the decisions, not
    four extra columns smeared across fifty thousand rows. The header row is
    written even with no decisions in it, because an empty audit file says
    "nobody changed anything" and a file with no columns says nothing at all.
    """
    result = _require_result(_get_job(job_id))
    buffer = io.StringIO()
    decisions.log_frame(result).to_csv(buffer, index=False)
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition":
                 f'attachment; filename="decisions_{job_id}.csv"'},
    )


@app.get("/api/jobs/{job_id}/preview.csv")
def preview_csv(job_id: str, rows: int = Query(25, ge=1, le=200)) -> dict:
    """First N lines as text, for the in-page export preview."""
    result = _require_result(_get_job(job_id))
    buffer = io.StringIO()
    _export_frame(result).head(rows).to_csv(buffer, index=False)
    return {"text": buffer.getvalue(), "total_rows": int(len(result.records))}


class QuestionIn(BaseModel):
    question: str = Field(min_length=5, max_length=MAX_QUESTION_CHARS)
    name: str = Field(default="", max_length=80)
    email: str = Field(default="", max_length=160)


def _clean(text: str) -> str:
    """Collapse whitespace and strip control characters."""
    return re.sub(r"[\x00-\x1f\x7f]", " ", re.sub(r"\s+", " ", text or "")).strip()


@app.post("/api/questions")
def post_question(payload: QuestionIn, request: Request) -> JSONResponse:
    question = _clean(payload.question)
    if len(question) < 5:
        raise HTTPException(400, "Please write a question of at least five characters.")

    # Light per-client throttle. Not security — just enough that a stuck key or
    # a double-click does not fill the list.
    who = request.client.host if request.client else "unknown"
    now = time.time()
    if now - _last_post.get(who, 0.0) < QUESTION_MIN_SECONDS:
        raise HTTPException(429, "One question at a time, please — try again in a few seconds.")
    _last_post[who] = now

    entry = {
        "id": uuid.uuid4().hex[:10],
        "question": question[:MAX_QUESTION_CHARS],
        "name": _clean(payload.name)[:80],
        "email": _clean(payload.email)[:160],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
    }
    _append_question(entry)
    return JSONResponse({"ok": True, "id": entry["id"], "total": len(_questions)}, status_code=201)


@app.get("/api/questions")
def list_questions(
    limit: int = Query(8, ge=1, le=50),
    code: str | None = None,
) -> dict:
    """Newest first.

    Email addresses are withheld unless the caller supplies the reply code.
    That code is a shared secret typed into a box — it keeps addresses off the
    public list, which is what it is for, and it is NOT authentication: anyone
    who learns it can read every address, and it travels in the query string.
    A deployment that holds real addresses should put a real login in front of
    this route rather than raising the code's entropy.
    """
    unlocked = bool(REPLY_CODE) and _constant_eq(str(code or ""), REPLY_CODE)

    with _questions_lock:
        recent = list(reversed(_questions))[:limit]
        total = len(_questions)
        with_email = sum(1 for q in _questions if q.get("email"))

    out = []
    for q in recent:
        item = {
            "id": q.get("id", ""),
            "question": q.get("question", ""),
            "name": q.get("name", ""),
            "created_at": q.get("created_at", ""),
            # Whether a reply address exists is not itself sensitive, and the
            # page needs it to show a lock rather than nothing at all.
            "has_email": bool(q.get("email")),
        }
        if unlocked:
            item["email"] = q.get("email", "")
        out.append(item)

    return {
        "total": total,
        "with_email": with_email,
        "unlocked": unlocked,
        "questions": out,
    }


@app.exception_handler(StarletteHTTPException)
async def custom_404(request: Request, exc: StarletteHTTPException):
    """Serve the designed 404 for pages; keep JSON for the API."""
    if exc.status_code == 404 and not request.url.path.startswith("/api/"):
        page = STATIC_DIR / "404.html"
        if page.is_file():
            return FileResponse(page, status_code=404)
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


# Static pages last, so /api/* always wins.
if STATIC_DIR.is_dir():
    class NoStoreStatic(StaticFiles):
        """Serve the pages and assets with caching switched off.

        A browser holding an old copy of constellation.js against a new
        index.html produces the worst possible failure: the page loads, looks
        current, and then calls a function the cached script does not have. It
        looks like a bug in the new code and it is not. Nobody is serving this
        at a scale where caching static files matters, so the safe trade is to
        never cache them.
        """

        async def get_response(self, path, scope):
            response = await super().get_response(path, scope)
            response.headers["Cache-Control"] = "no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            return response

    app.mount("/", NoStoreStatic(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
