"""Runs that survive a restart.

Everything this application computed used to live in one Python dictionary. A
restart erased it; so did the fifth upload, because only four runs are kept in
memory. Both are reasonable for a scratchpad and wrong for a review tool: a
person spends twenty minutes confirming mappings, merging duplicates and
splitting bad families, and then the process restarts and every one of those
decisions is gone with nothing to show it ever happened.

So a completed run is written to disk, and read back at startup.

What is persisted, and what deliberately is not
-----------------------------------------------
The RESULT — the two frames, the statistics, and the decision log with them —
is persisted. That is the reviewed material master and the record of who
changed it, and it is the only thing that cannot be reconstructed.

The uploaded FILES are not. They are held in memory for the length of the run
and discarded, exactly as the privacy page has always said, and nothing here
changes that. A restored run can be read, exported and reviewed further; it
cannot be re-clustered at a different threshold, because the source rows it
would need are gone by design.

Failures here are never fatal. If the directory is read-only, the disk is full,
or a file was written by an older version and no longer loads, the run simply
stays in memory the way it always did and a warning is recorded. A tool that
refuses to work because it could not write a cache is worse than one that
forgets.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

# Where runs go. Configurable because the container this ships in may mount
# something writable somewhere else; defaults beside the app.
DATA_DIR = Path(os.getenv("HARMONIZER_DATA", "./data/runs")).expanduser()

# How many runs to keep on disk. Larger than the in-memory limit — disk is
# cheap and the whole point is that a run outlives its slot in memory.
KEEP_RUNS = int(os.getenv("HARMONIZER_KEEP_RUNS", "40"))

# Bumped when the pickled shape changes. A file from an older version is
# ignored rather than half-loaded.
FORMAT = 3

_lock = threading.Lock()
_warnings: list[str] = []


def warnings() -> list[str]:
    """Anything that went wrong on disk, for the health endpoint."""
    return list(_warnings)


def _note(message: str) -> None:
    if message not in _warnings:
        _warnings.append(message)


def available() -> bool:
    """Can we write here at all? Checked once, honestly, at startup."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        probe = DATA_DIR / ".writable"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except Exception as exc:
        _note(f"Runs are not being saved to disk: {type(exc).__name__}. "
              "Everything still works; nothing survives a restart.")
        return False


def _path(job_id: str) -> Path:
    return DATA_DIR / f"{job_id}.run"


def _write_atomic(path: Path, payload: bytes) -> None:
    """Never leave a half-written run where a reader will find it.

    A crash mid-write would otherwise produce a file that loads as far as the
    records frame and then raises, which is a worse failure than no file."""
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".part")
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(payload)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def save(job: dict) -> bool:
    """Persist one finished run. Returns whether it landed."""
    result = job.get("result")
    if result is None or job.get("status") != "done":
        return False
    if not available():
        return False

    payload = {
        "format": FORMAT,
        "job_id": job["job_id"],
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "meta": {k: v for k, v in job.items() if k != "result"},
        "records": result.records,
        "clusters": result.clusters,
        "stats": result.stats,
        "warnings": result.warnings,
        # The reason this module exists. Written with the frames, in the same
        # file, so a restored run cannot disagree with its own history.
        "decisions": list(getattr(result, "decisions", []) or []),
    }
    try:
        with _lock:
            _write_atomic(_path(job["job_id"]), pickle.dumps(payload, protocol=4))
            _prune()
        return True
    except Exception as exc:
        _note(f"Could not save run {job['job_id']}: {type(exc).__name__}. "
              "It stays in memory for this session only.")
        return False


def load_all() -> list[dict]:
    """Every readable run on disk, newest first.

    A file that will not load is skipped and named, not raised — one bad run
    from an older version must not stop the server from starting."""
    if not DATA_DIR.exists():
        return []
    jobs = []
    for path in sorted(DATA_DIR.glob("*.run"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(path, "rb") as handle:
                payload = pickle.load(handle)
            if payload.get("format") != FORMAT:
                continue
            jobs.append(payload)
        except Exception as exc:
            _note(f"Skipped an unreadable saved run ({path.name}): {type(exc).__name__}.")
    return jobs


def forget(job_id: str) -> bool:
    try:
        _path(job_id).unlink()
        return True
    except FileNotFoundError:
        return False
    except Exception as exc:
        _note(f"Could not delete run {job_id}: {type(exc).__name__}.")
        return False


def _prune() -> None:
    """Oldest first, past KEEP_RUNS."""
    files = sorted(DATA_DIR.glob("*.run"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files[KEEP_RUNS:]:
        try:
            path.unlink()
        except OSError:
            pass


def usage() -> dict:
    """What is on disk, for the health endpoint and the privacy page."""
    if not DATA_DIR.exists():
        return {"enabled": False, "runs": 0, "bytes": 0, "path": str(DATA_DIR)}
    files = list(DATA_DIR.glob("*.run"))
    return {
        "enabled": True,
        "runs": len(files),
        "bytes": sum(f.stat().st_size for f in files),
        "path": str(DATA_DIR.resolve()),
        "keep": KEEP_RUNS,
    }
