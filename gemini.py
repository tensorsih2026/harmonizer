"""The only place this application talks to a model it does not run itself.

Everything else here is local: MiniLM on the operator's own machine, arithmetic
over their own columns, nothing leaving the building. That is a real selling
point for a public sector unit and it is not given up lightly, so this module
is written to be switched off:

  * No GEMINI_API_KEY in the environment and the app behaves exactly as it did
    before this file existed. Not degraded — identical. The AI features simply
    report themselves as unavailable, with the one line needed to enable them.
  * The key is read server-side, from the environment, once. It is never sent
    to the browser, never written to disk, never logged, and never included in
    an error message.
  * Only material DESCRIPTIONS are ever sent. Not prices, not unit names, not
    legacy codes, not the file. The prompts below are the entire surface.

Standard library only — no SDK, no new dependency to install, nothing that can
break an offline deployment that never calls this anyway.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

# Flash, not flash-lite: attribute extraction is a domain-knowledge task —
# knowing SCH 40 is a wall thickness and 2RS a seal type — and the lite tiers
# trade exactly that away for throughput. Model IDs are retired on Google's
# schedule, not ours (2.0-flash was shut down), so this is overridable without
# touching code: set GEMINI_MODEL in the environment.
DEFAULT_MODEL = "gemini-3.5-flash"
MODEL = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

TIMEOUT_S = float(os.environ.get("GEMINI_TIMEOUT", "45"))

# Rate limiting, which the free tier makes unavoidable.
#
# A single AI pass is not one request. Standardization batches ten families per
# call and the review pass batches six, so a run of eighty families fires
# fifteen or twenty calls back to back — and a free-tier key is metered per
# MINUTE (the 429 body says so: "limit: 20 ... retry in 5.9s"). Firing them as
# fast as the network allows spends a minute's budget in four seconds and the
# pass dies half-finished with a wall of Google's error text.
#
# So two things. A floor on the interval between calls, which keeps a normal
# pass inside the free tier without anybody configuring anything; and a retry
# that honours the delay Google actually asks for instead of guessing 1.5s and
# giving up. Both are overridable for a paid key, where neither is needed.
MIN_INTERVAL_S = float(os.environ.get("GEMINI_MIN_INTERVAL", "3.2"))
MAX_ATTEMPTS = int(os.environ.get("GEMINI_MAX_ATTEMPTS", "4"))
MAX_BACKOFF_S = float(os.environ.get("GEMINI_MAX_BACKOFF", "30"))

KEY_ENV = "GEMINI_API_KEY"

# When the last request went out. Module-level because the pass runs in one
# worker thread; the lock is there so a second job cannot lap the first.
_last_call = 0.0
_pace_lock = threading.Lock()


def _pace() -> None:
    """Block until at least MIN_INTERVAL_S has passed since the last call."""
    global _last_call
    if MIN_INTERVAL_S <= 0:
        return
    with _pace_lock:
        wait = MIN_INTERVAL_S - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()


def _retry_after(detail: str) -> float:
    """The delay Google asked for, from its own message.

    A 429 body ends "Please retry in 5.920910853s". Guessing a backoff when the
    server has told you the number is how a pass fails against a limit it was
    one sleep away from clearing."""
    found = re.search(r"retry in ([0-9.]+)\s*s", detail, re.IGNORECASE)
    if not found:
        return 0.0
    try:
        return min(float(found.group(1)) + 0.4, MAX_BACKOFF_S)
    except ValueError:
        return 0.0


def quota_advice(detail: str) -> str:
    """One sentence a person can act on, instead of Google's paragraph."""
    if "quota" not in detail.lower() and "429" not in detail:
        return ""
    return (
        "That is the free tier's per-minute request budget, not a bill \u2014 it "
        "refills within a minute. The pass paces itself to stay under it; if you "
        "hit it anyway, wait a minute and press Run again, or set "
        "GEMINI_MIN_INTERVAL higher on the server."
    )


class AIUnavailable(Exception):
    """No key, no network, or the provider refused. Never fatal to a run."""


def available() -> bool:
    return bool(os.environ.get(KEY_ENV, "").strip())


def why_unavailable() -> str:
    if available():
        return ""
    return (
        "No %s is set on the server, so the AI pass is switched off. Everything "
        "else runs exactly as it does without it. To enable it, set that "
        "environment variable to a Google AI Studio key and restart." % KEY_ENV
    )


def _post(payload: dict) -> dict:
    key = os.environ.get(KEY_ENV, "").strip()
    if not key:
        raise AIUnavailable(why_unavailable())

    url = ENDPOINT.format(model=MODEL)
    body = json.dumps(payload).encode("utf-8")
    last: Exception | None = None

    for attempt in range(MAX_ATTEMPTS):
        _pace()
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                # Header rather than a query string: a key in a URL ends up in
                # proxy logs and browser history.
                "x-goog-api-key": key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = json.loads(exc.read().decode("utf-8")).get("error", {}).get("message", "")
            except Exception:
                pass
            # The key must never reach a message that might be displayed.
            detail = detail.replace(key, "[key]")
            if exc.code in (429, 500, 502, 503) and attempt + 1 < MAX_ATTEMPTS:
                last = exc
                # Google's own number first; our exponential guess only when it
                # did not give one.
                time.sleep(_retry_after(detail) or min(1.5 * 2 ** attempt, MAX_BACKOFF_S))
                continue
            if exc.code in (400, 404) and "model" in detail.lower():
                # The most likely first-run failure, and the least obvious:
                # Google retires model IDs, so a name that worked last quarter
                # returns 404. Say which name was tried and how to change it.
                raise AIUnavailable(
                    f"Gemini does not recognise the model \"{MODEL}\". Model IDs are "
                    f"retired periodically; set GEMINI_MODEL in the server environment "
                    f"to a current one and restart. ({detail})".strip()
                )
            advice = quota_advice(detail) if exc.code == 429 else ""
            # A rate limit is not a broken key, and reading Google's paragraph
            # to work that out is not the reviewer's job.
            if exc.code == 429:
                raise AIUnavailable(
                    f"Gemini is rate-limiting this key. {advice}".strip())
            raise AIUnavailable(f"Gemini refused the request ({exc.code}). {detail}".strip())
        except Exception as exc:  # network, timeout, malformed JSON
            last = exc
            if attempt + 1 < MAX_ATTEMPTS:
                time.sleep(1.0)
                continue
            raise AIUnavailable(f"Could not reach Gemini: {type(exc).__name__}.")

    raise AIUnavailable(f"Could not reach Gemini: {type(last).__name__}.")


def _text_of(response: dict) -> str:
    for candidate in response.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if "text" in part:
                return part["text"]
    return ""


def generate_json(prompt: str, *, temperature: float = 0.0, transport=None) -> object:
    """Ask for JSON and return it parsed, or raise AIUnavailable.

    temperature 0 by default: a material master is not a place for variety, and
    a cached result should stay the same on a second run.

    `transport` exists so the whole feature can be exercised end to end without
    a key and without a network — the test suite passes a stand-in that returns
    fixed replies. Production never sets it.
    """
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
        },
    }
    response = (transport or _post)(payload)
    text = _text_of(response).strip()
    if not text:
        raise AIUnavailable("Gemini returned an empty response.")

    # responseMimeType asks for bare JSON, but a fenced block still shows up
    # occasionally and is not worth failing a whole pass over.
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fenced:
        text = fenced.group(1)

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise AIUnavailable(f"Gemini returned text that is not JSON ({exc.msg}).")
