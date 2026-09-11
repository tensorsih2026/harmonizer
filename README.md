---
title: Material Code Harmonizer
emoji: 🧩
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Material Code Harmonizer — PS26099

Resolves fragmented CPSE material descriptions into one canonical taxonomy.
Landing page, API and sandbox workspace all served from a single origin.

```
index.html  →  POST /api/jobs  →  poll  →  sandbox.html?job=<id>
```

---

## What runs where

| Path | What it is |
|---|---|
| `/` , `/index.html` | Landing page. Upload one or more unit exports, run the pipeline, watch real progress. |
| `/sandbox.html?job=<id>` | Verified workspace for a finished run: families, similarity scores, review queue, export. |
| `/sandbox.html` | Same page with no job — shows an empty state. It never invents results. |
| `/api/*` | The API below. |

The verification gate on the sandbox **accepts any input** and verifies nothing.
It is a demo gate. The uploaded ID never leaves the browser tab.

---

## Local run

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

Open <http://localhost:7860>. First start downloads the embedding model
(~90 MB) — subsequent starts are instant.

Need a test file?

```bash
python tests/make_sample.py 5000 sample_master.csv            # one clean file
python tests/make_sample.py 5000 out.csv --messy              # banner rows, latin-1, blanks, dupes
python tests/make_sample.py 5000 exports/ --per-unit --messy  # one file per CPSE, mismatched headers
```

Dev-only: nothing in `app.py` or `harmonizer.py` imports from `tests/`.

---
## Deploy to Google Cloud Run

Cloud Run is the right fit for this stack — persistent container, no cold-start
size limits, and it builds straight from the Dockerfile above.

1. Create or open a Google Cloud project with billing enabled, then open
   **Cloud Shell** (the `>_` icon top-right of the console).
2. Upload this folder to Cloud Shell (drag-and-drop the zip via the Cloud Shell
   **⋮ menu → Upload**, then unzip it), or `git clone` this repo directly.
3. `cd` into the folder that contains this `Dockerfile` — deploying from the
   wrong directory is the most common failure here; watch the build log for
   the line **"Building using Dockerfile"**, not "Buildpacks".
4. Deploy:

```bash
   gcloud run deploy harmonizer \
     --source . \
     --region asia-south1 \
     --allow-unauthenticated \
     --memory 4Gi \
     --cpu 2 \
     --port 7860 \
     --timeout 300 \
     --min-instances 1 \
     --set-env-vars GEMINI_API_KEY=your_key_here
```

   `--min-instances 1` keeps one instance warm so there's no cold-start delay
   during a demo. Omit `--set-env-vars` (or the whole flag) to run without the
   optional AI pass.

5. First build takes several minutes — Cloud Build compiles the Dockerfile,
   installs `torch` + `sentence-transformers`, and bakes the embedding model
   into the image so the first real request doesn't pay for it.
6. If the deploy fails on a permissions error (`storage.objectViewer` or
   `artifactregistry.writer`), grant the missing role to the project's default
   compute service account:

```bash
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
     --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
     --role="roles/artifactregistry.writer"
```

7. On success, `gcloud` prints a `*.run.app` URL — that's the live, publicly
   reachable app.

To redeploy after a change, re-run the same `gcloud run deploy` command from
inside the folder — Cloud Run builds a new revision and shifts traffic to it
automatically.

### Splitting frontend and backend

You don't need to, but if you host the pages elsewhere, set the API base
before the page scripts run and add CORS on the server:

```html
<script>window.HARMONIZER_API = 'https://your-space.hf.space';</script>
```

```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["https://your-frontend"],
                   allow_methods=["*"], allow_headers=["*"])
```

---

## API

| Method | Path | Returns |
|---|---|---|
| `GET` | `/api/health` | Liveness and whether the model is loaded. |
| `POST` | `/api/jobs` | `202` + job record. Multipart field name is `files` (repeatable). |
| `GET` | `/api/jobs/{id}` | `status`, `stage_index` 0–3, `stage_progress` 0–1, `message`. |
| `GET` | `/api/jobs/{id}/summary` | KPIs, confidence distribution, view counts, unit list. |
| `GET` | `/api/jobs/{id}/clusters` | Paginated. `view`, `unit`, `q`, `limit`, `offset`. |
| `GET` | `/api/jobs/{id}/clusters/{family_id}` | One family with all its source records. |
| `GET` | `/api/jobs/{id}/clusters/{family_id}/approve` | What confirming this mapping would sign off. |
| `POST` | `/api/jobs/{id}/clusters/{family_id}/approve` | Confirm it. Leaves the review queue; score untouched. |
| `GET` | `/api/jobs/{id}/clusters/{family_id}/split` | The seam, and both resulting groups, before splitting. |
| `POST` | `/api/jobs/{id}/clusters/{family_id}/split` | Split it. `?force=1` overrides a weak seam. |
| `GET` | `/api/jobs/{id}/merge/{left}/{right}` | Which code survives, which retires, and the resulting record. |
| `POST` | `/api/jobs/{id}/merge/{left}/{right}` | Merge them. The absorbed code stops existing. |
| `GET` | `/api/runs` | Every run still held, in memory or on disk. |
| `GET` | `/api/jobs/{id}/sweep` | What every auto-merge floor costs and buys. |
| `GET` | `/api/jobs/{id}/decisions` | The audit trail: every decision a person made, in order. |
| `POST` | `/api/jobs/{id}/decisions/undo` | Take back the last one. `409` with a sentence if it cannot. |
| `GET` | `/api/jobs/{id}/decisions.csv` | The same as a file, one row per decision. |
| `GET` | `/api/jobs/{id}/export.csv` | Full source → canonical mapping. `?audit=1` appends four provenance columns. |
| `GET` | `/api/jobs/{id}/preview.csv` | First N rows as text, for the in-page preview. |

Filtering and search happen **server-side**. At 50,000+ families the browser
cannot hold the list, so the page requests 50 at a time and asks for more.

### Input columns

Detected case-insensitively; the first match wins.

- **Description** (required) — `Description`, `Material Description`, `Item Description`, `Short Text`, …
- **Unit** (optional) — `CPSE`, `Organisation`, `Company`, `Business Unit`, `Plant`, `Refinery`, …
  (a bare `Unit` column is deliberately NOT matched — see below)
- **Code** (optional) — `Material Code`, `Item Code`, `Material Number`, …

Upload one file per CPSE and **omit the unit column** — the filename stem
becomes the unit name. That is the cleanest way to keep provenance, and it
avoids the dedup trap described below.

---

## No fabricated output

The pages never invent numbers. There is no bundled sample dataset and no
simulated run:

- `sandbox.html` without a `?job=` shows an empty state, not example families.
- Every figure is blank (`—`) until a real summary loads.
- If the backend is unreachable, the upload page says so and the Run button
  resets. It does not play a progress animation or open a completion dialog.
- If a job has been evicted (`RETAIN_JOBS`, default 4), the sandbox says the run
  is gone rather than showing stale results.

## Things to know before you demo

**The AI pass paces itself for a free-tier key.** One pass is not one request —
standardization batches ten families per call and the review pass six, so eighty
families is fifteen to twenty calls. A free Google AI Studio key is metered per
*minute* (the 429 says so: `limit: 20`), and firing those calls as fast as the
network allows spends a minute's budget in four seconds. `gemini.py` therefore
leaves `GEMINI_MIN_INTERVAL` seconds between calls (3.2 by default, ~18/min) and
honours the retry delay Google returns rather than guessing one. A pass on a
large run takes a minute; the progress bar shows it. On a paid key set
`GEMINI_MIN_INTERVAL=0`.

| Variable | Default | What it does |
|---|---|---|
| `GEMINI_API_KEY` | — | Enables the AI pass. Never leaves the server. |
| `GEMINI_MODEL` | `gemini-3.5-flash` | Model IDs are retired on Google's schedule. |
| `GEMINI_MIN_INTERVAL` | `3.2` | Seconds between calls. `0` on a paid key. |
| `GEMINI_MAX_ATTEMPTS` | `4` | Retries per call, honouring Google's own delay. |
| `RETAIN_JOBS` | `4` | Runs kept in memory before the oldest is evicted. |

**It is built for a 1366x768 laptop.** Dialogs cap at the window height and
scroll inside themselves, and the family list drops its unit chips before it
drops the description — a container query on the list column, so it responds to
how wide that column actually is rather than to the window, since the detail
pane can be open or closed at the same window width.

**Runs survive a restart.** A finished run — both frames, the statistics and
every reviewer decision — is written to `HARMONIZER_DATA` (default `./data/runs`)
when it completes and after each decision, and read back at startup. Memory
becomes a cache in front of disk, so the five-run eviction stops being a loss.
The uploaded FILES are still never written down: a restored run can be read,
exported and reviewed further, but not re-clustered, because the source rows
are gone by design. If the directory is unwritable the app carries on exactly as
before and says so on `/api/health`.

**Decisions carry a name.** There is no login, so what is captured is a
signature the reviewer types — shown in each dialog before the decision is
taken, and written to `Decided_By` in the log and `Reviewed_By` in the audited
mapping. An unsigned decision records `unattributed` rather than borrowing
somebody else's name.

**Each canonical record gets a proposed code.** `MTL-BOLT-0001` instead of
`MTL-0000007`. The class is discovered from the uploaded file — a word many
families share is a class, a word one family uses is a specification — and
numbering runs inside a class in description order, so the same input yields the
same codes whatever order the families came out in. A proposal, beside the
working code, in the audited export only.

**The auto-merge floor is a control, not a constant.** Above it a family is
published with nobody reading it; below it a person has to. `/sweep` reports,
for every candidate floor, how many families ship unread, how many go to the
queue, how many source records that is — and, when the upload carries a
ground-truth column, what fraction of the unread families really are one item.
Nothing is re-clustered when the floor moves: family membership was fixed at
run time, and the band a family lands in is a pure function of a score already
measured. "Why 0.82?" is answered with the run's own curve.

**Units of measure are folded, and prices are compared inside one.** `EA`,
`NOS`, `NO`, `PCS` and `EACH` are one unit, the same way the descriptions are
one description. What follows from that matters more: every price figure — the
spread, the excess, the per-CPSE ladder — is computed inside a single unit of
measure, and the family says which one and how many priced rows sit outside it.
Nothing is converted. A price per kilogram does not become a price per piece,
because that needs a weight this application does not have, and a plausible
invented number in a procurement tool is worse than no number.

**Dirty unit values are folded, not fragmented.** `A`, ` A`, `a` and `A  ` are
one unit. `UNKNOWN`, `N/A`, `ERROR` and blanks become `UNSPECIFIED` rather than
four separate units in the rail. Descriptions that are placeholders (`#N/A`,
`?`, blank) are excluded and counted, instead of clustering into their own
canonical family.

**A bare `Unit` column is NOT read as the organisation.** In material masters
that column almost always holds the unit of measure (nos, Pcs, EA). Name the
column `CPSE`, `Organisation`, `Plant` or `Business Unit`, or upload one file
per unit and let the filename carry it.

**Files with different column names are normalised before they are combined.**
Ten exports where one says `Description`, another `Material Description` and a
third `Short Text` all load. Without this, rows from the odd files out are
silently dropped as blank.

**Banner rows are skipped.** The header is searched for in the first 20 lines,
so a `MATERIAL MASTER EXTRACT` title block and a timestamp above the real header
do not break the parse. The delimiter is sniffed, so TSV and semicolon files
work too.

**Deduplication can eat your best slide.** `drop_duplicates()` removes rows
identical across *every* column. If your CSV is description-only, four units
writing the same bolt collapse to one row *before* clustering runs — and the
cross-unit duplicate you wanted to show disappears. Keep a unit column, or
upload one file per unit. The API reports `rows_duplicate` in the summary so
you can see what was dropped.

**Unassigned records are reported, not hidden.** If `community_detection`
leaves any record out of every family, a warning appears in the summary and on
the sandbox page rather than the output silently being short.

**A quirk in the normalization rules.** `\bS\.?S\.?\b` matches `S.S` but not the
trailing dot in `S.S.`, so `BLT HEX S.S. 10MM` normalizes to
`BLT HEX STAINLESS STEEL. 10MM` — with a stray period. Harmless for the
embeddings, visible in the UI. Left as-is because it is your rule; change the
pattern to `\bS\.?S\.?\.?(?=\s|$)` if you want it gone.

**No typed attribute extraction.** The pipeline goes normalize → embed →
cluster. It does not parse material, thread, length or standard into fields.
The sandbox shows the normalized form instead, and the landing page copy has
been changed to match. If a judge asks about "typed attributes", that is the
honest answer — or add extractors before Saturday.

**Timing.** MiniLM on 2 vCPU encodes roughly a few hundred short descriptions
per second. Budget minutes, not seconds, for 50,000+ rows — which is exactly
why the browser polls instead of waiting on one request.

---

## Files

```
app.py              FastAPI: routes, job store, background worker
harmonizer.py       The pipeline. Import it anywhere; no server needed
static/index.html   Landing page — upload, progress, completion dialog
static/sandbox.html Verification gate + workspace
Dockerfile          HF Spaces build, port 7860, model baked in
requirements.txt    Floors, not pins — freeze after your first green build
review.py           Split, approve and merge — the three reviewer verbs
uom.py              Unit-of-measure folding. EA/NOS/NO are one unit; KG is not
sweep.py            The auto-merge floor as a trade curve, not a constant
codes.py            Proposed codes built from what an item is, not when it was made
store.py            Finished runs, and their decisions, written to disk
decisions.py        The audit trail: who decided what, and when
evaluate.py         Ground-truth scoring when the upload carries a label column
standardize.py      Optional: typed attributes per canonical record (Gemini)
ai_review.py        Optional: flags and duplicate suggestions (Gemini)
gemini.py           The only place a key is read. Never reaches the browser
tests/              Sample-data generator and no-dependency test harness
```

Every reviewer decision is previewed before it is taken — approve, split and
merge all open the same dialog, state what will change, and only then offer the
confirm. Decisions are held in the run for the session; nothing is written back
to an ERP.

Each one is also logged. `decisions.csv` is one row per decision, in plain
words, and `export.csv?audit=1` marks every mapping row `unreviewed`,
`confirmed`, `merged` or `split` with the time a person last touched it. The
plain `export.csv` is unchanged by any of this, so a loader that has been fed
that shape for weeks does not silently receive four new columns because a
reviewer clicked something.

And each one can be taken back. Undo replays the reversal payload the log
already carries — captured inside each verb before it changed anything — so the
run comes back exactly as it was rather than as closely as it can be
recomputed. Last in, first out, because decisions compound: a split creates a
family a later merge can absorb. An undone decision stays in the log, marked,
because "merged and then unmerged" and "never touched" are different histories.
Undo appears on the toast that announced the decision and on the newest row of
the decision trail.

`tests/stub_env.py` fakes torch and sentence-transformers with numpy, so you
can exercise the pipeline and the whole UI on a machine that cannot install
them:

```bash
python tests/mock_server.py sample_master.csv 8765
```
