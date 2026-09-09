#!/usr/bin/env bash
# Local development server. First run downloads the model (~90 MB).
set -euo pipefail
python -m venv .venv 2>/dev/null || true
source .venv/bin/activate
pip install -q -r requirements.txt
exec uvicorn app:app --host 0.0.0.0 --port 7860 --reload
