"""
Streamlit Cloud entrypoint.

Running `dashboard/app.py` directly puts the dashboard/ folder on sys.path and
shadows the top-level `dashboard` package. Importing via this root module keeps
repo-root resolution working for `dashboard.components.*` imports.
"""

from __future__ import annotations

import runpy
from pathlib import Path

# Execute the dashboard app as __main__ with repo root already on sys.path
runpy.run_path(str(Path(__file__).resolve().parent / "dashboard" / "app.py"), run_name="__main__")
