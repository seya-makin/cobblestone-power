"""
Streamlit Cloud entrypoint (optional).

Prefer Main file path = dashboard/app.py after the components/utils import fix.
This root module remains available if the Cloud app is pointed here instead.
"""

from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "dashboard" / "app.py"), run_name="__main__")
