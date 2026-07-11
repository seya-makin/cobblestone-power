"""
Cobblestone Power — FastAPI REST surface over existing pipeline artefacts.

Reads forecast / signal / metrics / submission outputs only. Does not retrain.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from config.settings import PIPELINE_VERSION, PROJECT_ROOT, get_settings
from src.regime import REGIME_NAMES

app = FastAPI(
    title="Cobblestone Power Forecast API",
    description="REST access to DE-LU day-ahead forecast, trading signal, curve view, and metrics.",
    version=PIPELINE_VERSION,
)

SETTINGS = get_settings()
FORECASTS = SETTINGS.forecasts_dir
ROOT = PROJECT_ROOT


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Missing artefact: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid JSON in {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail=f"Expected object in {path.name}")
    return data


def _round(x: Any, nd: int = 1) -> Optional[float]:
    if x is None:
        return None
    try:
        return round(float(x), nd)
    except (TypeError, ValueError):
        return None


def _period_block(block: Dict[str, Any]) -> Dict[str, Any]:
    baseload = float(block.get("baseload", 0.0))
    peak = float(block.get("peak", 0.0))
    spread = float(block.get("peak_base_spread", peak - baseload))
    return {
        "baseload": _round(baseload, 1),
        "peak": _round(peak, 1),
        "spread": _round(spread, 1),
        "ci_80_low": _round(block.get("conformal_80_low"), 1),
        "ci_80_high": _round(block.get("conformal_80_high"), 1),
    }


def _hourly_from_walk_forward() -> tuple[str, List[Dict[str, Any]]]:
    """Build hourly forecast rows from the last delivery day in walk_forward_results."""
    path = FORECASTS / "walk_forward_results.parquet"
    if not path.exists():
        raise HTTPException(status_code=404, detail="walk_forward_results.parquet not found")
    cols = ["y_pred", "conformal_90_low", "conformal_90_high", "price_regime"]
    wf = pd.read_parquet(path, columns=[c for c in cols if True])
    # Keep only columns that exist
    cols = [c for c in cols if c in wf.columns]
    wf = wf[cols] if cols else wf
    if wf.empty:
        raise HTTPException(status_code=404, detail="walk_forward_results is empty")
    last = wf.index.max().normalize()
    day = wf.loc[last : last + pd.Timedelta(hours=23)].copy()
    if day.empty:
        day = wf.tail(24)
    rows: List[Dict[str, Any]] = []
    for ts, row in day.iterrows():
        ts_utc = pd.Timestamp(ts)
        if ts_utc.tzinfo is None:
            ts_utc = ts_utc.tz_localize("UTC")
        else:
            ts_utc = ts_utc.tz_convert("UTC")
        regime = row.get("price_regime") if "price_regime" in day.columns else None
        rows.append(
            {
                "hour": ts_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "y_pred": _round(row.get("y_pred"), 1),
                "conformal_90_low": _round(row.get("conformal_90_low"), 1),
                "conformal_90_high": _round(row.get("conformal_90_high"), 1),
                "regime": int(regime) if regime is not None and pd.notna(regime) else None,
            }
        )
    forecast_date = str(pd.Timestamp(day.index.min()).date())
    return forecast_date, rows


@app.get("/health")
def health() -> Dict[str, Any]:
    """System status and headline model MAE."""
    metrics = _read_json(FORECASTS / "walk_forward_metrics.json")
    commentary = FORECASTS / "commentary_latest.json"
    signal = FORECASTS / "trading_signal.json"
    last_run = None
    if commentary.exists():
        last_run = _read_json(commentary).get("generated_at")
    if last_run is None and signal.exists():
        last_run = _read_json(signal).get("signal_generated_at")

    data_through = None
    wf_path = FORECASTS / "walk_forward_results.parquet"
    if wf_path.exists():
        idx_max = pd.read_parquet(wf_path, columns=["y_pred"]).index.max()
        data_through = str(pd.Timestamp(idx_max).date())

    mae = metrics.get("MAE")
    return {
        "status": "ok",
        "last_run": last_run,
        "data_through": data_through,
        "model_version": PIPELINE_VERSION,
        "mae": _round(mae, 2) if mae is not None else None,
    }


@app.get("/forecast/latest")
def forecast_latest() -> Dict[str, Any]:
    """Latest hourly DA forecast with conformal bands and summary."""
    curve = _read_json(FORECASTS / "latest_forecast.json")
    signal = _read_json(FORECASTS / "trading_signal.json")
    forecast_date, hourly = _hourly_from_walk_forward()

    regime_id = signal.get("dominant_regime", 2)
    try:
        regime_id_int = int(regime_id)
    except (TypeError, ValueError):
        regime_id_int = 2

    neg_level = signal.get("negative_price_risk_level")
    if not neg_level:
        neg_p = float(signal.get("negative_price_risk") or 0.0)
        neg_level = "ELEVATED" if neg_p >= 0.3 else "LOW"

    tomorrow = curve.get("tomorrow") or {}
    if regime_id_int == 1:
        regime_label = "LOW"
    elif regime_id_int == 0:
        regime_label = "NEGATIVE/ZERO"
    elif regime_id_int == 3:
        regime_label = "HIGH/DUNKELFLAUTE"
    else:
        regime_label = REGIME_NAMES.get(regime_id_int, "NORMAL")

    summary = {
        "baseload_eur_mwh": _round(tomorrow.get("baseload", signal.get("expected_da_baseload")), 1),
        "peak_eur_mwh": _round(tomorrow.get("peak", signal.get("expected_da_peak")), 1),
        "dominant_regime": regime_label,
        "negative_price_risk": str(neg_level),
        "dunkelflaute_risk": _round(signal.get("dunkelflaute_risk"), 2),
    }

    return {
        "forecast_date": forecast_date,
        "hourly_forecast": hourly,
        "summary": summary,
    }


@app.get("/signal/latest")
def signal_latest() -> Dict[str, Any]:
    """Current trading signal with conviction and rationale."""
    signal = _read_json(FORECASTS / "trading_signal.json")
    strength = signal.get("signal_strength")
    try:
        strength_pct = int(round(float(strength) * 100)) if strength is not None else None
    except (TypeError, ValueError):
        strength_pct = None

    return {
        "signal_date": signal.get("signal_date"),
        "direction": signal.get("direction"),
        "conviction": signal.get("conviction"),
        "strength_pct": strength_pct,
        "instrument": signal.get("suggested_instrument"),
        "rationale": signal.get("trading_rationale"),
        "invalidation_conditions": signal.get("invalidation_conditions") or [],
    }


@app.get("/curve/latest")
def curve_latest() -> Dict[str, Any]:
    """Prompt day / week / month delivery-period view."""
    curve = _read_json(FORECASTS / "latest_forecast.json")
    return {
        "prompt_day": _period_block(curve.get("tomorrow") or {}),
        "prompt_week": _period_block(curve.get("next_week") or {}),
        "prompt_month": _period_block(curve.get("next_month") or {}),
    }


@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    """Full walk-forward validation metrics."""
    return _read_json(FORECASTS / "walk_forward_metrics.json")


@app.get("/submission")
def submission() -> FileResponse:
    """Download submission.csv."""
    path = ROOT / "submission.csv"
    if not path.exists():
        alt = SETTINGS.submission_csv
        path = alt if alt.exists() else path
    if not path.exists():
        raise HTTPException(status_code=404, detail="submission.csv not found")
    return FileResponse(
        path=str(path),
        media_type="text/csv",
        filename="submission.csv",
    )
