#!/usr/bin/env python3
"""
Q4 2024 Dunkelflaute smoke test, then full walk-forward.

After extreme-threshold / routing / neg-classifier fixes:
  1) Run only windows whose forecast start is in Q4 2024
  2) Report Q4 MAE + directional accuracy
  3) Run full 2024 walk-forward and report final MAE
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings
from src.utils import setup_logging, write_json
from src.validation import WalkForwardValidator, MIN_TRAIN_DAYS

logger = logging.getLogger("cobblestone.q4_smoke")


def _load_panel():
    settings = get_settings()
    X = pd.read_parquet(settings.features_path)
    y = pd.read_parquet(settings.data_processed / "features_with_target.parquet")["da_price"]
    regime_path = settings.data_processed / "master_with_regimes.parquet"
    regime = pd.read_parquet(regime_path)["price_regime"] if regime_path.exists() else None
    return X, y, regime


def _clear_windows() -> None:
    settings = get_settings()
    for p in settings.walk_forward_splits.glob("window_*.parquet"):
        p.unlink()


def _directional_accuracy(df: pd.DataFrame) -> float:
    """Hour-ahead direction vs previous actual."""
    d_true = df["y_true"].diff()
    d_pred = df["y_pred"] - df["y_true"].shift(1)
    m = d_true.notna() & d_pred.notna() & (d_true != 0)
    if not m.any():
        return float("nan")
    return float(100.0 * ((np.sign(d_true.loc[m]) == np.sign(d_pred.loc[m])).mean()))


def run_q4_smoke() -> dict:
    """Fit/forecast only Q4 2024 weekly windows."""
    from src.model import EXTREME_PROB_THRESHOLD, EXTREME_PRICE_PERCENTILE, USE_PRICE_TRANSFORM

    settings = get_settings()
    print(
        f"Q4 smoke — extreme=p{EXTREME_PRICE_PERCENTILE:.0f} | "
        f"route>{EXTREME_PROB_THRESHOLD} | log_transform={USE_PRICE_TRANSFORM} | "
        f"min_train={MIN_TRAIN_DAYS}d"
    )
    _clear_windows()
    X, y, regime = _load_panel()

    # Monkey-patch window builder to Q4 2024 only
    v = WalkForwardValidator(fast=True)
    q4_start = pd.Timestamp("2024-10-01", tz="UTC")
    q4_end = pd.Timestamp("2024-12-31 23:00:00", tz="UTC")
    all_windows = v._build_windows(
        pd.Timestamp(settings.test_start, tz="UTC"),
        pd.Timestamp(settings.end_date, tz="UTC") + pd.Timedelta(hours=23),
    )
    q4_windows = [w for w in all_windows if w[1] >= q4_start and w[1] <= q4_end]
    print(f"  Q4 windows: {len(q4_windows)} (of {len(all_windows)} full-year)")

    # Temporarily restrict by overriding _build_windows
    def _q4_only(test_start, test_end):
        return q4_windows

    v._build_windows = _q4_only  # type: ignore[method-assign]
    t0 = time.perf_counter()
    results = v.run(X, y, regime, resume=False, tune=False)
    elapsed = time.perf_counter() - t0

    df = results.dropna(subset=["y_true", "y_pred"])
    mae = float((df["y_true"] - df["y_pred"]).abs().mean())
    dir_acc = _directional_accuracy(df)
    # Nov / Dec splits
    nov = df.loc["2024-11"]
    dec = df.loc["2024-12"]
    oct_ = df.loc["2024-10"]
    pct_extreme_train_logged = None
    # Routing share
    routed = float(df["used_extreme_model"].mean()) if "used_extreme_model" in df.columns else None
    thr = float(df["extreme_high_eur"].median()) if "extreme_high_eur" in df.columns else None

    report = {
        "period": "Q4_2024",
        "n_hours": int(len(df)),
        "MAE": mae,
        "directional_accuracy_pct": dir_acc,
        "mae_oct": float((oct_["y_true"] - oct_["y_pred"]).abs().mean()) if len(oct_) else None,
        "mae_nov": float((nov["y_true"] - nov["y_pred"]).abs().mean()) if len(nov) else None,
        "mae_dec": float((dec["y_true"] - dec["y_pred"]).abs().mean()) if len(dec) else None,
        "pct_routed_extreme": None if routed is None else 100.0 * routed,
        "median_extreme_threshold_eur": thr,
        "elapsed_s": round(elapsed, 1),
    }
    write_json(settings.forecasts_dir / "q4_2024_smoke.json", report)
    print(
        f"\n=== Q4 2024 SMOKE === MAE={mae:.2f} | dir_acc={dir_acc:.1f}% | "
        f"Oct={report['mae_oct']:.1f} Nov={report['mae_nov']:.1f} Dec={report['mae_dec']:.1f} | "
        f"routed={report['pct_routed_extreme']:.1f}% | thr≈{thr:.0f} EUR ({elapsed:.0f}s)\n"
    )
    return report


def run_full() -> dict:
    settings = get_settings()
    print("FULL walk-forward 2024…")
    _clear_windows()
    X, y, regime = _load_panel()
    t0 = time.perf_counter()
    WalkForwardValidator(fast=True).run(X, y, regime, resume=False, tune=False)
    elapsed = time.perf_counter() - t0
    metrics = json.loads((settings.forecasts_dir / "walk_forward_metrics.json").read_text())
    metrics["elapsed_s"] = round(elapsed, 1)
    write_json(settings.forecasts_dir / "final_metrics_after_extreme_fix.json", metrics)
    print(
        f"\n=== FULL 2024 === MAE={metrics['MAE']:.4f} skill={metrics.get('skill_vs_naive_pct'):+.1f}% "
        f"cov={100*(metrics.get('conformal_coverage_90_empirical') or 0):.1f}% "
        f"dir={metrics.get('directional_accuracy_pct'):.1f}% "
        f"neg_recall={100*(metrics.get('negative_price_recall') or 0):.1f}% "
        f"routed={metrics.get('n_routed_to_extreme_model')} "
        f"extreme_n={metrics.get('n_extreme_hours')} ({elapsed:.0f}s)\n"
    )
    keys = [
        "MAE",
        "RMSE",
        "skill_vs_naive_pct",
        "conformal_coverage_90_empirical",
        "directional_accuracy_pct",
        "negative_price_recall",
        "peak_mae",
        "offpeak_mae",
        "tail_mae_p95",
        "mae_regime_0",
        "mae_regime_1",
        "mae_regime_2",
        "mae_regime_3",
        "n_extreme_hours",
        "n_routed_to_extreme_model",
        "extreme_definition",
    ]
    print("===== FINAL METRICS =====")
    for k in keys:
        print(f"  {k}: {metrics.get(k)}")
    return metrics


def main() -> None:
    setup_logging()
    get_settings().validate_all()
    # Ensure transform stays off
    import src.model as m

    m.USE_PRICE_TRANSFORM = False
    run_q4_smoke()
    run_full()


if __name__ == "__main__":
    main()
