#!/usr/bin/env python3
"""
Sequential MAE improvement steps on real SMARD data.

Reports metrics after each step so contributions are attributable.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings
from src.utils import setup_logging, write_json

logger = logging.getLogger("cobblestone.mae_steps")
RESULTS: list[dict] = []


def _mae_from_metrics() -> float:
    settings = get_settings()
    m = json.loads((settings.forecasts_dir / "walk_forward_metrics.json").read_text())
    return float(m["MAE"])


def _load_panel():
    settings = get_settings()
    X = pd.read_parquet(settings.features_path)
    y = pd.read_parquet(settings.data_processed / "features_with_target.parquet")["da_price"]
    regime_path = settings.data_processed / "master_with_regimes.parquet"
    regime = (
        pd.read_parquet(regime_path)["price_regime"]
        if regime_path.exists()
        else None
    )
    return X, y, regime


def _clear_windows() -> None:
    settings = get_settings()
    for p in settings.walk_forward_splits.glob("window_*.parquet"):
        p.unlink()


def _run_validate(fast: bool = True, label: str = "") -> dict:
    from src.validation import WalkForwardValidator

    _clear_windows()
    X, y, regime = _load_panel()
    t0 = time.perf_counter()
    WalkForwardValidator(fast=fast).run(X, y, regime, resume=False, tune=False)
    elapsed = time.perf_counter() - t0
    settings = get_settings()
    metrics = json.loads((settings.forecasts_dir / "walk_forward_metrics.json").read_text())
    entry = {
        "step": label,
        "MAE": metrics.get("MAE"),
        "RMSE": metrics.get("RMSE"),
        "skill_vs_naive_pct": metrics.get("skill_vs_naive_pct"),
        "conformal_coverage_90_empirical": metrics.get("conformal_coverage_90_empirical"),
        "directional_accuracy_pct": metrics.get("directional_accuracy_pct"),
        "negative_price_recall": metrics.get("negative_price_recall"),
        "elapsed_s": round(elapsed, 1),
    }
    RESULTS.append(entry)
    write_json(settings.forecasts_dir / "mae_improvement_steps.json", {"steps": RESULTS})
    print(f"\n=== {label} === MAE={entry['MAE']:.4f} skill={entry['skill_vs_naive_pct']:+.1f}% "
          f"cov={100*(entry['conformal_coverage_90_empirical'] or 0):.1f}% "
          f"neg_recall={100*(entry['negative_price_recall'] or 0):.1f}% ({elapsed:.0f}s)\n")
    return metrics


def step1_optuna(skip_tune: bool = False) -> None:
    """50-trial Optuna on real data, then validate with best params."""
    from src.model import XGBoostPointForecaster

    settings = get_settings()
    best_path = settings.models_dir / "best_params.json"
    if skip_tune and best_path.exists():
        best = json.loads(best_path.read_text()).get("best_params")
        print(f"STEP 1 — Reusing Optuna best_params (skip tune): {best}")
    else:
        print("STEP 1 — Optuna 50 trials (walk-forward MAE objective)…")
        X, y, _ = _load_panel()
        # Tune on all history before 2024
        train = X.index < "2024-01-01"
        X_tr, y_tr = X.loc[train], y.loc[train]
        # Drop rows with any NaN features for stable tuning
        m = X_tr.notna().all(axis=1) & y_tr.notna()
        X_tr, y_tr = X_tr.loc[m], y_tr.loc[m]
        print(f"  Tuning panel: {len(X_tr)} hours ending {X_tr.index.max()}")
        tuner = XGBoostPointForecaster()
        best = tuner.hyperparameter_tune(X_tr, y_tr, n_trials=50, walk_forward=True)
        print(f"  Best params: {best}")
    _run_validate(fast=True, label="STEP1_optuna")


def step2_lags() -> None:
    """Verify price_lag_168h, add 672h/8736h, rebuild features, validate."""
    from src.cleaning import DataCleaner
    from src.features import FeatureEngineer
    from src.regime import RegimeDetector

    print("STEP 2 — Price lag features…")
    settings = get_settings()
    master = DataCleaner().build_master()
    regime_df = RegimeDetector().run(master)
    feat = FeatureEngineer().transform(regime_df)
    target = pd.read_parquet(settings.data_processed / "features_with_target.parquet")["da_price"]

    # Verify price_lag_168h
    lag = feat["price_lag_168h"]
    after = lag.iloc[168:]
    nan_pct = float(after.isna().mean())
    aligned = pd.concat([target.rename("y"), lag.rename("lag168")], axis=1).dropna()
    corr168 = float(aligned["y"].corr(aligned["lag168"]))
    print(f"  price_lag_168h: NaN after first 168h = {nan_pct*100:.2f}% | corr(target)={corr168:.4f}")
    assert nan_pct < 0.01, "price_lag_168h has unexpected NaNs after warm-up"
    assert corr168 > 0.7, f"price_lag_168h correlation {corr168:.3f} < 0.7 — lag construction broken"

    # Top-10 feature correlations
    common = feat.index.intersection(target.index)
    F = feat.loc[common].select_dtypes("number")
    y = target.loc[common]
    corrs = F.corrwith(y).abs().sort_values(ascending=False).head(10)
    print("  Top-10 |corr| with da_price:")
    for name, c in corrs.items():
        print(f"    {name:40s} {c:.4f}")
    write_json(
        settings.forecasts_dir / "feature_correlations_top10.json",
        {"price_lag_168h_corr": corr168, "top10": corrs.to_dict()},
    )
    _run_validate(fast=True, label="STEP2_lags")


def step3_thermal() -> None:
    """Fetch real SMARD gas/lignite generation, rebuild features, validate."""
    from src.cleaning import DataCleaner
    from src.features import FeatureEngineer
    from src.regime import RegimeDetector
    from src.smard_ingestion import SMARDIngester

    print("STEP 3 — SMARD thermal mix features…")
    # Re-ingest to pull real gen_gas / gen_lignite / gen_coal
    SMARDIngester().fetch_all()
    master = DataCleaner().build_master()
    print(
        f"  gen_gas mean={master['gen_gas'].mean():.0f} MW | "
        f"gen_lignite mean={master['gen_lignite'].mean():.0f} MW"
    )
    regime_df = RegimeDetector().run(master)
    FeatureEngineer().transform(regime_df)
    feat = pd.read_parquet(get_settings().features_path)
    assert "thermal_share" in feat.columns
    assert "gas_plus_lignite_mw" in feat.columns
    _run_validate(fast=True, label="STEP3_thermal")


def step4_walkforward() -> None:
    """MIN_TRAIN_DAYS=400 already in code; re-validate."""
    from src.validation import MIN_TRAIN_DAYS

    print(f"STEP 4 — Walk-forward min train days = {MIN_TRAIN_DAYS} (weekly 168h forecasts)…")
    assert MIN_TRAIN_DAYS >= 400
    _run_validate(fast=True, label="STEP4_min_train_400d")


def step5_transform() -> None:
    """Enable signed log1p price transform and re-validate."""
    import src.model as model_mod

    print("STEP 5 — Signed log1p price transform…")
    before = RESULTS[-1]["MAE"] if RESULTS else None
    model_mod.USE_PRICE_TRANSFORM = True
    print(f"  USE_PRICE_TRANSFORM={model_mod.USE_PRICE_TRANSFORM} | MAE before={before}")
    metrics = _run_validate(fast=True, label="STEP5_log_transform")
    print(f"  MAE after transform={metrics['MAE']:.4f} (before={before})")


def step6_neg_classifier() -> None:
    """NegativePriceClassifier weight already ~18.2; re-validate and check recall."""
    from src.model import NEGATIVE_PRICE_DEFAULT_SCALE_POS_WEIGHT

    print(
        f"STEP 6 — NegativePriceClassifier scale_pos_weight prior={NEGATIVE_PRICE_DEFAULT_SCALE_POS_WEIGHT:.2f}…"
    )
    metrics = _run_validate(fast=True, label="STEP6_neg_classifier")
    recall = metrics.get("negative_price_recall") or 0
    print(f"  negative_price_recall={100*recall:.1f}% (target ≥ 90%)")


def step7_final() -> None:
    """Full walk-forward (non-fast if feasible), submission, figures."""
    from run_pipeline import cmd_forecast, cmd_submission

    print("STEP 7 — Final walk-forward + submission + figures…")
    # Use tuned n_estimators (capped at 200 in fast for runtime; bump slightly)
    metrics = _run_validate(fast=True, label="STEP7_final")
    cmd_forecast()
    cmd_submission()
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
    ]
    print("\n===== FINAL METRICS =====")
    for k in keys:
        v = metrics.get(k)
        print(f"  {k}: {v}")
    write_json(
        get_settings().forecasts_dir / "final_metrics_table.json",
        {k: metrics.get(k) for k in keys},
    )


def main() -> None:
    import argparse

    setup_logging()
    settings = get_settings()
    settings.validate_all()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-optuna",
        action="store_true",
        help="Reuse existing best_params.json (skip 50-trial search)",
    )
    parser.add_argument(
        "--from-step",
        type=int,
        default=1,
        help="Start from step N (1-7)",
    )
    args = parser.parse_args()

    # Fixed baseline from pre-improvement SMARD run
    RESULTS.append(
        {
            "step": "BASELINE_pre_improvements",
            "MAE": 27.886959628443233,
            "RMSE": 39.927469195976755,
            "skill_vs_naive_pct": 15.002385578773037,
            "conformal_coverage_90_empirical": 0.923155737704918,
            "directional_accuracy_pct": 81.80783242258653,
            "negative_price_recall": 0.8665207877461707,
        }
    )
    print("BASELINE MAE=27.8870")

    steps = {
        1: lambda: step1_optuna(skip_tune=args.skip_optuna),
        2: step2_lags,
        3: step3_thermal,
        4: step4_walkforward,
        5: step5_transform,
        6: step6_neg_classifier,
        7: step7_final,
    }
    for n in range(args.from_step, 8):
        steps[n]()

    print("\n===== STEP SUMMARY =====")
    for r in RESULTS:
        print(f"  {r['step']:30s} MAE={r.get('MAE')}")


if __name__ == "__main__":
    main()
