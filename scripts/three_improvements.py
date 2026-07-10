#!/usr/bin/env python3
"""
Three targeted improvements — report MAE / recall after each.

1) True price_lag_8736h (no short-lag fill)
2) summer_solar_weekend for NegativePriceClassifier → full-year recall
3) MAE by calendar year 2022 / 2023 / 2024
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
from src.cleaning import DataCleaner
from src.features import FeatureEngineer
from src.regime import RegimeDetector
from src.utils import setup_logging, write_json
from src.validation import WalkForwardValidator, MIN_TRAIN_DAYS
import src.model as model_mod

logger = logging.getLogger("cobblestone.three_improvements")
REPORT: dict = {"steps": []}


def _clear_windows() -> None:
    settings = get_settings()
    for p in settings.walk_forward_splits.glob("window_*.parquet"):
        p.unlink()


def _rebuild_features() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    settings = get_settings()
    master = DataCleaner().build_master()
    regime_df = RegimeDetector().run(master)
    feat = FeatureEngineer().transform(regime_df)
    y = pd.read_parquet(settings.data_processed / "features_with_target.parquet")["da_price"]
    regime = regime_df["price_regime"]
    return feat, y, regime


def _metrics_from_results(df: pd.DataFrame) -> dict:
    d = df.dropna(subset=["y_true", "y_pred"])
    mae = float((d["y_true"] - d["y_pred"]).abs().mean())
    rmse = float(np.sqrt(((d["y_true"] - d["y_pred"]) ** 2).mean()))
    naive = float((d["y_true"] - d["y_naive"]).abs().mean()) if "y_naive" in d else None
    skill = None if not naive else 100.0 * (1.0 - mae / naive)
    neg = d["y_true"] < 0
    if neg.any() and "high_negative_price_risk" in d.columns:
        recall = float(d.loc[neg, "high_negative_price_risk"].astype(bool).mean())
    elif neg.any() and "negative_price_probability" in d.columns:
        recall = float((d.loc[neg, "negative_price_probability"] > 0.3).mean())
    else:
        recall = None
    return {
        "MAE": mae,
        "RMSE": rmse,
        "skill_vs_naive_pct": skill,
        "negative_price_recall": recall,
        "n_hours": int(len(d)),
        "n_negative": int(neg.sum()),
    }


def _run_wf(
    X: pd.DataFrame,
    y: pd.Series,
    regime: pd.Series,
    *,
    test_start: str,
    test_end: str,
    min_train_days: int,
    label: str,
) -> dict:
    _clear_windows()
    t0 = time.perf_counter()
    v = WalkForwardValidator(fast=True)
    results = v.run(
        X,
        y,
        regime,
        resume=False,
        tune=False,
        test_start=pd.Timestamp(test_start, tz="UTC"),
        test_end=pd.Timestamp(test_end, tz="UTC") + pd.Timedelta(hours=23),
        min_train_days=min_train_days,
    )
    elapsed = time.perf_counter() - t0
    m = _metrics_from_results(results)
    m["step"] = label
    m["elapsed_s"] = round(elapsed, 1)
    m["test_start"] = test_start
    m["test_end"] = test_end
    REPORT["steps"].append(m)
    write_json(get_settings().forecasts_dir / "three_improvements.json", REPORT)
    print(
        f"\n=== {label} === MAE={m['MAE']:.4f} skill={m.get('skill_vs_naive_pct') or 0:+.1f}% "
        f"neg_recall={100*(m.get('negative_price_recall') or 0):.1f}% "
        f"n={m['n_hours']} ({elapsed:.0f}s)\n"
    )
    return m


def main() -> None:
    setup_logging()
    settings = get_settings()
    settings.validate_all()
    model_mod.USE_PRICE_TRANSFORM = False

    print("Rebuilding features (true price_lag_8736h + summer_solar_weekend)…")
    feat, y, regime = _rebuild_features()
    assert "price_lag_8736h" in feat.columns
    assert "summer_solar_weekend" in feat.columns

    # Verify annual lag quality (true shift, post warm-up)
    raw_lag = y.shift(8736)
    aligned = pd.concat([y.rename("y"), raw_lag.rename("lag")], axis=1).dropna()
    corr = float(aligned["y"].corr(aligned["lag"]))
    nan_after = float(feat["price_lag_8736h"].iloc[8736:].isna().mean())
    print(f"  price_lag_8736h: NaN after warm-up={nan_after*100:.2f}% | corr(true shift)={corr:.4f}")
    REPORT["lag8736"] = {"corr_true_shift": corr, "nan_after_warmup": nan_after}

    # —— STEP 1: annual lag (exclude summer features from matrix for clean attribution) ——
    drop_summer = [c for c in ("summer_solar_weekend", "month_in_summer") if c in feat.columns]
    X1 = feat.drop(columns=drop_summer)
    # Classifier without summer features
    orig_neg_feats = list(model_mod.NEGATIVE_PRICE_FEATURES)
    model_mod.NEGATIVE_PRICE_FEATURES = [
        f for f in orig_neg_feats if f not in ("summer_solar_weekend", "month_in_summer")
    ]
    print("STEP 1 — Walk-forward 2024 with true price_lag_8736h…")
    m1 = _run_wf(
        X1, y, regime,
        test_start="2024-01-01",
        test_end="2024-12-31",
        min_train_days=MIN_TRAIN_DAYS,
        label="STEP1_price_lag_8736h",
    )

    # —— STEP 2: summer_solar_weekend in classifier + feature matrix ——
    model_mod.NEGATIVE_PRICE_FEATURES = orig_neg_feats
    print(
        "STEP 2 — Retrain with summer_solar_weekend "
        f"(classifier features={model_mod.NEGATIVE_PRICE_FEATURES})…"
    )
    m2 = _run_wf(
        feat, y, regime,
        test_start="2024-01-01",
        test_end="2024-12-31",
        min_train_days=MIN_TRAIN_DAYS,
        label="STEP2_summer_solar_weekend",
    )
    print(f"  Full-year negative_price_recall = {100*(m2.get('negative_price_recall') or 0):.1f}%")

    # Summer-only recall diagnostic
    res = pd.read_parquet(settings.forecasts_dir / "walk_forward_results.parquet")
    res = res.dropna(subset=["y_true", "y_pred"])
    summer = res.index.month.isin([4, 5, 6, 7, 8, 9])
    neg = res["y_true"] < 0
    if "high_negative_price_risk" in res.columns:
        flag = res["high_negative_price_risk"].astype(bool)
    else:
        flag = res["negative_price_probability"] > 0.3
    summer_recall = float(flag.loc[neg & summer].mean()) if (neg & summer).any() else None
    winter_recall = float(flag.loc[neg & ~summer].mean()) if (neg & ~summer).any() else None
    REPORT["recall_split"] = {
        "full_year": m2.get("negative_price_recall"),
        "summer_apr_sep": summer_recall,
        "winter_oct_mar": winter_recall,
        "n_neg_summer": int((neg & summer).sum()),
        "n_neg_winter": int((neg & ~summer).sum()),
    }
    print(
        f"  Recall summer={100*(summer_recall or 0):.1f}% "
        f"winter={100*(winter_recall or 0):.1f}%"
    )

    # —— STEP 3: MAE by year ——
    print("STEP 3 — MAE by calendar year…")
    year_specs = [
        ("2022", "2022-01-01", "2022-12-31", 90),   # limited history from Jan 2022
        ("2023", "2023-01-01", "2023-12-31", 300),
        ("2024", "2024-01-01", "2024-12-31", MIN_TRAIN_DAYS),
    ]
    by_year = {}
    for year, ts, te, mind in year_specs:
        print(f"  Walk-forward test year {year} (min_train={mind}d)…")
        m = _run_wf(
            feat, y, regime,
            test_start=ts,
            test_end=te,
            min_train_days=mind,
            label=f"YEAR_{year}",
        )
        by_year[year] = m

    REPORT["mae_by_year"] = {
        y: {"MAE": by_year[y]["MAE"], "n_hours": by_year[y]["n_hours"],
            "skill_vs_naive_pct": by_year[y].get("skill_vs_naive_pct"),
            "negative_price_recall": by_year[y].get("negative_price_recall")}
        for y in by_year
    }
    write_json(settings.forecasts_dir / "three_improvements.json", REPORT)

    print("\n===== SUMMARY =====")
    print(f"  STEP1 price_lag_8736h     MAE={m1['MAE']:.4f}")
    print(f"  STEP2 summer_solar_weekend MAE={m2['MAE']:.4f}  recall={100*(m2.get('negative_price_recall') or 0):.1f}%")
    for y, m in by_year.items():
        print(f"  YEAR {y}                  MAE={m['MAE']:.4f}  n={m['n_hours']}")
    print(
        "\nNote: 2022 MAE is expected to be much higher (Ukraine gas crisis / "
        "extreme price levels); 2023–2024 reflect the post-crisis regime."
    )


if __name__ == "__main__":
    main()
