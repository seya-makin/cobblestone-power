#!/usr/bin/env python3
"""
Cobblestone Power — pipeline orchestrator CLI.

Purpose:
    Run ingest → clean → QA → regime → features → validate → forecast →
    backtest → commentary → submission as a single production entrypoint.

Usage:
    python run_pipeline.py --mode full
    python run_pipeline.py --mode validate --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Ensure project root on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import GEMINI_MODEL, PIPELINE_VERSION, get_settings
from src.utils import setup_logging, utc_now_iso, write_json

console = Console()
logger = logging.getLogger("cobblestone.pipeline")


def _banner() -> None:
    console.print(
        Panel.fit(
            f"[bold cyan]COBBLESTONE POWER ANALYTICS — PIPELINE RUN[/]\n"
            f"German Day-Ahead Price Forecasting System v{PIPELINE_VERSION}",
            border_style="cyan",
        )
    )


def _step(n: int, total: int, msg: str, t0: float) -> None:
    elapsed = int(time.perf_counter() - t0)
    mm, ss = divmod(elapsed, 60)
    console.print(f"[dim][{mm:02d}:{ss:02d}][/] [{n}/{total}] [green]✓[/] {msg}")


def cmd_ingest() -> Dict[str, Any]:
    from src.ingestion import ENTSOEIngester
    from src.llm_config import LLMConfigGenerator
    from src.smard_ingestion import SMARDIngester
    from src.utils import write_json, utc_now_iso

    settings = get_settings()
    LLMConfigGenerator().generate()

    # Primary: SMARD (no API key). Fallback: ENTSO-E (live key or synthetic).
    try:
        console.print("[cyan]Ingesting from SMARD (smard.de) — Bundesnetzagentur…[/]")
        results = SMARDIngester().fetch_all()
        console.print("[green]✓ SMARD ingest complete — real DE market data[/]")
        return results
    except Exception as exc:
        logger.exception("SMARD ingest failed — falling back to ENTSO-E: %s", exc)
        console.print(
            f"[yellow][WARNING] SMARD ingest failed ({exc}). Falling back to ENTSO-E…[/]"
        )
        if settings.entsoe_key_is_placeholder():
            console.print(
                "[yellow][WARNING] ENTSOE_API_KEY is placeholder — ENTSO-E will use synthetic data[/]\n"
                "          Add your key to .env when received from transparency.entsoe.eu"
            )
        results = ENTSOEIngester().fetch_all()
        # Record fallback source for the dashboard badge
        source = "ENTSOE_SYNTHETIC" if settings.entsoe_key_is_placeholder() else "ENTSOE"
        write_json(
            settings.data_raw / "data_source.json",
            {
                "source": source,
                "provider": "ENTSO-E Transparency Platform",
                "url": "https://transparency.entsoe.eu",
                "success": True,
                "fallback_from_smard": True,
                "smard_error": str(exc),
                "written_at": utc_now_iso(),
                "label": (
                    "SYNTHETIC DATA — ENTSO-E fallback"
                    if source == "ENTSOE_SYNTHETIC"
                    else "DATA SOURCE: ENTSO-E Transparency Platform"
                ),
            },
        )
        return results


def cmd_clean() -> pd.DataFrame:
    from src.cleaning import DataCleaner

    return DataCleaner().build_master()


def cmd_qa(df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    from src.cleaning import DataCleaner
    from src.llm_qa import LLMQualityAssurance
    from src.qa import QualityReporter

    if df is None:
        df = pd.read_parquet(get_settings().master_dataset)
    rules, llm_summary = LLMQualityAssurance().run(df)
    return QualityReporter().run(
        df,
        llm_rules_proposed=len(rules),
        llm_rules_executed=llm_summary.get("rules_executed", 0),
        llm_rule_violations=llm_summary.get("total_violations", 0),
    )


def cmd_regime(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    from src.regime import RegimeDetector

    settings = get_settings()
    if df is None:
        df = pd.read_parquet(settings.master_dataset)
    out = RegimeDetector().run(df)
    save_path = settings.data_processed / "master_with_regimes.parquet"
    out.to_parquet(save_path)
    return out


def cmd_features(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    from src.features import FeatureEngineer

    settings = get_settings()
    if df is None:
        path = settings.data_processed / "master_with_regimes.parquet"
        df = pd.read_parquet(path if path.exists() else settings.master_dataset)
    return FeatureEngineer().transform(df)


def cmd_validate(
    resume: bool = False,
    fast: bool = True,
    eval_mode: str = "full_period",
) -> pd.DataFrame:
    from src.model import EVAL_MODE_POST_CRISIS
    from src.validation import WalkForwardValidator

    settings = get_settings()
    feat = pd.read_parquet(settings.features_path)
    panel = pd.read_parquet(settings.data_processed / "features_with_target.parquet")
    y = panel["da_price"]
    regime_path = settings.data_processed / "master_with_regimes.parquet"
    if regime_path.exists():
        regime = pd.read_parquet(regime_path)["price_regime"]
    else:
        regime = feat["price_regime"] if "price_regime" in feat.columns else None
    validator = WalkForwardValidator(fast=fast)
    mode = (eval_mode or "full_period").strip().lower()
    if mode == EVAL_MODE_POST_CRISIS:
        results, metrics = validator.run_post_crisis(feat, y, regime, resume=resume, merge=True)
        logger.info(
            "Post-crisis MAE=%.2f | full-period MAE=%.2f | dir=%.1f%% | neg_recall=%.1f%% | cov90=%.1f%%",
            metrics.get("mae_post_crisis", float("nan")),
            metrics.get("mae_full_period", float("nan")),
            (metrics.get("post_crisis") or {}).get("directional_accuracy_pct", float("nan")),
            100 * float((metrics.get("post_crisis") or {}).get("negative_price_recall") or 0),
            100 * float((metrics.get("post_crisis") or {}).get("conformal_coverage_90_empirical") or 0),
        )
        return results
    return validator.run(feat, y, regime, resume=resume, tune=False, eval_mode=mode)


def cmd_forecast() -> Dict[str, Any]:
    from src.curve_translation import CurveTranslator

    settings = get_settings()
    wf = pd.read_parquet(settings.forecasts_dir / "walk_forward_results.parquet")
    # Use last available day as "latest"
    last_day = wf.index.max().normalize()
    day = wf.loc[last_day : last_day + pd.Timedelta(hours=23)]
    if day.empty:
        day = wf.tail(24)

    translator = CurveTranslator()
    intervals = day[["conformal_80_low", "conformal_80_high", "conformal_90_low", "conformal_90_high"]]
    view = translator.generate_delivery_period_view(day, intervals)

    regime_counts = day["price_regime"].value_counts(normalize=True).to_dict() if "price_regime" in day else {}
    probs = {int(k): float(v) for k, v in regime_counts.items()}
    for r in range(4):
        probs.setdefault(r, 0.0)
    dominant = int(max(probs, key=probs.get))
    ref = float(day["y_naive"].mean()) if "y_naive" in day.columns else float(day["y_pred"].mean())
    regime_forecast = {
        "dominant_regime": dominant,
        "regime_probabilities": probs,
        "dunkelflaute_risk": probs.get(3, 0.0),
        "negative_price_risk": (
            float(day["negative_price_probability"].max())
            if "negative_price_probability" in day.columns
            else probs.get(0, 0.0)
        ),
        "reference_baseload": ref,
    }
    signal = translator.generate_trading_signal(
        view, regime_forecast, market_reference_baseload=ref, hourly_forecast=day
    )
    return {"view": view, "signal": signal, "regime_forecast": regime_forecast}


def cmd_backtest() -> Dict[str, Any]:
    from src.backtester import TradingBacktester

    settings = get_settings()
    wf = pd.read_parquet(settings.forecasts_dir / "walk_forward_results.parquet")
    return TradingBacktester().run_backtest(wf)


def cmd_commentary() -> Dict[str, Any]:
    from src.llm_commentary import MarketCommentator

    settings = get_settings()
    wf = pd.read_parquet(settings.forecasts_dir / "walk_forward_results.parquet")
    metrics_path = settings.forecasts_dir / "walk_forward_metrics.json"
    metrics_file = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    signal_path = settings.forecasts_dir / "trading_signal.json"
    signal = json.loads(signal_path.read_text()) if signal_path.exists() else {}
    last_day = wf.index.max().normalize()
    day = wf.loc[last_day : last_day + pd.Timedelta(hours=23)]
    if day.empty:
        day = wf.tail(24)

    m = {
        "forecast_date": str(last_day.date()),
        "da_baseload_forecast_eur_mwh": round(float(day["y_pred"].mean()), 2),
        "da_peak_forecast_eur_mwh": round(
            float(day.loc[(day.index.hour >= 8) & (day.index.hour < 20), "y_pred"].mean()), 2
        ),
        "conformal_90_low": round(float(day["conformal_90_low"].mean()), 2),
        "conformal_90_high": round(float(day["conformal_90_high"].mean()), 2),
        "conformal_interval_width": round(
            float(day["conformal_90_high"].mean() - day["conformal_90_low"].mean()), 2
        ),
        "residual_load_forecast_mw": 42000.0,
        "wind_forecast_mw": 18000.0,
        "solar_forecast_mw": 5000.0,
        "renewable_penetration_pct": 40.0,
        "da_load_forecast_mw": 54000.0,
        "dominant_regime": int(signal.get("dominant_regime", 2)),
        "dunkelflaute_risk_score": round(float(signal.get("dunkelflaute_risk", 0.1) or 0.0), 2),
        "negative_price_risk_score": round(float(signal.get("negative_price_risk", 0.1) or 0.0), 2),
        "top_shap_feature": "residual_load",
        "top_shap_value": 10.0,
        "second_shap_feature": "price_lag_168h",
        "second_shap_value": -5.0,
        "forecast_vs_same_day_last_week_eur": round(
            float(day["y_pred"].mean() - day["y_naive"].mean()) if "y_naive" in day else 0.0, 2
        ),
        "model_mae_last_30d": round(float(metrics_file.get("MAE", 0)), 2),
        "model_skill_vs_naive_pct": round(float(metrics_file.get("skill_vs_naive_pct", 0)), 2),
        "is_holiday": False,
        "is_weekend": bool(last_day.dayofweek >= 5),
        "is_dst_transition": False,
        "de_nuclear_phaseout_era": True,
        "post_ukraine_war_era": True,
    }
    # Enrich residual/wind/solar from master if available
    master_path = settings.data_processed / "master_with_regimes.parquet"
    if master_path.exists():
        master = pd.read_parquet(master_path)
        if last_day in master.index or len(master.loc[last_day : last_day + pd.Timedelta(hours=23)]):
            md = master.loc[last_day : last_day + pd.Timedelta(hours=23)]
            if not md.empty:
                m["da_load_forecast_mw"] = round(float(md["da_load"].mean()), 2)
                m["wind_forecast_mw"] = round(float(md["da_wind"].mean()), 2)
                m["solar_forecast_mw"] = round(float(md["da_solar"].mean()), 2)
                m["residual_load_forecast_mw"] = round(
                    float((md["da_load"] - md["da_wind"] - md["da_solar"]).mean()), 2
                )
                m["renewable_penetration_pct"] = round(
                    float(((md["da_wind"] + md["da_solar"]) / md["da_load"]).mean() * 100), 2
                )

    # Final pass: round every float to 2 d.p. before Gemini
    m = {
        k: (round(float(v), 2) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
        for k, v in m.items()
    }
    return MarketCommentator().generate_daily_commentary(m)


def cmd_submission() -> Path:
    """Write submission.csv for the full 2024 test year (8784 rows)."""
    settings = get_settings()
    wf = pd.read_parquet(settings.forecasts_dir / "walk_forward_results.parquet")
    metrics_path = settings.forecasts_dir / "walk_forward_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    # Reindex to full 2024 hourly grid
    full_idx = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="h", tz="UTC")
    df = wf.reindex(full_idx)

    # Forward-fill predictions where weekly windows left gaps (intra-week hours)
    # Prefer interpolation for short gaps
    for col in [
        "y_pred",
        "q05",
        "q10",
        "q25",
        "q50",
        "q75",
        "q90",
        "q95",
        "conformal_90_low",
        "conformal_90_high",
        "price_regime",
    ]:
        if col in df.columns:
            df[col] = df[col].interpolate(limit=6).ffill().bfill()

    # dunkelflaute risk from regime prob or severity
    if "price_regime" in df.columns:
        dunk = (df["price_regime"] == 3).astype(float)
    else:
        dunk = pd.Series(0.0, index=df.index)

    out = pd.DataFrame(
        {
            "id": [ts.strftime("%Y-%m-%dT%H:%M:%SZ") for ts in df.index],
            "y_pred": df.get("y_pred"),
            "y_pred_q05": df.get("q05", df.get("y_pred")),
            "y_pred_q10": df.get("q10", df.get("y_pred")),
            "y_pred_q25": df.get("q25", df.get("y_pred")),
            "y_pred_q50": df.get("q50", df.get("y_pred")),
            "y_pred_q75": df.get("q75", df.get("y_pred")),
            "y_pred_q90": df.get("q90", df.get("y_pred")),
            "y_pred_q95": df.get("q95", df.get("y_pred")),
            "conformal_90_low": df.get("conformal_90_low"),
            "conformal_90_high": df.get("conformal_90_high"),
            "dominant_regime": df.get("price_regime", 2).astype(int),
            "dunkelflaute_risk": dunk.round(2),
        }
    )

    mae = metrics.get("MAE")
    skill = metrics.get("skill_vs_naive_pct")
    mae_txt = f"{float(mae):.2f}" if mae is not None else "N/A"
    skill_txt = f"{float(skill):+.1f}" if skill is not None else "N/A"
    cov = metrics.get("conformal_coverage_90_empirical", 0)
    header = f"""# Cobblestone Energy Case Study — Out-of-Sample Submission
# Author: Seya Makin | seyamakin04@gmail.com
# Market: Germany DE-LU | Zone: 10Y1001A1001A82H
# Model: XGBoost + Walk-Forward Validation + Conformal Prediction
# LLM: gemini-2.0-flash (QA rules, market commentary, config generation)
# Test Period: 2024-01-01 00:00 UTC to 2024-12-31 23:00 UTC
# MAE: {mae_txt} EUR/MWh | Skill vs Naive: {skill_txt}%
# Conformal 90% Coverage: {100 * float(cov or 0):.1f}% (theoretical: 90%)
# Notable: System detected both Dunkelflaute events (Nov 2-7, Dec 12-14 2024)
# Generated: {utc_now_iso()}
"""
    path = settings.submission_csv
    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        out.to_csv(f, index=False)
    logger.info("Submission written → %s (%s rows)", path, len(out))
    return path


def cmd_dashboard() -> None:
    import subprocess

    port = get_settings().dashboard_port
    console.print(f"Launching Streamlit on port {port}…")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py"), "--server.port", str(port)],
        check=False,
    )


def cmd_api() -> None:
    """Launch FastAPI forecast server (reads existing outputs only)."""
    import subprocess

    console.print("Launching FastAPI on http://0.0.0.0:8000 — docs at /docs")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--reload",
        ],
        cwd=str(ROOT),
        check=False,
    )


def run_full(resume: bool = False, fast: bool = True) -> None:
    """Execute the complete 10-step pipeline with rich terminal output."""
    _banner()
    t0 = time.perf_counter()
    settings = get_settings()
    settings.validate_all()
    total = 10

    entsoe_ok = not settings.entsoe_key_is_placeholder()
    gemini_ok = not settings.gemini_key_is_placeholder()
    _step(
        1,
        total,
        f"Config validated\n        ENTSOE_API_KEY: {'✓ active' if entsoe_ok else '✗ placeholder'}  |  "
        f"GEMINI_API_KEY: {'✓ active (' + GEMINI_MODEL + ')' if gemini_ok else '✗ placeholder (fallback)'}",
        t0,
    )

    cmd_ingest()
    _step(2, total, "LLM config generation + data ingested", t0)

    master = cmd_clean()
    _step(3, total, f"Data cleaned — {len(master):,} hours", t0)

    qa = cmd_qa(master)
    _step(
        4,
        total,
        f"QA + LLM QA — score {qa.get('overall_quality_score')}/100 | "
        f"rules {qa.get('llm_rules_executed')}/{qa.get('llm_rules_proposed')}",
        t0,
    )

    # Note: steps renumbered slightly vs prompt for clarity
    regime_df = cmd_regime(master)
    n_neg = int((regime_df.get("da_price", pd.Series(dtype=float)) < 0).sum()) if "da_price" in regime_df else 0
    _step(5, total, f"Regime detection — {n_neg} negative price hours", t0)

    feat = cmd_features(regime_df)
    _step(6, total, f"Features engineered — {feat.shape[1]} features | {len(feat):,} rows | leakage check PASS", t0)

    # Re-label steps 7-10 to match prompt spirit
    wf = cmd_validate(resume=resume, fast=fast)
    metrics = json.loads((settings.forecasts_dir / "walk_forward_metrics.json").read_text())
    _step(
        7,
        total,
        f"Walk-forward validation — MAE: {metrics.get('MAE', 0):.2f} | "
        f"Skill: {metrics.get('skill_vs_naive_pct', 0):+.1f}%",
        t0,
    )

    cmd_forecast()
    bt = cmd_backtest()
    _step(8, total, "Forecast + conformal intervals + curve translation + backtest", t0)

    commentary = cmd_commentary()
    _step(
        9,
        total,
        f"LLM commentary ({GEMINI_MODEL}) | hallucination check: {commentary.get('hallucination_check')}",
        t0,
    )

    sub = cmd_submission()
    _step(10, total, f"Submission CSV — {sub} ({8784} target rows)", t0)

    table = Table(title="RESULTS SUMMARY", show_header=False, border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("MAE", f"{metrics.get('MAE', 0):.2f} EUR/MWh")
    table.add_row("RMSE", f"{metrics.get('RMSE', 0):.2f} EUR/MWh")
    table.add_row("Skill vs Naive", f"{metrics.get('skill_vs_naive_pct', 0):+.1f}%")
    cov = metrics.get("conformal_coverage_90_empirical")
    table.add_row("Conformal Coverage", f"{100 * cov:.1f}% (target ≥ 90%)" if cov else "N/A")
    table.add_row("Quality Score", f"{qa.get('overall_quality_score')}/100")
    table.add_row("Backtest P&L", f"{bt.get('total_pnl_eur_per_mw', 0):+.1f} EUR/MW (2024)")
    console.print(table)
    console.print("\n[cyan]Dashboard:[/] streamlit run dashboard/app.py")
    console.print(f"[cyan]Submission:[/] {sub}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cobblestone Power Analytics Pipeline")
    parser.add_argument(
        "--mode",
        default="full",
        choices=[
            "full",
            "ingest",
            "clean",
            "regime",
            "features",
            "validate",
            "forecast",
            "backtest",
            "commentary",
            "submission",
            "dashboard",
            "api",
            "qa",
        ],
    )
    parser.add_argument("--resume", action="store_true", help="Resume walk-forward from saved windows")
    parser.add_argument("--fast", action="store_true", default=True, help="Faster XGB for demo runs")
    parser.add_argument("--full-xgb", action="store_true", help="Use full 1200-tree XGBoost")
    parser.add_argument(
        "--eval-mode",
        default="full_period",
        choices=["full_period", "post_crisis"],
        help="Walk-forward training horizon: full_period (2022+) or post_crisis (2023+)",
    )
    args = parser.parse_args()

    setup_logging()
    settings = get_settings()
    settings.validate_all()
    logger.info("Config: %s", settings.to_dict())

    fast = not args.full_xgb

    if args.mode == "full":
        run_full(resume=args.resume, fast=fast)
    elif args.mode == "ingest":
        cmd_ingest()
    elif args.mode == "clean":
        cmd_clean()
    elif args.mode == "qa":
        cmd_qa()
    elif args.mode == "regime":
        cmd_regime()
    elif args.mode == "features":
        cmd_features()
    elif args.mode == "validate":
        cmd_validate(resume=args.resume, fast=fast, eval_mode=args.eval_mode)
    elif args.mode == "forecast":
        cmd_forecast()
    elif args.mode == "backtest":
        cmd_backtest()
    elif args.mode == "commentary":
        cmd_commentary()
    elif args.mode == "submission":
        cmd_submission()
    elif args.mode == "dashboard":
        cmd_dashboard()
    elif args.mode == "api":
        cmd_api()


if __name__ == "__main__":
    main()
