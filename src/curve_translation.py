"""
Cobblestone Power — curve translation and trading signal construction.

Purpose:
    Aggregate hourly forecasts into delivery-period views (baseload/peak/offpeak)
    and emit structured trading signals with invalidation conditions.

Inputs:
    Hourly forecast DataFrame; conformal intervals; regime forecast.

Outputs:
    Delivery view dict; trading signal dict; JSON artefacts.

Side Effects:
    Writes latest_forecast.json / regime_forecast.json under outputs/forecasts/.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import get_settings
from src.model import NEGATIVE_PRICE_RISK_THRESHOLD
from src.utils import utc_now_iso, write_json

logger = logging.getLogger(__name__)

INVALIDATION_CONDITIONS: List[str] = [
    "Wind forecast revised > 20% from signal generation time",
    "TTF gas price moves > 3 EUR/MWh overnight",
    "French nuclear availability drops > 2000 MW unexpectedly",
    "Actual DA clearing > 2 sigma from model forecast",
    "Residual load changes > 15% from signal generation time",
    "New unplanned plant outage > 1000 MW announced after signal",
    "Model regime classification changes between signal generation and delivery",
    "Conformal interval width expands > 50% vs 7-day mean",
]

SIGNAL_THRESHOLD_EUR: float = 8.0


class CurveTranslator:
    """
    Translate hourly fair-value forecasts into tradable delivery views.

    Purpose:
        Bridge model output to how power traders actually quote the curve.

    Inputs:
        Hourly forecasts + conformal intervals + regime probabilities.

    Outputs:
        Delivery-period table; trading signal JSON.

    Side Effects:
        Persists forecast/signal JSON files.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    def generate_delivery_period_view(
        self,
        hourly_forecast_df: pd.DataFrame,
        conformal_intervals_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Build baseload/peak/offpeak views for tomorrow, next week, next month.

        Args:
            hourly_forecast_df: Must include y_pred (and optionally actuals).
            conformal_intervals_df: Optional lo/hi columns aligned by index.

        Returns:
            Nested delivery view dict with conformal bands.
        """
        df = hourly_forecast_df.copy()
        if conformal_intervals_df is not None:
            df = df.join(conformal_intervals_df, how="left", rsuffix="_ci")

        pred_col = "y_pred" if "y_pred" in df.columns else df.columns[0]

        def _period_stats(sub: pd.DataFrame) -> Dict[str, float]:
            if sub.empty:
                return {
                    "baseload": None,
                    "peak": None,
                    "offpeak": None,
                    "peak_base_spread": None,
                    "conformal_80_low": None,
                    "conformal_80_high": None,
                    "conformal_95_low": None,
                    "conformal_95_high": None,
                }
            baseload = float(sub[pred_col].mean())
            peak_mask = (sub.index.hour >= 8) & (sub.index.hour < 20) & (sub.index.dayofweek < 5)
            off_mask = ~peak_mask
            peak = float(sub.loc[peak_mask, pred_col].mean()) if peak_mask.any() else baseload
            offpeak = float(sub.loc[off_mask, pred_col].mean()) if off_mask.any() else baseload

            def _band(lo_col: str, hi_col: str) -> tuple:
                if lo_col in sub.columns and hi_col in sub.columns:
                    return float(sub[lo_col].mean()), float(sub[hi_col].mean())
                # Approximate from point ± residual proxy
                return baseload - 25.0, baseload + 25.0

            c80 = _band("conformal_80_low", "conformal_80_high")
            # 95% ≈ wider; use 90% scaled if no dedicated cols
            if "conformal_95_low" in sub.columns:
                c95 = _band("conformal_95_low", "conformal_95_high")
            else:
                c90 = _band("conformal_90_low", "conformal_90_high")
                mid = baseload
                half = max(c90[1] - mid, mid - c90[0]) * 1.2
                c95 = (mid - half, mid + half)

            return {
                "baseload": baseload,
                "peak": peak,
                "offpeak": offpeak,
                "peak_base_spread": peak - baseload,
                "conformal_80_low": c80[0],
                "conformal_80_high": c80[1],
                "conformal_95_low": c95[0],
                "conformal_95_high": c95[1],
            }

        start = df.index.min()
        tomorrow = df.loc[start : start + pd.Timedelta(hours=23)]
        next_week = df.loc[start : start + pd.Timedelta(days=7) - pd.Timedelta(hours=1)]
        next_month = df.loc[start : start + pd.Timedelta(days=30) - pd.Timedelta(hours=1)]

        view = {
            "generated_at": utc_now_iso(),
            "tomorrow": _period_stats(tomorrow),
            "next_week": _period_stats(next_week),
            "next_month": _period_stats(next_month),
        }

        # Forward-curve decay when walk-forward only supplies one prompt day
        # (next_week / next_month collapse to the same hours as tomorrow).
        unique_days = pd.Index(df.index.normalize().unique())
        if len(unique_days) <= 1:
            t = view["tomorrow"]

            def _scale_block(block: Dict[str, Any], factor: float) -> Dict[str, Any]:
                out = dict(block)
                for key in (
                    "baseload",
                    "peak",
                    "offpeak",
                    "peak_base_spread",
                    "conformal_80_low",
                    "conformal_80_high",
                    "conformal_95_low",
                    "conformal_95_high",
                ):
                    val = out.get(key)
                    if val is not None:
                        try:
                            out[key] = float(val) * factor
                        except (TypeError, ValueError):
                            pass
                return out

            view["next_week"] = _scale_block(t, 0.98)
            view["next_month"] = _scale_block(t, 0.96)
            view["forward_curve_decay_applied"] = True
            view["forward_curve_decay_note"] = (
                "Single forecast day available — applied decay "
                "Next Week=Tomorrow×0.98, Next Month=Tomorrow×0.96"
            )
        else:
            view["forward_curve_decay_applied"] = False

        # Convenience aliases from prompt
        t = view["tomorrow"]
        view["baseload_tomorrow"] = t["baseload"]
        view["peak_tomorrow"] = t["peak"]
        view["offpeak_tomorrow"] = t["offpeak"]
        view["peak_base_spread_tomorrow"] = t["peak_base_spread"]
        view["next_week_baseload"] = view["next_week"]["baseload"]
        view["next_week_peak"] = view["next_week"]["peak"]
        view["next_month_baseload"] = view["next_month"]["baseload"]

        write_json(self.settings.forecasts_dir / "latest_forecast.json", view)
        return view

    def generate_trading_signal(
        self,
        delivery_view: Dict[str, Any],
        regime_forecast: Dict[str, Any],
        market_reference_baseload: Optional[float] = None,
        hourly_forecast: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Construct a structured trading signal from delivery view + regime.

        Args:
            delivery_view: Output of generate_delivery_period_view.
            regime_forecast: Dominant regime + probabilities + risk scores.
            market_reference_baseload: Optional market quote to diverge from;
                if None, uses 7d mean proxy from regime_forecast.
            hourly_forecast: Optional hourly frame with
                ``negative_price_probability`` for HIGH-risk hour flags.

        Returns:
            Signal dict matching the Part 11 schema.
        """
        expected = float(delivery_view.get("baseload_tomorrow") or 0.0)
        peak = float(delivery_view.get("peak_tomorrow") or expected)
        ref = market_reference_baseload
        if ref is None:
            ref = float(regime_forecast.get("reference_baseload", expected))

        delta = expected - ref
        if delta > SIGNAL_THRESHOLD_EUR:
            direction = "LONG"
        elif delta < -SIGNAL_THRESHOLD_EUR:
            direction = "SHORT"
        else:
            direction = "NEUTRAL"

        strength = min(1.0, abs(delta) / 30.0)
        if strength > 0.66:
            conviction = "HIGH"
        elif strength > 0.33:
            conviction = "MEDIUM"
        else:
            conviction = "LOW"

        dominant = int(regime_forecast.get("dominant_regime", 2))
        probs = regime_forecast.get("regime_probabilities", {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0})
        # Normalise keys to int
        probs = {int(k): float(v) for k, v in probs.items()}

        width80 = float(delivery_view["tomorrow"]["conformal_80_high"] or 0) - float(
            delivery_view["tomorrow"]["conformal_80_low"] or 0
        )
        snr = abs(delta) / max(width80 / 2.0, 1.0)

        # —— Negative-price HIGH risk flags (classifier P > 0.3) ——
        neg_hour_flags: List[Dict[str, Any]] = []
        high_neg_hours: List[int] = []
        max_neg_prob = float(regime_forecast.get("negative_price_risk", probs.get(0, 0.0)))
        if hourly_forecast is not None and "negative_price_probability" in hourly_forecast.columns:
            probs_h = hourly_forecast["negative_price_probability"].astype(float)
            max_neg_prob = float(probs_h.max()) if len(probs_h) else max_neg_prob
            for ts, p in probs_h.items():
                hour = int(pd.Timestamp(ts).hour)
                entry = {
                    "timestamp": pd.Timestamp(ts).isoformat(),
                    "hour": hour,
                    "negative_price_probability": float(p),
                    "risk_level": "HIGH" if float(p) > NEGATIVE_PRICE_RISK_THRESHOLD else "NORMAL",
                }
                neg_hour_flags.append(entry)
                if float(p) > NEGATIVE_PRICE_RISK_THRESHOLD:
                    high_neg_hours.append(hour)
        neg_risk_level = "HIGH" if high_neg_hours or max_neg_prob > NEGATIVE_PRICE_RISK_THRESHOLD else "NORMAL"

        if dominant == 3:
            instrument = "DE prompt peak / peaker spread"
            rationale = (
                f"Model sees Dunkelflaute-leaning residual tightness; fair-value baseload "
                f"{expected:.1f} vs ref {ref:.1f} (Δ {delta:+.1f}). Prefer long prompt peak."
            )
        elif dominant == 0 or neg_risk_level == "HIGH":
            instrument = "DE prompt baseload (short) / storage spread"
            rationale = (
                f"Renewable glut / negative-price risk {neg_risk_level} "
                f"(max P(neg)={max_neg_prob:.2f}); fair-value {expected:.1f} vs ref {ref:.1f}. "
                f"Short prompt / long storage spread."
            )
            if high_neg_hours:
                rationale += f" HIGH neg-price hours (UTC): {sorted(set(high_neg_hours))}."
        else:
            instrument = "DE prompt baseload"
            rationale = (
                f"Fair-value baseload {expected:.1f} EUR/MWh vs reference {ref:.1f} "
                f"(Δ {delta:+.1f}). Conviction {conviction}."
            )

        signal = {
            "signal_date": str(datetime.now(timezone.utc).date()),
            "signal_generated_at": utc_now_iso(),
            "horizon": "prompt_day",
            "direction": direction,
            "conviction": conviction,
            "expected_da_baseload": expected,
            "expected_da_peak": peak,
            "conformal_80_low": delivery_view["tomorrow"]["conformal_80_low"],
            "conformal_80_high": delivery_view["tomorrow"]["conformal_80_high"],
            "conformal_95_low": delivery_view["tomorrow"]["conformal_95_low"],
            "conformal_95_high": delivery_view["tomorrow"]["conformal_95_high"],
            "peak_base_spread": delivery_view.get("peak_base_spread_tomorrow"),
            "dominant_regime": dominant,
            "regime_probabilities": probs,
            "dunkelflaute_risk": float(regime_forecast.get("dunkelflaute_risk", probs.get(3, 0.0))),
            "negative_price_risk": max_neg_prob,
            "negative_price_risk_level": neg_risk_level,
            "negative_price_risk_threshold": NEGATIVE_PRICE_RISK_THRESHOLD,
            "high_negative_price_risk_hours": sorted(set(high_neg_hours)),
            "negative_price_hour_flags": neg_hour_flags,
            "signal_strength": float(strength),
            "suggested_instrument": instrument,
            "trading_rationale": rationale,
            "invalidation_conditions": INVALIDATION_CONDITIONS,
            "risk_metrics": {
                "var_1d_95": float(abs(delta) + width80 * 0.6),
                "expected_shortfall": float(abs(delta) + width80 * 0.8),
                "signal_to_noise": float(snr),
            },
        }
        write_json(self.settings.forecasts_dir / "trading_signal.json", signal)
        write_json(self.settings.forecasts_dir / "regime_forecast.json", regime_forecast)
        logger.info(
            "Trading signal %s conviction=%s strength=%.2f | neg_risk=%s (maxP=%.2f, HIGH hours=%s)",
            direction,
            conviction,
            strength,
            neg_risk_level,
            max_neg_prob,
            sorted(set(high_neg_hours)),
        )
        return signal
