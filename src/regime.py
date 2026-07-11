"""
Cobblestone Power — price regime detection and Dunkelflaute early warning.

Purpose:
    Label German power hours into four regimes (negative/low/normal/high),
    detect Dunkelflaute events, score solar cannibalization risk, and
    estimate regime probabilities for use as model features.

Inputs:
    Cleaned master DataFrame with load, wind, solar, price.

Outputs:
    DataFrame with regime columns; figures under outputs/figures/regime/.

Side Effects:
    Writes PNG figures at 300 DPI; logs event verification for Nov/Dec 2024.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config.settings import FIGURE_DPI, RANDOM_SEED, get_settings
from src.utils import save_figure, validate_columns

logger = logging.getLogger(__name__)

# Regime thresholds (MW / ratios)
REGIME_NEG_RENEW_PEN: float = 0.80
REGIME_LOW_RESIDUAL_MW: float = 30_000.0
REGIME_HIGH_RESIDUAL_MW: float = 55_000.0
REGIME_HIGH_WIND_MAX_MW: float = 5_000.0
DUNKEL_RENEW_RATIO: float = 0.20
DUNKEL_MIN_HOURS: int = 12
DUNKEL_PRICE_EUR: float = 200.0
DUNKEL_PRICE_MIN_HOURS: int = 6
SOLAR_CANNIBAL_SHARE: float = 0.15
SOLAR_CANNIBAL_MW: float = 20_000.0
SUMMER_MONTHS: Tuple[int, ...] = (4, 5, 6, 7, 8, 9)

REGIME_NAMES: Dict[int, str] = {
    0: "NEGATIVE/ZERO",
    1: "LOW",
    2: "NORMAL",
    3: "HIGH/DUNKELFLAUTE",
}

# Known Dunkelflaute events for verification
KNOWN_DUNKELFLAUTE: List[Tuple[date, date, str]] = [
    (date(2024, 11, 2), date(2024, 11, 7), "Nov 2024 Dunkelflaute"),
    (date(2024, 12, 12), date(2024, 12, 14), "Dec 2024 Dunkelflaute"),
]


class RegimeDetector:
    """
    Detect German power price regimes and Dunkelflaute stress events.

    Purpose:
        Translate fundamentals into discrete regimes and soft probabilities
        that drive conformal intervals and trading signals.

    Inputs:
        Master DataFrame with da_load, da_wind, da_solar, da_price.

    Outputs:
        Augmented DataFrame; regime figures.

    Side Effects:
        Fits logistic regression; writes figures to outputs/figures/regime/.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._logit: Optional[LogisticRegression] = None
        self._scaler: Optional[StandardScaler] = None

    def detect_dunkelflaute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flag Dunkelflaute via renewable drought or extreme price runs.

        Triggers (OR):
          1) (wind+solar)/load < 20% for ≥12 consecutive hours
          2) da_price > 200 EUR/MWh for ≥6 consecutive hours

        Adds dunkelflaute_day_index (days into event) and dunkelflaute_severity
        (0–3). Verifies detection against Nov 2–7 and Dec 12–14 2024.

        Args:
            df: Panel with da_load, da_wind, da_solar (and da_price if available).

        Returns:
            Copy with Dunkelflaute columns.

        Raises:
            ValueError: If required columns missing.
        """
        validate_columns(df, ["da_load", "da_wind", "da_solar"], "detect_dunkelflaute")
        out = df.copy()
        ratio = (out["da_wind"] + out["da_solar"]) / out["da_load"].replace(0, np.nan)
        renew_stress = (ratio < DUNKEL_RENEW_RATIO).fillna(False)

        # Run-length encoding for consecutive renewable-stress hours
        renew_gid = (renew_stress != renew_stress.shift(fill_value=False)).cumsum()
        renew_run = renew_stress.groupby(renew_gid).transform("sum")
        renew_event = renew_stress & (renew_run >= DUNKEL_MIN_HOURS)

        # Alternative: extreme price run (da_price > 200 for ≥6h)
        price_event = pd.Series(False, index=out.index)
        if "da_price" in out.columns:
            price_stress = (out["da_price"] > DUNKEL_PRICE_EUR).fillna(False)
            price_gid = (price_stress != price_stress.shift(fill_value=False)).cumsum()
            price_run = price_stress.groupby(price_gid).transform("sum")
            price_event = price_stress & (price_run >= DUNKEL_PRICE_MIN_HOURS)

        in_event = renew_event | price_event

        out["dunkelflaute_active"] = in_event
        # Day index within event
        event_id = (in_event != in_event.shift(fill_value=False)).cumsum()
        event_id = event_id.where(in_event, 0)
        hours_into = in_event.groupby(event_id).cumsum()
        out["dunkelflaute_day_index"] = np.where(in_event, np.ceil(hours_into / 24.0), 0).astype(int)

        severity = np.zeros(len(out), dtype=int)
        severity = np.where(in_event & (ratio < 0.05), 3, severity)
        severity = np.where(in_event & (ratio >= 0.05) & (ratio < 0.10), 2, severity)
        severity = np.where(in_event & (ratio >= 0.10) & (ratio < 0.20), 1, severity)
        # Price-triggered hours with milder renewable drought still get severity ≥1
        severity = np.where(in_event & (severity == 0), 1, severity)
        out["dunkelflaute_severity"] = severity

        self._verify_known_events(out)
        return out

    def _verify_known_events(self, df: pd.DataFrame) -> None:
        """Log whether known 2024 Dunkelflaute windows were detected."""
        for start, end, label in KNOWN_DUNKELFLAUTE:
            mask = (df.index.date >= start) & (df.index.date <= end)
            if not mask.any():
                logger.info("Known event %s outside data range — skip verify", label)
                continue
            hours = int(df.loc[mask, "dunkelflaute_active"].sum())
            detected = hours > 0
            if detected:
                logger.info(
                    "✓ Dunkelflaute detection caught %s — %s hours flagged in %s to %s",
                    label,
                    hours,
                    start.isoformat(),
                    end.isoformat(),
                )
            else:
                logger.warning("✗ Dunkelflaute detection MISSED %s — check thresholds", label)

    def detect_solar_cannibalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score solar cannibalization risk and capture-rate proxy.

        Condition: da_solar > 15% of load AND da_solar > 20 GW.

        Args:
            df: Panel with da_load, da_solar, da_price.

        Returns:
            Copy with solar_cannibal_risk and capture_rate_proxy.
        """
        validate_columns(df, ["da_load", "da_solar"], "detect_solar_cannibalization")
        out = df.copy()
        share = out["da_solar"] / out["da_load"].replace(0, np.nan)
        trigger = (share > SOLAR_CANNIBAL_SHARE) & (out["da_solar"] > SOLAR_CANNIBAL_MW)
        # Risk score 0-1 from solar share above threshold
        risk = ((share - SOLAR_CANNIBAL_SHARE) / 0.35).clip(0, 1).fillna(0.0)
        out["solar_cannibal_risk"] = np.where(trigger, risk, risk * 0.3)
        if "da_price" in out.columns:
            # Capture rate proxy: solar-weighted price / baseload
            w = out["da_solar"].clip(lower=0)
            solar_w_price = (out["da_price"] * w).rolling(24 * 7, min_periods=24).sum()
            solar_w = w.rolling(24 * 7, min_periods=24).sum().replace(0, np.nan)
            baseload = out["da_price"].rolling(24 * 7, min_periods=24).mean()
            out["capture_rate_proxy"] = (solar_w_price / solar_w / baseload.replace(0, np.nan)).clip(0, 2).fillna(1.0)
        else:
            out["capture_rate_proxy"] = 1.0
        return out

    def label_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign hard regime labels 0–3 and soft logistic probabilities.

        REGIME_0: renewable_pen > 0.80 AND weekend AND summer month
        REGIME_1: residual_load < 30 GW
        REGIME_2: residual 30–55 GW
        REGIME_3: residual > 55 GW AND (wind < 5 GW OR cold_dark_week)

        Args:
            df: Panel preferably after dunkelflaute/solar detection.

        Returns:
            DataFrame with price_regime and regime_probability_0..3.
        """
        validate_columns(df, ["da_load", "da_wind", "da_solar"], "label_regimes")
        out = df.copy()
        if "dunkelflaute_severity" not in out.columns:
            out = self.detect_dunkelflaute(out)
        if "solar_cannibal_risk" not in out.columns:
            out = self.detect_solar_cannibalization(out)

        residual = out["da_load"] - out["da_wind"] - out["da_solar"]
        renew_pen = (out["da_wind"] + out["da_solar"]) / out["da_load"].replace(0, np.nan)
        is_weekend = out.index.dayofweek >= 5
        is_summer = out.index.month.isin(SUMMER_MONTHS)
        cold_dark = out["dunkelflaute_active"] if "dunkelflaute_active" in out.columns else False

        regime = np.full(len(out), 2, dtype=int)
        regime = np.where(residual < REGIME_LOW_RESIDUAL_MW, 1, regime)
        regime = np.where(
            (residual > REGIME_HIGH_RESIDUAL_MW)
            & ((out["da_wind"] < REGIME_HIGH_WIND_MAX_MW) | cold_dark),
            3,
            regime,
        )
        # Negative/zero glut overrides when conditions met
        regime = np.where(
            (renew_pen > REGIME_NEG_RENEW_PEN) & is_weekend & is_summer,
            0,
            regime,
        )
        # Also label actual negative prices as regime 0 when glut-like
        if "da_price" in out.columns:
            regime = np.where((out["da_price"] <= 0) & (renew_pen > 0.5), 0, regime)

        out["residual_load"] = residual
        out["renewable_penetration"] = renew_pen
        out["price_regime"] = regime

        out = self._fit_regime_probabilities(out)
        return out

    def _fit_regime_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit multinomial logistic regression for soft regime probabilities."""
        out = df.copy()
        feature_cols = ["residual_load", "renewable_penetration", "da_wind", "da_solar", "da_load"]
        X = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y = out["price_regime"].astype(int)

        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X)
        self._logit = LogisticRegression(
            multi_class="multinomial",
            max_iter=500,
            random_state=RANDOM_SEED,
            C=1.0,
        )
        try:
            self._logit.fit(Xs, y)
            proba = self._logit.predict_proba(Xs)
            classes = list(self._logit.classes_)
            for r in range(4):
                if r in classes:
                    out[f"regime_probability_{r}"] = proba[:, classes.index(r)]
                else:
                    out[f"regime_probability_{r}"] = 0.0
        except Exception as exc:
            logger.warning("Regime logistic fit failed (%s) — using one-hot probs", exc)
            for r in range(4):
                out[f"regime_probability_{r}"] = (y == r).astype(float)

        return out

    def run(self, df: pd.DataFrame, produce_figures: bool = True) -> pd.DataFrame:
        """
        Full regime pipeline: Dunkelflaute → solar cannibal → labels → figures.

        Args:
            df: Cleaned master DataFrame.
            produce_figures: If True, write regime figures.

        Returns:
            Augmented DataFrame.
        """
        out = self.detect_dunkelflaute(df)
        out = self.detect_solar_cannibalization(out)
        out = self.label_regimes(out)

        n_neg = int((out.get("da_price", pd.Series(dtype=float)) < 0).sum()) if "da_price" in out.columns else 0
        if out["dunkelflaute_active"].any():
            active_dates = out.index[out["dunkelflaute_active"]].date
            n_dunkel_days = len(set(active_dates))
        else:
            n_dunkel_days = 0
        logger.info(
            "Regime detection — neg/zero hours=%s | dunkelflaute days≈%s | regime counts=%s",
            n_neg,
            n_dunkel_days,
            out["price_regime"].value_counts().sort_index().to_dict(),
        )

        if produce_figures and "da_price" in out.columns:
            self.produce_figures(out)
        return out

    def produce_figures(self, df: pd.DataFrame) -> None:
        """Write the four mandatory regime analysis figures."""
        out_dir = self.settings.figures / "regime"
        out_dir.mkdir(parents=True, exist_ok=True)
        colors = {0: "#00e676", 1: "#9aa0a6", 2: "#00d4ff", 3: "#ff9100"}

        # 1. Regime timeline
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(df.index, df["da_price"], color="white", lw=0.3, alpha=0.9)
        for r, c in colors.items():
            mask = df["price_regime"] == r
            if mask.any():
                ax.fill_between(df.index, df["da_price"].min(), df["da_price"].max(), where=mask, color=c, alpha=0.15, linewidth=0)
        ax.set_facecolor("#0a0d14")
        fig.patch.set_facecolor("#0a0d14")
        ax.set_ylabel("EUR/MWh", color="#e8eaed")
        ax.set_title("Regime timeline — DE DA prices", color="#e8eaed")
        ax.tick_params(colors="#9aa0a6")
        save_figure(fig, out_dir / "regime_timeline.png")
        plt.close(fig)

        # 2. Violin by regime
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_df = df[["da_price", "price_regime"]].dropna().copy()
        plot_df["price_regime"] = plot_df["price_regime"].astype(int)
        sns.violinplot(
            data=plot_df,
            x="price_regime",
            y="da_price",
            hue="price_regime",
            ax=ax,
            palette=[colors[i] for i in sorted(plot_df["price_regime"].unique())],
            legend=False,
        )
        ax.set_title("Price distribution by regime")
        ax.set_xlabel("Regime")
        ax.set_ylabel("EUR/MWh")
        save_figure(fig, out_dir / "regime_price_distribution.png")
        plt.close(fig)

        # 3. Dunkelflaute zoom Nov + Dec 2024
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, (start, end, label) in zip(axes, KNOWN_DUNKELFLAUTE):
            m = (df.index.date >= start) & (df.index.date <= end)
            if m.any():
                ax.plot(df.index[m], df.loc[m, "da_price"], color="#ff9100", lw=1.2)
                ax.set_title(label)
                ax.set_ylabel("EUR/MWh")
            else:
                ax.set_title(f"{label} (no data)")
        save_figure(fig, out_dir / "dunkelflaute_events.png")
        plt.close(fig)

        # 4. Solar cannibalization scatter
        fig, ax = plt.subplots(figsize=(8, 5))
        if "renewable_penetration" in df.columns:
            sc = ax.scatter(
                df["da_solar"] / df["da_load"].replace(0, np.nan),
                df["da_price"],
                c=df.index.hour,
                cmap="viridis",
                s=3,
                alpha=0.4,
            )
            fig.colorbar(sc, ax=ax, label="Hour UTC")
        ax.set_xlabel("Solar penetration")
        ax.set_ylabel("DA price EUR/MWh")
        ax.set_title("Solar cannibalization: penetration vs price")
        save_figure(fig, out_dir / "solar_cannibalization_scatter.png")
        plt.close(fig)
