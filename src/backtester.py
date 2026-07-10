"""
Cobblestone Power — simplified fundamental-divergence trading backtest.

Purpose:
    Translate forecast skill into illustrative P&L under a transparent rule:
    trade next-day peak when |model − actual DA| would have exceeded €8/MWh.
    NOTE: Uses realised DA as the 'market' reference for research illustration;
    live trading would compare to the prevailing forward/prompt quote.

Inputs:
    Forecast DataFrame with y_pred, y_true, price_regime; optional peak flags.

Outputs:
    Backtest stats dict; PnL figures with disclaimer.

Side Effects:
    Writes figures under outputs/figures/forecasts/; JSON stats.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.settings import get_settings
from src.utils import save_figure, write_json

logger = logging.getLogger(__name__)

BACKTEST_DISCLAIMER: str = (
    "Simplified research backtest. Does not account for bid-ask spreads, "
    "market impact, position limits, counterparty credit, or actual EPEX "
    "trading constraints. For illustrative purposes only."
)

DEFAULT_THRESHOLD_EUR: float = 8.0


class TradingBacktester:
    """
    Fundamental Divergence strategy backtester.

    Logic (research illustration):
        Each day, using the model's next-day peak forecast vs realised DA peak:
        - If forecast > actual by > threshold → conceptually the model was long
          relative to clearing (we score as if we went LONG when model > prior
          reference). Practical implementation: signal = sign(y_pred_peak −
          y_naive_peak) when |y_pred − y_true| setup is evaluated ex-post for
          skill attribution, and ex-ante using lag-168 as market proxy.

    We implement the ex-ante version: compare model peak forecast to seasonal
    naive peak; trade when |model − naive| > threshold; P&L = direction ×
    (actual_peak − naive_peak).
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD_EUR) -> None:
        self.threshold = threshold
        self.settings = get_settings()

    def run_backtest(
        self,
        forecast_df: pd.DataFrame,
        actual_df: Optional[pd.DataFrame] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Run daily peak divergence backtest for 2024.

        Args:
            forecast_df: Must contain y_pred, y_true (or actual_df), optionally
                y_naive and price_regime.
            actual_df: Optional override for actuals (column da_price or y_true).
            threshold: EUR/MWh divergence threshold (default 8).

        Returns:
            Stats dict including trades, win rate, PnL, Sharpe, drawdown.
        """
        thr = threshold if threshold is not None else self.threshold
        df = forecast_df.copy()
        if actual_df is not None:
            col = "da_price" if "da_price" in actual_df.columns else actual_df.columns[0]
            df["y_true"] = actual_df[col].reindex(df.index)

        # Daily peak aggregation (hours 8-19 weekdays; weekends use hours 8-19 mean)
        daily = []
        for day, g in df.groupby(df.index.date):
            peak_mask = (g.index.hour >= 8) & (g.index.hour < 20)
            if not peak_mask.any():
                continue
            row = {
                "date": pd.Timestamp(day, tz="UTC"),
                "pred_peak": float(g.loc[peak_mask, "y_pred"].mean()),
                "actual_peak": float(g.loc[peak_mask, "y_true"].mean()),
                "naive_peak": float(g.loc[peak_mask, "y_naive"].mean())
                if "y_naive" in g.columns
                else float(g.loc[peak_mask, "y_true"].mean()),
                "regime": int(g["price_regime"].mode().iloc[0]) if "price_regime" in g.columns else 2,
            }
            daily.append(row)
        daily_df = pd.DataFrame(daily).set_index("date")

        # Ex-ante signal vs naive; PnL vs actual
        daily_df["divergence"] = daily_df["pred_peak"] - daily_df["naive_peak"]
        daily_df["position"] = 0
        daily_df.loc[daily_df["divergence"] > thr, "position"] = 1  # LONG
        daily_df.loc[daily_df["divergence"] < -thr, "position"] = -1  # SHORT
        daily_df["pnl"] = daily_df["position"] * (daily_df["actual_peak"] - daily_df["naive_peak"])

        trades = daily_df[daily_df["position"] != 0]
        n_trades = int(len(trades))
        win_rate = float((trades["pnl"] > 0).mean()) if n_trades else 0.0
        total_pnl = float(daily_df["pnl"].sum())
        rets = daily_df["pnl"]
        sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
        cum = daily_df["pnl"].cumsum()
        drawdown = cum - cum.cummax()
        max_dd = float(drawdown.min()) if len(drawdown) else 0.0
        gains = trades.loc[trades["pnl"] > 0, "pnl"].sum()
        losses = trades.loc[trades["pnl"] < 0, "pnl"].abs().sum()
        profit_factor = float(gains / losses) if losses > 0 else float("inf") if gains > 0 else 0.0

        by_regime = {}
        for r in range(4):
            sub = trades[trades["regime"] == r]
            by_regime[str(r)] = {
                "n_trades": int(len(sub)),
                "total_pnl": float(sub["pnl"].sum()) if len(sub) else 0.0,
                "win_rate": float((sub["pnl"] > 0).mean()) if len(sub) else None,
            }

        stats = {
            "threshold_eur": thr,
            "n_trades": n_trades,
            "win_rate": win_rate,
            "total_pnl_eur_per_mw": total_pnl,
            "sharpe_ratio": sharpe,
            "max_drawdown_eur": max_dd,
            "profit_factor": profit_factor,
            "by_regime": by_regime,
            "disclaimer": BACKTEST_DISCLAIMER,
        }
        write_json(self.settings.forecasts_dir / "backtest_stats.json", stats)
        self._figures(daily_df, cum, stats)
        logger.info(
            "Backtest — trades=%s win=%.1f%% PnL=%+.1f EUR/MW Sharpe=%.2f | %s",
            n_trades,
            100 * win_rate,
            total_pnl,
            sharpe,
            BACKTEST_DISCLAIMER[:60] + "...",
        )
        return stats

    def _figures(self, daily_df: pd.DataFrame, cum: pd.Series, stats: Dict[str, Any]) -> None:
        """Write PnL curve, trade distribution, regime breakdown."""
        out = self.settings.figures / "forecasts"
        out.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(cum.index, cum.values, color="#00e676", lw=1.5)
        ax.axhline(0, color="#5f6368", lw=0.8)
        ax.set_title("Cumulative P&L — Fundamental Divergence (2024)")
        ax.set_ylabel("EUR per MW")
        ax.text(0.01, 0.02, BACKTEST_DISCLAIMER, transform=ax.transAxes, fontsize=6, color="#9aa0a6", wrap=True)
        # Annotate key events
        for d, label in [("2024-11-02", "Nov DF"), ("2024-12-12", "Dec DF")]:
            ts = pd.Timestamp(d, tz="UTC")
            if ts in cum.index or (cum.index.min() <= ts <= cum.index.max()):
                ax.axvline(ts, color="#ff9100", ls="--", alpha=0.5)
                ax.text(ts, cum.max() * 0.9 if cum.max() else 1, label, color="#ff9100", fontsize=8)
        save_figure(fig, out / "pnl_curve.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        trades = daily_df.loc[daily_df["position"] != 0, "pnl"]
        ax.hist(trades, bins=30, color="#00d4ff", edgecolor="none")
        ax.set_title("Trade P&L distribution")
        ax.set_xlabel("EUR/MW")
        ax.text(0.01, 0.02, BACKTEST_DISCLAIMER, transform=ax.transAxes, fontsize=6, color="#9aa0a6")
        save_figure(fig, out / "trade_distribution.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        regimes = list(stats["by_regime"].keys())
        pnls = [stats["by_regime"][r]["total_pnl"] for r in regimes]
        ax.bar(regimes, pnls, color=["#00e676", "#9aa0a6", "#00d4ff", "#ff9100"])
        ax.set_xlabel("Regime")
        ax.set_ylabel("Total P&L EUR/MW")
        ax.set_title("P&L by regime")
        ax.text(0.01, 0.02, BACKTEST_DISCLAIMER, transform=ax.transAxes, fontsize=6, color="#9aa0a6")
        save_figure(fig, out / "regime_pnl_breakdown.png")
        plt.close(fig)
