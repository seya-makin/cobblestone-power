"""Backtest panel with disclaimer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import streamlit as st

from dashboard.utils.dashboard_helpers import metric_card_html, render_placeholder, safe_render

DISCLAIMER = (
    "Simplified research backtest. Does not account for bid-ask spreads, market impact, "
    "position limits, counterparty credit, or actual EPEX trading constraints. "
    "For illustrative purposes only."
)


@safe_render("Backtest panel unavailable — run pipeline --mode backtest")
def render_backtest_panel(stats: Dict[str, Any], figures_dir: Path) -> None:
    """Render backtest KPIs and PnL figures."""
    st.info(DISCLAIMER)
    if not stats:
        render_placeholder("Run pipeline to generate this data")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            metric_card_html("Trades", str(stats.get("n_trades", 0))),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            metric_card_html("Win Rate", f"{100 * float(stats.get('win_rate', 0)):.1f}%"),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            metric_card_html(
                "Total P&L",
                f"{stats.get('total_pnl_eur_per_mw', 0):+.1f}",
                subtext="EUR/MW",
            ),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            metric_card_html("Sharpe", f"{stats.get('sharpe_ratio', 0):.2f}"),
            unsafe_allow_html=True,
        )

    st.subheader("P&L by Regime")
    st.json(stats.get("by_regime", {}))

    for name in ["pnl_curve.png", "trade_distribution.png", "regime_pnl_breakdown.png"]:
        path = figures_dir / "forecasts" / name
        if path.exists():
            try:
                st.image(str(path), caption=name)
            except Exception:
                pass

    st.markdown(f'<p class="disclaimer">{DISCLAIMER}</p>', unsafe_allow_html=True)
