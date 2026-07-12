"""Backtest panel with disclaimer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import streamlit as st

from utils.dashboard_helpers import metric_card_html, render_placeholder, safe_render

DISCLAIMER = (
    "Simplified research backtest. Does not account for bid-ask spreads, market impact, "
    "position limits, counterparty credit, or actual EPEX trading constraints. "
    "For illustrative purposes only."
)

REGIME_LABELS = {
    "0": "Negative/Glut",
    "1": "Low",
    "2": "Normal",
    "3": "Dunkelflaute",
    0: "Negative/Glut",
    1: "Low",
    2: "Normal",
    3: "Dunkelflaute",
}


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
            '<div class="metric-card" style="border-color:#f59e0b;">'
            '<div class="metric-label">SHARPE</div>'
            f'<div class="metric-value" style="color:#f59e0b;">{float(stats.get("sharpe_ratio", 0)):.2f}</div>'
            '<div class="metric-subtext" style="color:#f59e0b;text-transform:none;letter-spacing:0;">'
            "WARNING: No transaction costs assumed — not comparable to live trading."
            "</div></div>",
            unsafe_allow_html=True,
        )

    st.subheader("P&L by Regime")
    by_reg = stats.get("by_regime", {}) or {}
    rows = []
    for rid in ["0", "1", "2", "3"]:
        block = by_reg.get(rid) or by_reg.get(int(rid)) or {}
        if not block:
            continue
        pnl = float(block.get("total_pnl", 0) or 0)
        wr = float(block.get("win_rate", 0) or 0)
        n = int(block.get("n_trades", 0) or 0)
        pnl_cls = "cell-low" if pnl >= 0 else "cell-high"
        rows.append(
            f"<tr>"
            f"<td>{rid}</td>"
            f"<td>{REGIME_LABELS.get(rid, rid)}</td>"
            f"<td style='font-family:JetBrains Mono,monospace'>{n}</td>"
            f"<td class='{pnl_cls}' style='font-family:JetBrains Mono,monospace'>{pnl:+.1f}</td>"
            f"<td style='font-family:JetBrains Mono,monospace'>{100 * wr:.1f}%</td>"
            f"</tr>"
        )
    if rows:
        st.markdown(
            '<table class="qa-table"><thead><tr>'
            "<th>Regime</th><th>Name</th><th>N Trades</th>"
            "<th>Total P&L EUR/MW</th><th>Win Rate</th>"
            "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>",
            unsafe_allow_html=True,
        )
    else:
        render_placeholder("No regime P&L breakdown available")

    for name in ["pnl_curve.png", "trade_distribution.png", "regime_pnl_breakdown.png"]:
        path = figures_dir / "forecasts" / name
        try:
            if path.exists():
                st.image(str(path), caption=name)
        except Exception:
            pass

    st.markdown(f'<p class="disclaimer">{DISCLAIMER}</p>', unsafe_allow_html=True)
