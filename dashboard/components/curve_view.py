"""Curve view + trading signal card with custom HTML tables."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.utils.dashboard_helpers import (
    render_placeholder,
    safe_plotly,
    safe_render,
    tab_section_header,
)


def _fmt(v: Any, digits: int = 1) -> str:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


@safe_render("Curve view unavailable — run pipeline --mode forecast")
def render_curve_view(delivery_view: Dict[str, Any], signal: Dict[str, Any], wf: pd.DataFrame) -> None:
    """Render delivery-period HTML table, signal card, history, invalidation."""
    tab_section_header("TRADING SIGNAL — Prompt curve view and actionable position recommendation")
    if not delivery_view:
        render_placeholder("Run pipeline to generate this data")
        return

    st.divider()
    st.subheader("Delivery Period View")
    if delivery_view.get("forward_curve_decay_applied"):
        st.caption(delivery_view.get("forward_curve_decay_note", "Forward curve decay applied"))

    rows_html = []
    for horizon, key in [("Tomorrow", "tomorrow"), ("Next Week", "next_week"), ("Next Month", "next_month")]:
        block = delivery_view.get(key, {}) or {}
        base = block.get("baseload")
        peak = block.get("peak")
        ref = (delivery_view.get("tomorrow") or {}).get("baseload")
        cls_b = ""
        cls_p = ""
        try:
            if ref is not None and base is not None:
                cls_b = "cell-high" if float(base) > float(ref) * 1.02 else ("cell-low" if float(base) < float(ref) * 0.98 else "")
            if ref is not None and peak is not None:
                cls_p = "cell-high" if float(peak) > float(ref) * 1.05 else ("cell-low" if float(peak) < float(ref) * 0.95 else "")
        except (TypeError, ValueError):
            pass
        rows_html.append(
            f"<tr>"
            f"<td>{horizon}</td>"
            f'<td class="{cls_b}">{_fmt(base)}</td>'
            f'<td class="{cls_p}">{_fmt(peak)}</td>'
            f"<td>{_fmt(block.get('peak_base_spread'))}</td>"
            f"<td>{_fmt(block.get('conformal_80_low'))}</td>"
            f"<td>{_fmt(block.get('conformal_80_high'))}</td>"
            f"</tr>"
        )

    st.markdown(
        '<table class="curve-table">'
        "<thead><tr>"
        "<th>Horizon</th><th>Baseload</th><th>Peak</th><th>Peak-Base</th><th>80% Low</th><th>80% High</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("Trading Signal")
    direction = (signal or {}).get("direction", "NEUTRAL")
    border_cls = {
        "LONG": "signal-border-long",
        "SHORT": "signal-border-short",
    }.get(direction, "signal-border-neutral")
    css = {"LONG": "signal-long", "SHORT": "signal-short"}.get(direction, "signal-neutral")
    strength = float((signal or {}).get("signal_strength", 0) or 0)
    strength_pct = max(0.0, min(100.0, 100.0 * strength))
    fill_cls = {"LONG": "long", "SHORT": "short"}.get(direction, "neutral")

    st.markdown(
        f'<div class="signal-card {border_cls}">'
        f'<div class="{css} signal-hero">{direction}</div>'
        f'<div style="color:#6b7280;font-size:13px;margin-bottom:4px;">'
        f"Conviction: <b style='color:#f9fafb'>{(signal or {}).get('conviction', '—')}</b>"
        f" &nbsp;|&nbsp; Strength: <b style='color:#f9fafb'>{strength_pct:.0f}%</b>"
        f" &nbsp;|&nbsp; Instrument: <b style='color:#f9fafb'>"
        f"{(signal or {}).get('suggested_instrument', '—')}</b></div>"
        f'<div class="signal-strength-track">'
        f'<div class="signal-strength-fill {fill_cls}" style="width:{strength_pct:.0f}%;"></div>'
        f"</div>"
        f'<div style="font-size:15px;line-height:1.55;color:#f9fafb;font-weight:300;">'
        f"{(signal or {}).get('trading_rationale', '')}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("**Invalidation conditions**")
    conditions: List[str] = (signal or {}).get("invalidation_conditions") or []
    dunk = float((signal or {}).get("dunkelflaute_risk", 0) or 0)
    neg = float((signal or {}).get("negative_price_risk", 0) or 0)
    items = []
    for cond in conditions:
        triggered = False
        low = cond.lower()
        if dunk > 0.5 and "regime" in low:
            triggered = True
        if neg > 0.5 and "residual" in low:
            triggered = True
        if dunk > 0.7 and "conformal" in low:
            triggered = True
        cls = "inv-hit" if triggered else "inv-ok"
        items.append(f'<li class="{cls}">{cond}</li>')
    if items:
        st.markdown(f'<ul class="inv-list">{"".join(items)}</ul>', unsafe_allow_html=True)
    else:
        render_placeholder("No invalidation conditions in signal JSON")

    st.divider()
    risk = (signal or {}).get("risk_metrics", {}) or {}
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">VaR 95%</div>'
            f'<div class="metric-value">{_fmt(risk.get("var_1d_95"))}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Expected Shortfall</div>'
            f'<div class="metric-value">{_fmt(risk.get("expected_shortfall"))}</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Signal/Noise</div>'
            f'<div class="metric-value">{_fmt(risk.get("signal_to_noise"), 2)}</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Signal History (7d)")
    _render_signal_history(wf, signal)

    st.divider()
    if wf is not None and not wf.empty and "y_true" in wf.columns and "y_pred" in wf.columns:
        st.subheader("Forecast vs Actual (by regime)")
        sample = wf.dropna(subset=["y_true", "y_pred"])
        if len(sample) > 2000:
            sample = sample.sample(2000, random_state=42)
        fig = px.scatter(
            sample,
            x="y_true",
            y="y_pred",
            color="price_regime" if "price_regime" in sample.columns else None,
            color_continuous_scale=[[0, "#10b981"], [0.33, "#6b7280"], [0.66, "#3b82f6"], [1, "#f59e0b"]],
            opacity=0.5,
        )
        lim = float(max(sample["y_true"].max(), sample["y_pred"].max(), 100))
        fig.add_shape(
            type="line",
            x0=-100,
            y0=-100,
            x1=lim,
            y1=lim,
            line=dict(color="#374151", dash="dash"),
        )
        fig.update_layout(
            title=dict(text="Forecast vs Actual (by regime)", x=0.0, xanchor="left"),
            height=400,
            xaxis_title="Actual EUR/MWh",
            yaxis_title="Forecast EUR/MWh",
        )
        safe_plotly(fig)


def _render_signal_history(wf: pd.DataFrame, signal: Dict[str, Any]) -> None:
    if wf is None or wf.empty or "y_pred" not in wf.columns:
        render_placeholder("Run pipeline to generate signal history")
        return
    rows = []
    thr = 8.0
    for day, g in list(wf.groupby(wf.index.date))[-7:]:
        pred = float(g["y_pred"].mean())
        naive = float(g["y_naive"].mean()) if "y_naive" in g.columns else pred
        actual = float(g["y_true"].mean()) if "y_true" in g.columns else None
        div = pred - naive
        if div > thr:
            direction = "LONG"
        elif div < -thr:
            direction = "SHORT"
        else:
            direction = "NEUTRAL"
        conviction = "HIGH" if abs(div) > 20 else ("MEDIUM" if abs(div) > 8 else "LOW")
        outcome = "—"
        if actual is not None and direction != "NEUTRAL":
            pnl = (1 if direction == "LONG" else -1) * (actual - naive)
            outcome = f"{'WIN' if pnl > 0 else 'LOSS'} ({pnl:+.1f})"
        rows.append(
            f"<tr><td>{day}</td><td>{direction}</td><td>{conviction}</td><td>{outcome}</td></tr>"
        )
    st.markdown(
        '<table class="curve-table"><thead><tr>'
        "<th>Date</th><th>Direction</th><th>Conviction</th><th>Outcome</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>",
        unsafe_allow_html=True,
    )
