"""Forecast chart — Plotly DA forecast with conformal PIs (never empty)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.dashboard_helpers import render_placeholder, safe_plotly, safe_render

REGIME_COLORS = {
    0: "rgba(16,185,129,0.10)",
    1: "rgba(107,114,128,0.08)",
    2: "rgba(59,130,246,0.06)",
    3: "rgba(245,158,11,0.12)",
}
REGIME_NAMES = {0: "GLUT", 1: "LOW", 2: "NORMAL", 3: "DUNKELFLAUTE"}
DISPLAY_BAND_CLIP_EUR: float = 150.0


def _clip_band(lo: pd.Series, hi: pd.Series, center: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Clip conformal bands to center ± DISPLAY_BAND_CLIP_EUR for display only."""
    lo_c = np.maximum(lo.to_numpy(dtype=float), center.to_numpy(dtype=float) - DISPLAY_BAND_CLIP_EUR)
    hi_c = np.minimum(hi.to_numpy(dtype=float), center.to_numpy(dtype=float) + DISPLAY_BAND_CLIP_EUR)
    return pd.Series(lo_c, index=lo.index), pd.Series(hi_c, index=hi.index)


@safe_render("Forecast chart unavailable — run pipeline --mode validate")
def render_forecast_chart(df: pd.DataFrame, title: Optional[str] = None) -> None:
    """Render main forecast chart; always draws something when df has y_pred."""
    if df is None or df.empty or "y_pred" not in getattr(df, "columns", []):
        render_placeholder("Run pipeline to generate this data")
        return

    fig = go.Figure()
    center = df["y_pred"]
    y_hi = float(center.max()) + DISPLAY_BAND_CLIP_EUR
    y_lo = float(center.min()) - DISPLAY_BAND_CLIP_EUR

    if "price_regime" in df.columns:
        regimes = df["price_regime"].fillna(2).astype(int)
        changes = regimes.ne(regimes.shift()).cumsum()
        for _, g in df.groupby(changes):
            r = int(g["price_regime"].iloc[0])
            fig.add_vrect(
                x0=g.index.min(),
                x1=g.index.max(),
                fillcolor=REGIME_COLORS.get(r, "rgba(0,0,0,0)"),
                layer="below",
                line_width=0,
            )

    if "conformal_90_low" in df.columns and "conformal_90_high" in df.columns:
        lo90, hi90 = _clip_band(df["conformal_90_low"], df["conformal_90_high"], center)
        y_hi = max(y_hi, float(hi90.max()))
        y_lo = min(y_lo, float(lo90.min()))
        fig.add_trace(
            go.Scatter(
                x=list(df.index) + list(df.index[::-1]),
                y=list(hi90) + list(lo90[::-1]),
                fill="toself",
                fillcolor="rgba(59,130,246,0.08)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Conformal 90% PI (display-clipped)",
                hoverinfo="skip",
            )
        )
    if "conformal_80_low" in df.columns and "conformal_80_high" in df.columns:
        lo80, hi80 = _clip_band(df["conformal_80_low"], df["conformal_80_high"], center)
        fig.add_trace(
            go.Scatter(
                x=list(df.index) + list(df.index[::-1]),
                y=list(hi80) + list(lo80[::-1]),
                fill="toself",
                fillcolor="rgba(59,130,246,0.14)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Conformal 80% PI (display-clipped)",
                hoverinfo="skip",
            )
        )

    if "y_naive" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["y_naive"],
                name="Seasonal Naive",
                line=dict(color="#374151", width=1, dash="dash"),
            )
        )
    if "y_true" in df.columns:
        # Clip actuals into display window so axis stays readable
        actual_disp = df["y_true"].clip(lower=y_lo, upper=y_hi)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=actual_disp,
                name="Actual",
                line=dict(color="#f9fafb", width=2),
            )
        )

    marker_mask = np.arange(len(df)) % 3 == 0
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["y_pred"],
            name="XGBoost",
            mode="lines+markers",
            line=dict(color="#3b82f6", width=4),
            marker=dict(
                color="#3b82f6",
                size=8,
                symbol="circle",
                line=dict(color="#0a0d14", width=1),
                opacity=[1.0 if m else 0.0 for m in marker_mask],
            ),
        )
    )
    fig.add_hline(y=0, line_color="#ef4444", line_width=1, annotation_text="€0/MWh")

    regime_name = "—"
    if "price_regime" in df.columns and df["price_regime"].notna().any():
        regime_name = REGIME_NAMES.get(int(df["price_regime"].mode().iloc[0]), "NORMAL")
    day_name = df.index[0].strftime("%A") if len(df) else ""
    date_str = str(df.index[0].date()) if len(df) else ""
    chart_title = title or f"DE Day-Ahead Forecast — {date_str} — {day_name} — Regime: {regime_name}"
    fig.update_layout(
        title=dict(text=chart_title, x=0.0, xanchor="left", y=0.98, yanchor="top"),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, xanchor="left"),
        margin=dict(l=48, r=24, t=88, b=40),
        xaxis_title="Hour (UTC)",
        yaxis_title="EUR/MWh",
        yaxis=dict(range=[y_lo - 10, y_hi + 10]),
        hovermode="x unified",
    )
    st.markdown('<div class="forecast-chart-container">', unsafe_allow_html=True)
    safe_plotly(fig)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="chart-subtitle">'
        "Conformal bands shown ±150 for clarity — full intervals in submission.csv"
        "</div>",
        unsafe_allow_html=True,
    )
