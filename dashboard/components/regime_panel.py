"""Regime analysis panel — timeline, Dunkelflaute cards, interactive filter."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.dashboard_helpers import (
    render_placeholder,
    safe_plotly,
    safe_render,
    tab_section_header,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REGIME_FIG_DIR = PROJECT_ROOT / "outputs" / "figures" / "regime"

REGIME_NAMES = {0: "NEGATIVE/GLUT", 1: "LOW", 2: "NORMAL", 3: "HIGH/DUNKELFLAUTE"}
REGIME_COLORS = ["#10b981", "#6b7280", "#3b82f6", "#f59e0b"]

DUNKEL_EVENTS = [
    {
        "label": "Nov 2–7 2024 Dunkelflaute",
        "start": date(2024, 11, 2),
        "end": date(2024, 11, 7),
        "precedent": "€820/MWh intraday peak",
    },
    {
        "label": "Dec 12–14 2024 Dunkelflaute",
        "start": date(2024, 12, 12),
        "end": date(2024, 12, 14),
        "precedent": "€900/MWh intraday peak",
    },
]

REGIME_FIGURES = [
    ("regime_timeline.png", "Regime timeline"),
    ("dunkelflaute_events.png", "Dunkelflaute events"),
    ("regime_price_distribution.png", "Regime price distribution"),
    ("solar_cannibalization_scatter.png", "Solar cannibalization"),
]


def _safe_image(path: Path, caption: str) -> None:
    """Show figure if present; otherwise Streamlit Cloud-friendly placeholder."""
    try:
        if path.exists():
            st.image(str(path), caption=caption)
        else:
            st.info(f"{caption} not found. Run: `python run_pipeline.py --mode regime`")
    except Exception:
        st.info(f"{caption} unavailable. Run: `python run_pipeline.py --mode regime`")


def _coerce_regime_frame(regime_df: Optional[pd.DataFrame], wf: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Prefer master regimes; fall back to walk-forward so Cloud works without data/processed."""
    if regime_df is not None and not regime_df.empty:
        return regime_df
    if wf is None or wf.empty:
        return pd.DataFrame()
    out = wf.copy()
    if "da_price" not in out.columns and "y_true" in out.columns:
        out = out.rename(columns={"y_true": "da_price"})
    return out


@safe_render("Regime panel unavailable — run pipeline --mode regime")
def render_regime_panel(
    regime_df: pd.DataFrame,
    figures_dir: Path,
    wf: Optional[pd.DataFrame] = None,
) -> None:
    """Full regime analysis with Dunkelflaute cards and interactive filter."""
    tab_section_header("🔄 MARKET REGIME — Price regime detection and Dunkelflaute analysis")

    fig_dir = REGIME_FIG_DIR
    if figures_dir is not None:
        candidate = Path(figures_dir) / "regime"
        if candidate.exists():
            fig_dir = candidate

    df = _coerce_regime_frame(regime_df, wf)
    if df.empty:
        st.warning(
            "Regime parquet not available locally (`data/processed/` is gitignored). "
            "Showing committed figures from `outputs/figures/regime/`."
        )
        for name, caption in REGIME_FIGURES:
            _safe_image(fig_dir / name, caption)
        return

    st.divider()
    options = ["All regimes"] + [f"{i} — {REGIME_NAMES[i]}" for i in range(4)]
    selected = st.selectbox("Filter timeline by regime", options, index=0)
    plot_df = df
    if selected != "All regimes" and "price_regime" in df.columns:
        rid = int(selected.split("—")[0].strip())
        plot_df = df[df["price_regime"] == rid]

    st.subheader("Regime Timeline (2022–2024)")
    st.markdown(
        '<div class="chart-subtitle" style="margin-bottom:12px;">'
        "Nov 2024: €820/MWh peak | Dec 2024: €900/MWh peak — highest in 18 years"
        "</div>",
        unsafe_allow_html=True,
    )
    if "da_price" in plot_df.columns:
        step = max(1, len(plot_df) // 8000)
        p = plot_df.iloc[::step]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=p.index,
                y=p["da_price"],
                mode="lines",
                line=dict(color="#f9fafb", width=0.7),
                name="DA Price",
            )
        )
        fig.add_vrect(
            x0="2024-11-02",
            x1="2024-11-08",
            fillcolor="rgba(245,158,11,0.3)",
            line_width=0,
        )
        fig.add_vrect(
            x0="2024-12-12",
            x1="2024-12-15",
            fillcolor="rgba(245,158,11,0.3)",
            line_width=0,
        )
        fig.add_annotation(
            x="2024-11-05",
            y=900,
            text="<b>Nov DF</b>",
            showarrow=False,
            font=dict(size=14, color="#f59e0b", family="Inter"),
        )
        fig.add_annotation(
            x="2024-12-13",
            y=900,
            text="<b>Dec DF</b>",
            showarrow=False,
            font=dict(size=14, color="#f59e0b", family="Inter"),
        )
        fig.update_layout(
            title=dict(text="Regime Timeline (2022–2024)", x=0.0, xanchor="left"),
            height=360,
            xaxis=dict(rangeslider=dict(visible=True), type="date"),
            yaxis_title="EUR/MWh",
            dragmode="pan",
        )
        safe_plotly(fig)
        st.markdown(
            '<div class="chart-subtitle">'
            "Nov 2024: €820/MWh peak | Dec 2024: €900/MWh peak — highest in 18 years"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        _safe_image(fig_dir / "regime_timeline.png", "Regime timeline")

    st.divider()
    st.subheader("Dunkelflaute Event Cards")
    cols = st.columns(2)
    for col, ev in zip(cols, DUNKEL_EVENTS):
        with col:
            _dunkelflaute_card(df, ev)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if "price_regime" in df.columns:
            counts = df["price_regime"].value_counts().sort_index()
            labels = [REGIME_NAMES.get(int(i), str(i)) for i in counts.index]
            fig = px.pie(
                values=counts.values,
                names=labels,
                color_discrete_sequence=REGIME_COLORS,
                title="Regime distribution (click legend to isolate)",
            )
            fig.update_layout(height=340)
            safe_plotly(fig)
        else:
            _safe_image(fig_dir / "regime_price_distribution.png", "Regime price distribution")
    with c2:
        if "da_price" in df.columns and "price_regime" in df.columns:
            sample = df.sample(min(5000, len(df)), random_state=42)
            fig = px.violin(
                sample,
                x="price_regime",
                y="da_price",
                color="price_regime",
                color_discrete_sequence=REGIME_COLORS,
                title="Price by regime",
            )
            fig.update_layout(height=340, showlegend=False)
            safe_plotly(fig)

    st.divider()
    _render_solar_cannibalization(df, fig_dir)

    st.subheader("Saved Regime Figures")
    for name, caption in REGIME_FIGURES:
        _safe_image(fig_dir / name, caption)


def _render_solar_cannibalization(regime_df: pd.DataFrame, fig_dir: Path) -> None:
    """Interactive solar cannibalization scatter with expected negative trend."""
    st.subheader("Solar Cannibalization Effect — Higher Penetration Drives Prices Down")
    pen_col = None
    for c in ("renewable_penetration", "solar_penetration", "da_solar"):
        if c in regime_df.columns:
            pen_col = c
            break
    if pen_col is None or "da_price" not in regime_df.columns:
        _safe_image(fig_dir / "solar_cannibalization_scatter.png", "Solar cannibalization")
        return

    sample = regime_df[[pen_col, "da_price"]].dropna()
    if len(sample) > 4000:
        sample = sample.sample(4000, random_state=42)

    x = sample[pen_col].to_numpy(dtype=float)
    y = sample["da_price"].to_numpy(dtype=float)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(size=4, color="#3b82f6", opacity=0.35),
            name="Hours",
        )
    )
    x_line = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 50)
    if len(x) > 10 and np.nanstd(x) > 0:
        coef = np.polyfit(x, y, 1)
        y_fit = coef[0] * x_line + coef[1]
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_fit,
                mode="lines",
                line=dict(color="#ef4444", width=3),
                name="Trend (OLS)",
            )
        )
        if coef[0] >= 0:
            y0 = float(np.nanpercentile(y, 75))
            y1 = float(np.nanpercentile(y, 10))
            fig.add_trace(
                go.Scatter(
                    x=[x_line[0], x_line[-1]],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(color="#ef4444", width=2, dash="dash"),
                    name="Expected negative relationship",
                )
            )
    fig.update_layout(
        title=dict(
            text="Solar Cannibalization Effect — Higher Penetration Drives Prices Down",
            x=0.0,
            xanchor="left",
        ),
        height=420,
        xaxis_title=pen_col.replace("_", " ").title(),
        yaxis_title="DA Price (EUR/MWh)",
    )
    safe_plotly(fig)


def _dunkelflaute_card(df: pd.DataFrame, ev: dict) -> None:
    mask = (df.index.date >= ev["start"]) & (df.index.date <= ev["end"])
    sub = df.loc[mask]
    if sub.empty:
        st.markdown(
            f'<div class="event-card"><h4>{ev["label"]}</h4>'
            f'<div class="event-stat">No overlapping data in panel</div></div>',
            unsafe_allow_html=True,
        )
        return
    duration_h = len(sub)
    min_wind = float(sub["da_wind"].min()) if "da_wind" in sub.columns else float("nan")
    max_price = float(sub["da_price"].max()) if "da_price" in sub.columns else float("nan")
    if "da_price" in sub.columns:
        opp = float((sub["da_price"] - 80).clip(lower=0).sum())
    else:
        opp = float("nan")
    wind_txt = f"{min_wind:,.0f} MW" if min_wind == min_wind else "—"
    price_txt = f"{max_price:,.1f} EUR/MWh" if max_price == max_price else "—"
    opp_txt = f"{opp:,.0f} EUR" if opp == opp else "—"
    st.markdown(
        f'<div class="event-card">'
        f'<h4>{ev["label"]}</h4>'
        f'<div class="event-stat">Duration: <b>{duration_h} h</b> '
        f'({ev["start"]} → {ev["end"]})</div>'
        f'<div class="event-stat">Min wind output: <b>{wind_txt}</b></div>'
        f'<div class="event-stat">Max DA price: <b>{price_txt}</b></div>'
        f'<div class="event-stat">Revenue opportunity (1 MW vs €80): '
        f'<b>{opp_txt}</b></div>'
        f'<div class="event-stat">Historical precedent: <b>{ev["precedent"]}</b></div>'
        f"</div>",
        unsafe_allow_html=True,
    )
