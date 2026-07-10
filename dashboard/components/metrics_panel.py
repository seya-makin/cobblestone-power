"""Validation metrics panel — large KPIs, Dunkelflaute bands, benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.dashboard_helpers import (
    metric_card_html,
    render_placeholder,
    safe_plotly,
    safe_render,
    section_spacer,
    tab_section_header,
)


@safe_render("Validation panel unavailable — run pipeline --mode validate")
def render_metrics_panel(
    metrics: Dict[str, Any],
    figures_dir: Path,
    wf: pd.DataFrame | None = None,
) -> None:
    """Headline metrics, MAE chart with DF bands, benchmark table."""
    tab_section_header("MODEL VALIDATION — Walk-forward performance vs benchmarks")
    if not metrics:
        render_placeholder("Run pipeline to generate this data")
        return

    section_spacer()
    mae = metrics.get("MAE")
    cov = metrics.get("conformal_coverage_90_empirical")
    direc = metrics.get("directional_accuracy_pct")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            metric_card_html(
                "MAE",
                f"{mae:.2f}" if mae is not None else "—",
                subtext="EUR/MWh",
                value_color="#3b82f6",
                large=True,
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            metric_card_html(
                "Conformal Coverage 90%",
                f"{100 * cov:.1f}%" if cov is not None else "—",
                subtext="target ≥ 90%",
                value_color="#10b981" if cov and cov >= 0.90 else "#f59e0b",
                large=True,
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            metric_card_html(
                "Directional Accuracy",
                f"{direc:.1f}%" if direc is not None else "—",
                large=True,
            ),
            unsafe_allow_html=True,
        )

    section_spacer()
    # Walk-forward MAE / forecast chart with Dunkelflaute bands
    st.subheader("Walk-forward Forecast vs Actual")
    if wf is not None and not wf.empty and "y_pred" in wf.columns:
        step = max(1, len(wf) // 6000)
        p = wf.iloc[::step]
        fig = go.Figure()
        if "y_true" in p.columns:
            fig.add_trace(
                go.Scatter(x=p.index, y=p["y_true"], name="Actual", line=dict(color="#f9fafb", width=1))
            )
        fig.add_trace(
            go.Scatter(x=p.index, y=p["y_pred"], name="XGBoost", line=dict(color="#3b82f6", width=1.2))
        )
        if "conformal_90_low" in p.columns:
            fig.add_trace(
                go.Scatter(
                    x=list(p.index) + list(p.index[::-1]),
                    y=list(p["conformal_90_high"]) + list(p["conformal_90_low"][::-1]),
                    fill="toself",
                    fillcolor="rgba(59,130,246,0.08)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="Conformal 90% PI",
                    hoverinfo="skip",
                )
            )
        fig.add_vrect(
            x0="2024-11-02",
            x1="2024-11-08",
            fillcolor="rgba(245,158,11,0.18)",
            line_width=0,
            annotation_text="Nov 2024 DF",
        )
        fig.add_vrect(
            x0="2024-12-12",
            x1="2024-12-15",
            fillcolor="rgba(245,158,11,0.18)",
            line_width=0,
            annotation_text="Dec 2024 DF",
        )
        fig.update_layout(
            title=dict(text="Walk-forward Forecast vs Actual", x=0.0, xanchor="left"),
            height=380,
            yaxis_title="EUR/MWh",
            xaxis=dict(rangeslider=dict(visible=True)),
        )
        safe_plotly(fig)
    else:
        path = figures_dir / "validation" / "walk_forward_forecast_vs_actual.png"
        if path.exists():
            st.image(str(path))
        else:
            render_placeholder("Walk-forward results not available")

    # Published benchmarks
    st.subheader("Published Benchmark Comparison")
    your_mae = f"{mae:.1f}" if mae is not None else "—"
    # Highlight winner (lowest MAE) — on synthetic data our MAE may be higher; still show honestly
    rows = [
        ("Ridge (Marcjasz et al. 2023)", "~12", 12.0),
        ("XGBoost (Marcjasz et al. 2023)", "~9", 9.0),
        ("Neural (Marcjasz et al. 2023)", "~8", 8.0),
        ("Cobblestone XGBoost + Conformal", your_mae, float(mae) if mae is not None else 999.0),
    ]
    best = min(r[2] for r in rows)
    html_rows = []
    for name, val, num in rows:
        cls = "win-cell" if abs(num - best) < 1e-9 else ""
        html_rows.append(f'<tr><td>{name}</td><td class="{cls}">{val} EUR/MWh</td></tr>')
    st.markdown(
        '<table class="qa-table"><thead><tr><th>Model</th><th>MAE (DE day-ahead)</th></tr></thead>'
        f"<tbody>{''.join(html_rows)}</tbody></table>"
        f'<p style="color:#6b7280;font-size:11px;margin-top:8px;letter-spacing:0.06em;text-transform:uppercase;">'
        f"Published MAE benchmarks for DE day-ahead (Marcjasz et al. 2023): "
        f"Ridge ~12 EUR/MWh, XGBoost ~9 EUR/MWh, Neural ~8 EUR/MWh. "
        f"Your model: <b style='color:#3b82f6;font-family:JetBrains Mono,monospace;'>{your_mae} EUR/MWh</b>. "
        f"Note: offline run may use synthetic fundamentals — live ENTSO-E data required for like-for-like comparison."
        f"</p>",
        unsafe_allow_html=True,
    )

    # Metrics comparison with winner highlight
    st.subheader("Model Metrics Comparison")
    skill_n = metrics.get("skill_vs_naive_pct")
    skill_r = metrics.get("skill_vs_ridge_pct")
    # Approximate naive/ridge MAE from skill
    naive_mae = mae / (1 - skill_n / 100) if mae and skill_n is not None and skill_n < 100 else None
    ridge_mae = mae / (1 - skill_r / 100) if mae and skill_r is not None and skill_r < 100 else None
    comp = [
        ("MAE ↓", naive_mae, ridge_mae, mae),
        ("Skill vs Naive % ↑", 0.0, skill_r, skill_n),
        ("Dir. Accuracy % ↑", None, None, direc),
        ("Cov 90% ↑", None, None, 100 * cov if cov is not None else None),
    ]
    body = []
    for label, n, r, x in comp:
        vals = [("Naive", n), ("Ridge", r), ("XGBoost", x)]
        numeric = [(k, v) for k, v in vals if v is not None]
        if "↓" in label and numeric:
            winner = min(numeric, key=lambda t: t[1])[0]
        elif "↑" in label and numeric:
            winner = max(numeric, key=lambda t: t[1])[0]
        else:
            winner = None

        def cell(name: str, v: Any) -> str:
            if v is None:
                return "<td>—</td>"
            cls = "win-cell" if name == winner else ""
            return f'<td class="{cls}">{v:.2f}</td>'

        body.append(
            f"<tr><td>{label}</td>{cell('Naive', n)}{cell('Ridge', r)}{cell('XGBoost', x)}</tr>"
        )
    st.markdown(
        '<table class="qa-table"><thead><tr>'
        "<th>Metric</th><th>Naive</th><th>Ridge</th><th>XGBoost</th>"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>",
        unsafe_allow_html=True,
    )

    st.subheader("Validation Figures")
    for name in [
        "residuals_by_hour.png",
        "residuals_by_regime.png",
        "metrics_dashboard.png",
        "quantile_reliability.png",
        "feature_importance_stability.png",
        "shap_waterfall_dunkelflaute.png",
        "shap_summary.png",
    ]:
        path = figures_dir / "validation" / name
        if path.exists():
            try:
                st.image(str(path), caption=name)
            except Exception:
                pass
