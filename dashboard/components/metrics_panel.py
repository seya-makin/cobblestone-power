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
    tab_section_header("MODEL VALIDATION — Walk-forward performance vs published benchmarks")
    if not metrics:
        render_placeholder("Run pipeline to generate this data")
        return

    section_spacer()
    mae = metrics.get("MAE")
    mae_full = metrics.get("mae_full_period", mae)
    mae_post = metrics.get("mae_post_crisis")
    cov = metrics.get("conformal_coverage_90_empirical")
    direc = metrics.get("directional_accuracy_pct")
    skill_n = metrics.get("skill_vs_naive_pct")
    neg_recall = metrics.get("negative_price_recall")
    post = metrics.get("post_crisis") or {}

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            metric_card_html(
                "MAE (full period)",
                f"{mae_full:.2f}" if mae_full is not None else "—",
                subtext="EUR/MWh · train 2022+",
                value_color="#3b82f6",
                large=True,
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            metric_card_html(
                "MAE (post-crisis)",
                f"{mae_post:.2f}" if mae_post is not None else "—",
                subtext="EUR/MWh · train 2023+",
                value_color="#10b981" if mae_post is not None else "#6b7280",
                large=True,
            ),
            unsafe_allow_html=True,
        )
    with c3:
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
    with c4:
        st.markdown(
            metric_card_html(
                "Directional Accuracy",
                f"{direc:.1f}%" if direc is not None else "—",
                large=True,
            ),
            unsafe_allow_html=True,
        )

    if mae_post is not None:
        section_spacer()
        post_dir = post.get("directional_accuracy_pct")
        post_neg = post.get("negative_price_recall")
        post_cov = post.get("conformal_coverage_90_empirical")
        skill_bit = f" &nbsp;·&nbsp; Skill {skill_n:+.1f}%" if skill_n is not None else ""
        dir_bit = f" &nbsp;·&nbsp; Dir. {float(post_dir):.1f}%" if post_dir is not None else ""
        neg_bit = (
            f" &nbsp;·&nbsp; Neg. recall {100 * float(post_neg):.1f}%"
            if post_neg is not None
            else ""
        )
        cov_bit = (
            f" &nbsp;·&nbsp; Cov90 {100 * float(post_cov):.1f}%"
            if post_cov is not None
            else ""
        )
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-label">Dual-model walk-forward (2024 test)</div>'
            '<div style="font-size:13px;color:#f9fafb;line-height:1.7;font-weight:300;margin-top:8px;">'
            "<b>Full period</b> (train 2022+, includes Ukraine crisis): "
            f"<span style='font-family:JetBrains Mono,monospace;color:#3b82f6;'>"
            f"{float(mae_full):.2f} EUR/MWh</span>{skill_bit}<br>"
            "<b>Post-crisis</b> (train 2023-01-01+, crisis excluded): "
            f"<span style='font-family:JetBrains Mono,monospace;color:#10b981;'>"
            f"{float(mae_post):.2f} EUR/MWh</span>{dir_bit}{neg_bit}{cov_bit}"
            "</div></div>",
            unsafe_allow_html=True,
        )

    section_spacer()
    # Published benchmarks — directly below headline metrics
    your_mae = f"{mae_full:.2f}" if mae_full is not None else "—"
    your_post = f"{mae_post:.2f}" if mae_post is not None else None
    post_bit = (
        f' / <b style="color:#10b981;font-family:JetBrains Mono,monospace;">'
        f"{your_post} EUR/MWh (post-crisis)</b>"
        if your_post
        else ""
    )
    neg_bit_pub = (
        f" &nbsp;·&nbsp; Neg. price recall: {100 * float(neg_recall):.1f}%"
        if neg_recall is not None
        else ""
    )
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">Published MAE Benchmarks — DE Day-Ahead</div>'
        f'<div style="font-size:13px;color:#f9fafb;line-height:1.7;font-weight:300;margin-top:8px;">'
        f"Ridge: ~12 EUR/MWh &nbsp;·&nbsp; "
        f"XGBoost (calm market): ~9 EUR/MWh &nbsp;·&nbsp; "
        f"XGBoost (2022-2024 including Ukraine crisis): ~25-30 EUR/MWh<br>"
        f"This system: <b style='color:#3b82f6;font-family:JetBrains Mono,monospace;'>"
        f"{your_mae} EUR/MWh (full period)</b>"
        f"{post_bit}"
        f"{neg_bit_pub}"
        f"</div></div>",
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
            text="<b>Nov DF — €820/MWh</b>",
            showarrow=False,
            font=dict(size=12, color="#f59e0b", family="Inter"),
        )
        fig.add_annotation(
            x="2024-12-13",
            y=900,
            text="<b>Dec DF — €900/MWh</b>",
            showarrow=False,
            font=dict(size=12, color="#f59e0b", family="Inter"),
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

    # Detailed published benchmark table
    st.subheader("Published Benchmark Comparison")
    rows = [
        ("Ridge (Marcjasz et al. 2023)", "~12", 12.0),
        ("XGBoost calm market (Marcjasz et al. 2023)", "~9", 9.0),
        ("XGBoost 2022–2024 incl. crisis", "~25–30", 27.5),
        (
            "Cobblestone full period (train 2022+)",
            your_mae,
            float(mae_full) if mae_full is not None else 999.0,
        ),
    ]
    if mae_post is not None:
        rows.append(
            (
                "Cobblestone post-crisis (train 2023+)",
                f"{float(mae_post):.2f}",
                float(mae_post),
            )
        )
    html_rows = []
    for name, val, _num in rows:
        cls = "win-cell" if "Cobblestone" in name else ""
        html_rows.append(f'<tr><td>{name}</td><td class="{cls}">{val} EUR/MWh</td></tr>')
    st.markdown(
        '<table class="qa-table"><thead><tr><th>Model</th><th>MAE (DE day-ahead)</th></tr></thead>'
        f"<tbody>{''.join(html_rows)}</tbody></table>"
        f'<p style="color:#6b7280;font-size:11px;margin-top:8px;letter-spacing:0.06em;text-transform:uppercase;">'
        f"Published MAE benchmarks for DE day-ahead — Ridge: ~12 EUR/MWh, "
        f"XGBoost (calm market): ~9 EUR/MWh, "
        f"XGBoost (2022-2024 including Ukraine crisis): ~25-30 EUR/MWh. "
        f"This system: <b style='color:#3b82f6;font-family:JetBrains Mono,monospace;'>{your_mae} EUR/MWh (2024)</b>."
        f"</p>",
        unsafe_allow_html=True,
    )

    # Metrics comparison with winner highlight
    st.subheader("Model Metrics Comparison")
    # Reconstruct absolute MAEs, then Ridge skill vs naive (NOT XGB-vs-Ridge)
    naive_mae = None
    ridge_mae = None
    if mae is not None and skill_n is not None and skill_n < 100:
        try:
            naive_mae = float(mae) / (1.0 - float(skill_n) / 100.0)
        except ZeroDivisionError:
            naive_mae = None
    # Prefer reconstructing ridge MAE from XGB skill-vs-ridge if available
    skill_r_xgb = metrics.get("skill_vs_ridge_pct")
    if mae is not None and skill_r_xgb is not None and skill_r_xgb < 100:
        try:
            ridge_mae = float(mae) / (1.0 - float(skill_r_xgb) / 100.0)
        except ZeroDivisionError:
            ridge_mae = None
    # Prefer explicit ridge skill vs naive from metrics JSON when present
    ridge_skill_vs_naive = None
    if metrics.get("ridge_skill_vs_naive_pct") is not None:
        ridge_skill_vs_naive = float(metrics["ridge_skill_vs_naive_pct"])
    if metrics.get("naive_mae") is not None:
        naive_mae = float(metrics["naive_mae"])
    if metrics.get("ridge_mae") is not None:
        ridge_mae = float(metrics["ridge_mae"])
    if ridge_skill_vs_naive is None and naive_mae and ridge_mae and naive_mae > 0:
        ridge_skill_vs_naive = 100.0 * (1.0 - ridge_mae / naive_mae)

    comp = [
        ("MAE (lower better)", naive_mae, ridge_mae, mae),
        ("Skill vs Naive % (higher better)", 0.0, ridge_skill_vs_naive, skill_n),
        ("Dir. Accuracy % (higher better)", None, None, direc),
        ("Cov 90% (higher better)", None, None, 100 * cov if cov is not None else None),
    ]
    body = []
    for label, n, r, x in comp:
        vals = [("Naive", n), ("Ridge", r), ("XGBoost", x)]
        numeric = [(k, v) for k, v in vals if v is not None]
        if "lower better" in label and numeric:
            winner = min(numeric, key=lambda t: t[1])[0]
        elif "higher better" in label and numeric:
            winner = max(numeric, key=lambda t: t[1])[0]
        else:
            winner = None

        def cell(name: str, v: Any) -> str:
            if v is None:
                return "<td>—</td>"
            cls = "win-cell" if name == winner else ""
            style = ""
            if "Skill" in label and v < 0:
                style = "color:#ef4444;font-family:JetBrains Mono,monospace;"
                cls = ""
            elif style == "" and "Skill" in label:
                style = "font-family:JetBrains Mono,monospace;"
            return f'<td class="{cls}" style="{style}">{v:.2f}</td>'

        body.append(
            f"<tr><td>{label}</td>{cell('Naive', n)}{cell('Ridge', r)}{cell('XGBoost', x)}</tr>"
        )
    st.markdown(
        '<table class="qa-table"><thead><tr>'
        "<th>Metric</th><th>Naive</th><th>Ridge</th><th>XGBoost</th>"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
        + (
            f'<p style="color:#6b7280;font-size:11px;margin-top:8px;">'
            f"Ridge MAE {ridge_mae:.2f} &gt; Naive MAE {naive_mae:.2f} - "
            f"Ridge skill vs naive <span style='color:#ef4444;font-family:JetBrains Mono,monospace;'>"
            f"{ridge_skill_vs_naive:.1f}%</span>.</p>"
            if ridge_mae and naive_mae and ridge_skill_vs_naive is not None
            else ""
        ),
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
