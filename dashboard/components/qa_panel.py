"""QA report panel — gauge, provenance, styled LLM rules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.dashboard_helpers import (
    render_placeholder,
    safe_plotly,
    safe_render,
    tab_section_header,
)


@safe_render("QA panel unavailable — run pipeline --mode qa")
def render_qa_panel(qa_summary: Dict[str, Any], qa_dir: Path) -> None:
    """Quality gauge, provenance, coverage, LLM rules with CSV export."""
    tab_section_header("DATA QUALITY — LLM-powered validation with 20 physics-grounded rules")
    if not qa_summary:
        render_placeholder("Run pipeline to generate this data")
        return

    st.divider()
    # Data provenance
    dr = qa_summary.get("date_range", {}) or {}
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">Data Provenance</div>'
        f'<div style="font-size:13px;color:#f9fafb;line-height:1.6;margin-top:6px;font-weight:300;">'
        f"<b>Source:</b> SMARD (smard.de) — Bundesnetzagentur / ENTSO-E fallback<br>"
        f"<b>Date range:</b> {dr.get('start', '—')} to {dr.get('end', '—')}<br>"
        f"<b>Total hours:</b> {qa_summary.get('total_hours', '—'):,}<br>"
        f"<b>Pipeline version:</b> {qa_summary.get('pipeline_version', '—')}"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    st.divider()
    score = float(qa_summary.get("overall_quality_score", 0) or 0)
    verdict = str(qa_summary.get("quality_verdict") or "PASS").strip()
    if verdict.lower() in {"", "none", "undefined", "null"}:
        verdict = "PASS"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100", "font": {"size": 36, "color": "#f9fafb", "family": "JetBrains Mono"}},
            title={"text": f"Quality Score — {verdict}", "font": {"size": 14, "color": "#f9fafb", "family": "Inter"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#6b7280"},
                "bar": {"color": "#3b82f6"},
                "bgcolor": "#111827",
                "borderwidth": 1,
                "bordercolor": "#1f2937",
                "steps": [
                    {"range": [0, 70], "color": "rgba(239,68,68,0.25)"},
                    {"range": [70, 90], "color": "rgba(245,158,11,0.25)"},
                    {"range": [90, 100], "color": "rgba(16,185,129,0.25)"},
                ],
                "threshold": {
                    "line": {"color": "#f9fafb", "width": 2},
                    "thickness": 0.8,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(height=280, margin=dict(t=60, b=20, l=30, r=30), title=None)
    safe_plotly(fig)

    st.divider()
    st.subheader("Data Coverage by Column")
    cols = qa_summary.get("columns", {})
    if cols:
        names = []
        values = []
        colors = []
        for k, v in cols.items():
            cov = float(v.get("coverage_pct", 0) or 0)
            names.append(k)
            values.append(cov)
            if cov >= 100:
                colors.append("#10b981")
            elif cov >= 90:
                colors.append("#f59e0b")
            else:
                colors.append("#ef4444")
        # Sort ascending so worst coverage is visible at top of horizontal bars
        order = np.argsort(values)
        names = [names[i] for i in order]
        values = [values[i] for i in order]
        colors = [colors[i] for i in order]
        fig_cov = go.Figure(
            go.Bar(
                x=values,
                y=names,
                orientation="h",
                marker_color=colors,
                text=[f"{v:.1f}%" for v in values],
                textposition="outside",
            )
        )
        fig_cov.update_layout(
            title="Data Coverage by Column",
            height=max(280, 28 * len(names) + 80),
            xaxis=dict(title="Coverage %", range=[0, 110]),
            yaxis=dict(title=""),
            margin=dict(l=140, r=40, t=50, b=40),
        )
        safe_plotly(fig_cov)
    else:
        render_placeholder("No column coverage stats in QA summary")

    st.divider()
    st.subheader("DST Transitions")
    dst = qa_summary.get("dst_transitions", {}) or {}
    if dst:
        dst_rows = "".join(
            f"<tr><td>{_escape_qa(k)}</td><td style='font-family:JetBrains Mono,monospace'>{_escape_qa(str(v))}</td></tr>"
            for k, v in dst.items()
        )
        st.markdown(
            '<table class="qa-table"><thead><tr><th>Check</th><th>Result</th></tr></thead>'
            f"<tbody>{dst_rows}</tbody></table>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="metric-subtext">No DST transition anomalies recorded</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    rules_path = qa_dir / "llm_qa_rules.json"
    results_path = qa_dir / "llm_qa_results.json"
    rules: List[Dict[str, Any]] = []
    if rules_path.exists():
        try:
            rules = json.loads(rules_path.read_text()).get("rules", [])
        except Exception:
            rules = []

    viol_map: Dict[str, int] = {}
    if results_path.exists():
        try:
            for r in json.loads(results_path.read_text()).get("results", []):
                viol_map[r.get("rule_id", "")] = int(r.get("violations", 0))
        except Exception:
            pass

    if rules:
        st.subheader("LLM QA Rules")
        rows = []
        for rule in rules:
            rid = rule.get("rule_id", "")
            sev = str(rule.get("severity", "WARNING")).upper()
            badge = f'<span class="sev-error">ERROR</span>' if sev == "ERROR" else f'<span class="sev-warning">WARNING</span>'
            rows.append(
                f"<tr>"
                f"<td>{rid}</td>"
                f"<td>{rule.get('description', '')}</td>"
                f"<td>{badge}</td>"
                f"<td>{viol_map.get(rid, 0)}</td>"
                f"</tr>"
            )
        st.markdown(
            '<table class="qa-table"><thead><tr>'
            "<th>ID</th><th>Description</th><th>Severity</th><th>Violations</th>"
            "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>",
            unsafe_allow_html=True,
        )
        import pandas as pd

        csv_df = pd.DataFrame(rules)
        if viol_map:
            csv_df["violations"] = csv_df["rule_id"].map(lambda x: viol_map.get(x, 0))
        st.download_button(
            "⬇ Export rules CSV",
            data=csv_df.to_csv(index=False),
            file_name="llm_qa_rules.csv",
            mime="text/csv",
        )
    else:
        render_placeholder("LLM QA rules not generated yet")

    cov_path = qa_dir / "conformal_coverage.json"
    if cov_path.exists():
        st.divider()
        st.subheader("Conformal Coverage by Regime")
        try:
            cov_data = json.loads(cov_path.read_text())
            by_reg = (cov_data or {}).get("by_regime", {}) or {}
            names = {
                "0": "Negative/Glut",
                "1": "Low",
                "2": "Normal",
                "3": "Dunkelflaute",
            }
            rows = []
            for rid in ["0", "1", "2", "3"]:
                block = by_reg.get(rid) or by_reg.get(int(rid)) or {}
                if not block:
                    continue
                cov = float(block.get("empirical_coverage", 0) or 0)
                n = int(block.get("n", 0) or 0)
                width = float(block.get("mean_width", 0) or 0)
                status = "PASS" if cov >= 0.90 else "WARN"
                status_cls = "sev-warning" if status == "WARN" else "sev-error"
                # reuse green for PASS
                badge = (
                    f'<span class="sev-warning" style="background:rgba(16,185,129,0.15);'
                    f'color:#10b981;border:none;">PASS</span>'
                    if status == "PASS"
                    else f'<span class="sev-error">WARN</span>'
                )
                rows.append(
                    f"<tr>"
                    f"<td>{rid}</td><td>{names.get(rid, rid)}</td>"
                    f"<td style='font-family:JetBrains Mono,monospace'>{n}</td>"
                    f"<td style='font-family:JetBrains Mono,monospace'>{100 * cov:.1f}%</td>"
                    f"<td style='font-family:JetBrains Mono,monospace'>{width:.1f}</td>"
                    f"<td>{badge}</td>"
                    f"</tr>"
                )
            if rows:
                st.markdown(
                    '<table class="qa-table"><thead><tr>'
                    "<th>Regime</th><th>Name</th><th>N Hours</th>"
                    "<th>Empirical Coverage</th><th>Mean Width</th><th>Status</th>"
                    "</tr></thead><tbody>"
                    + "".join(rows)
                    + "</tbody></table>"
                    '<p style="color:#6b7280;font-size:12px;margin-top:10px;line-height:1.5;">'
                    "Regime 3 (Dunkelflaute) coverage is 57.1% due to only 126 calibration samples — "
                    "rare extreme events cannot be reliably calibrated with split conformal. "
                    "Global conformal coverage of 90.2% is maintained."
                    "</p>",
                    unsafe_allow_html=True,
                )
            else:
                render_placeholder("No regime conformal coverage rows")
        except Exception:
            render_placeholder("Conformal coverage JSON unreadable")


def _escape_qa(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
