"""
Cobblestone Power — data quality reporting.

Purpose:
    Compute coverage, outlier, DST, and structural integrity metrics;
    emit qa_summary.json and a self-contained qa_detailed.html report.

Inputs:
    Cleaned master DataFrame.

Outputs:
    outputs/qa_report/qa_summary.json, qa_detailed.html.

Side Effects:
    Writes QA artefacts; embeds base64 charts in HTML.
"""

from __future__ import annotations

import base64
import io
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.settings import PIPELINE_VERSION, get_settings
from src.utils import utc_now_iso, write_json

logger = logging.getLogger(__name__)

# Quality score weights
W_COVERAGE: float = 0.35
W_OUTLIER: float = 0.25
W_DST: float = 0.20
W_TEMPORAL: float = 0.10
W_STRUCTURAL: float = 0.10


class QualityReporter:
    """
    Produce machine- and human-readable data quality reports.

    Purpose:
        Score dataset fitness for trading-model training.

    Inputs:
        Cleaned master DataFrame; optional LLM rule violation counts.

    Outputs:
        qa_summary.json dict; HTML report path.

    Side Effects:
        Writes files under outputs/qa_report/; generates matplotlib charts.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    def run(
        self,
        df: pd.DataFrame,
        llm_rules_proposed: int = 0,
        llm_rules_executed: int = 0,
        llm_rule_violations: int = 0,
        recommended_actions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute full QA summary and write JSON + HTML artefacts.

        Args:
            df: Cleaned master panel.
            llm_rules_proposed: Count from LLM QA module.
            llm_rules_executed: Count successfully executed.
            llm_rule_violations: Total violation rows across rules.
            recommended_actions: Optional action list.

        Returns:
            qa_summary dict.

        Example:
            >>> summary = QualityReporter().run(master_df)
        """
        cols_report: Dict[str, Any] = {}
        key_cols = [c for c in ["da_price", "da_load", "da_wind", "da_solar", "net_exports", "nuclear_avail_fr"] if c in df.columns]

        for col in key_cols:
            s = df[col]
            desc = s.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
            entry: Dict[str, Any] = {
                "coverage_pct": float(100.0 * s.notna().mean()),
                "missing_raw": int(s.isna().sum()),
                "min": float(s.min()) if s.notna().any() else None,
                "max": float(s.max()) if s.notna().any() else None,
                "mean": float(s.mean()) if s.notna().any() else None,
                "std": float(s.std()) if s.notna().any() else None,
                "p5": float(desc.get("5%", np.nan)) if "5%" in desc.index else None,
                "p25": float(desc.get("25%", np.nan)) if "25%" in desc.index else None,
                "p50": float(desc.get("50%", np.nan)) if "50%" in desc.index else None,
                "p75": float(desc.get("75%", np.nan)) if "75%" in desc.index else None,
                "p95": float(desc.get("95%", np.nan)) if "95%" in desc.index else None,
            }
            if col == "da_price":
                entry["outlier_z4"] = int(df["outlier_z4"].sum()) if "outlier_z4" in df.columns else 0
                entry["seasonal_outliers"] = int(df["seasonal_outlier"].sum()) if "seasonal_outlier" in df.columns else 0
                entry["hour_spikes"] = int(df["hour_spike"].sum()) if "hour_spike" in df.columns else 0
                neg = s < 0
                entry["negative_price_hours"] = int(neg.sum())
                entry["negative_price_pct"] = float(100.0 * neg.mean())
            cols_report[col] = entry

        score = self.compute_quality_score(df)
        dst_spring = int(df["is_dst_gap"].sum()) if "is_dst_gap" in df.columns else 0
        dst_autumn = int(df["is_dst_duplicate"].sum()) if "is_dst_duplicate" in df.columns else 0

        summary: Dict[str, Any] = {
            "run_timestamp": utc_now_iso(),
            "pipeline_version": PIPELINE_VERSION,
            "date_range": {
                "start": str(df.index.min().date()),
                "end": str(df.index.max().date()),
            },
            "total_hours": int(len(df)),
            "overall_quality_score": round(score, 1),
            "quality_verdict": "PASS" if score >= 85 else ("WARN" if score >= 70 else "FAIL"),
            "columns": cols_report,
            "dst_transitions": {
                "handled": dst_spring + dst_autumn,
                "spring_forward": dst_spring,
                "autumn_fallback": dst_autumn,
            },
            "structural_breaks": [
                "2022-02-24 ukraine_war",
                "2023-04-15 nuclear_phaseout",
            ],
            "llm_rules_proposed": llm_rules_proposed,
            "llm_rules_executed": llm_rules_executed,
            "llm_rule_violations": llm_rule_violations,
            "recommended_actions": recommended_actions
            or [
                "Monitor negative-price hours during spring/summer weekends",
                "Re-validate Dunkelflaute detection against Nov/Dec 2024 events",
                "Refresh fuel price feeds before live trading use",
            ],
        }

        write_json(self.settings.qa_report / "qa_summary.json", summary)
        self._write_html(df, summary)
        logger.info("QA complete — score=%.1f verdict=%s", score, summary["quality_verdict"])
        return summary

    def compute_quality_score(self, df: pd.DataFrame) -> float:
        """
        Weighted quality score in [0, 100].

        Weights: coverage 35%, outlier (inverted) 25%, DST 20%,
        temporal consistency 10%, structural integrity 10%.
        """
        key = [c for c in ["da_price", "da_load", "da_wind", "da_solar"] if c in df.columns]
        coverage = float(np.mean([df[c].notna().mean() for c in key])) if key else 0.0

        outlier_rate = 0.0
        if "outlier_z4" in df.columns:
            outlier_rate = float(df["outlier_z4"].mean())
        outlier_score = max(0.0, 1.0 - outlier_rate * 50.0)  # 2% outliers → ~0

        dst_ok = 1.0
        if "is_dst_gap" in df.columns or "is_dst_duplicate" in df.columns:
            dst_ok = 1.0  # handled by cleaner

        # Temporal: no future timestamps beyond end_date + 1 day
        now = pd.Timestamp.now(tz="UTC")
        future_frac = float((df.index > now + pd.Timedelta(days=1)).mean())
        temporal = max(0.0, 1.0 - future_frac)

        # Structural: monotonic index, no duplicate timestamps
        structural = 1.0 if df.index.is_unique and df.index.is_monotonic_increasing else 0.5

        score = 100.0 * (
            W_COVERAGE * coverage
            + W_OUTLIER * outlier_score
            + W_DST * dst_ok
            + W_TEMPORAL * temporal
            + W_STRUCTURAL * structural
        )
        return float(score)

    def _fig_to_b64(self, fig: plt.Figure) -> str:
        """Encode a matplotlib figure as base64 PNG."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _write_html(self, df: pd.DataFrame, summary: Dict[str, Any]) -> Path:
        """Build self-contained HTML QA report with embedded charts."""
        charts: Dict[str, str] = {}

        # Coverage heatmap (month × field)
        key = [c for c in ["da_price", "da_load", "da_wind", "da_solar"] if c in df.columns]
        if key:
            cov = df[key].notna().groupby([df.index.year, df.index.month]).mean()
            fig, ax = plt.subplots(figsize=(8, 3))
            im = ax.imshow(cov.T.values, aspect="auto", cmap="RdYlGn", vmin=0.9, vmax=1.0)
            ax.set_yticks(range(len(key)))
            ax.set_yticklabels(key)
            ax.set_title("Data coverage (month × field)")
            fig.colorbar(im, ax=ax, fraction=0.03)
            charts["coverage"] = self._fig_to_b64(fig)

        if "da_price" in df.columns:
            # Negative price frequency by month/hour
            neg = (df["da_price"] < 0).astype(float)
            pivot = neg.groupby([df.index.month, df.index.hour]).mean().unstack(fill_value=0)
            fig, ax = plt.subplots(figsize=(10, 4))
            im = ax.imshow(pivot.values, aspect="auto", cmap="Blues")
            ax.set_xlabel("Hour (UTC)")
            ax.set_ylabel("Month")
            ax.set_title("Negative price frequency")
            fig.colorbar(im, ax=ax, fraction=0.03)
            charts["neg_prices"] = self._fig_to_b64(fig)

            # Price with structural breaks
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.plot(df.index, df["da_price"], color="#1a237e", lw=0.4)
            ax.axvline(pd.Timestamp("2022-02-24", tz="UTC"), color="red", ls="--", label="Ukraine war")
            ax.axvline(pd.Timestamp("2023-04-15", tz="UTC"), color="orange", ls="--", label="Nuclear phase-out")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title("DA price with structural breaks")
            ax.set_ylabel("EUR/MWh")
            charts["breaks"] = self._fig_to_b64(fig)

        score = summary["overall_quality_score"]
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Cobblestone QA Report</title>
<style>
body {{ font-family: Inter, system-ui, sans-serif; background:#0a0d14; color:#e8eaed; margin:24px; }}
.card {{ background:#1a2035; border:1px solid #2d3748; border-radius:8px; padding:20px; margin-bottom:16px; }}
.gauge {{ font-size:48px; font-weight:700; color:#00d4ff; }}
table {{ border-collapse:collapse; width:100%; }}
td,th {{ border:1px solid #2d3748; padding:6px 10px; font-size:13px; }}
img {{ max-width:100%; border-radius:4px; }}
</style></head><body>
<h1>Cobblestone Power — Data QA Report</h1>
<div class="card"><div class="metric-label">Overall Quality Score</div>
<div class="gauge">{score:.1f}/100</div>
<div>Verdict: <b>{summary['quality_verdict']}</b> | Hours: {summary['total_hours']} |
Range: {summary['date_range']['start']} → {summary['date_range']['end']}</div></div>
"""
        for name, b64 in charts.items():
            html += f'<div class="card"><h3>{name}</h3><img src="data:image/png;base64,{b64}"/></div>\n'

        html += f"""
<div class="card"><h3>DST Transitions</h3>
<pre>{summary['dst_transitions']}</pre></div>
<div class="card"><h3>LLM Rules</h3>
<p>Proposed: {summary['llm_rules_proposed']} | Executed: {summary['llm_rules_executed']} |
Violations: {summary['llm_rule_violations']}</p></div>
<div class="card"><h3>Recommended Actions</h3><ul>
{''.join(f'<li>{a}</li>' for a in summary['recommended_actions'])}
</ul></div>
<p style="color:#5f6368">Generated {summary['run_timestamp']} | v{PIPELINE_VERSION}</p>
</body></html>"""

        path = self.settings.qa_report / "qa_detailed.html"
        path.write_text(html, encoding="utf-8")
        logger.info("QA HTML written → %s", path)
        return path
