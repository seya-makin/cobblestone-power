"""
Dashboard rendering helpers — safe wrappers, Plotly theme, placeholders.

Purpose:
    Prevent blank panels and Streamlit crashes when pipeline artefacts are
    missing or a chart fails. Every tab uses these helpers.
"""

from __future__ import annotations

import functools
import logging
import re
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

CHART_TEMPLATE: Dict[str, Any] = {
    "layout": {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "#111827",
        "font": {"family": "Inter, sans-serif", "color": "#6b7280", "size": 12},
        "title": {
            "font": {"family": "Inter", "color": "#f9fafb", "size": 14},
            "x": 0.0,
            "xanchor": "left",
            "pad": {"l": 0, "t": 0},
        },
        "xaxis": {
            "gridcolor": "#1f2937",
            "linecolor": "#1f2937",
            "tickfont": {"size": 11},
            "zeroline": False,
        },
        "yaxis": {
            "gridcolor": "#1f2937",
            "linecolor": "rgba(0,0,0,0)",
            "tickfont": {"size": 11},
            "zeroline": False,
        },
        "legend": {"bgcolor": "rgba(0,0,0,0)", "font": {"size": 11}},
        "margin": {"l": 48, "r": 24, "t": 48, "b": 40},
        "hoverlabel": {
            "bgcolor": "#1f2937",
            "bordercolor": "#374151",
            "font": {"family": "JetBrains Mono", "size": 12, "color": "#f9fafb"},
        },
    }
}

# Flat layout dict applied by safe_plotly (from CHART_TEMPLATE)
PLOTLY_LAYOUT: Dict[str, Any] = dict(CHART_TEMPLATE["layout"])

PLOTLY_CONFIG: Dict[str, Any] = {
    "displayModeBar": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "cobblestone_chart",
        "height": 600,
        "width": 1200,
        "scale": 2,
    },
}

FOOTER_HTML = (
    '<div class="dash-footer">'
    "Cobblestone Power Analytics v1.0.0 | Seya Makin | "
    "Data: SMARD (smard.de) — Bundesnetzagentur | "
    "Model: XGBoost + Conformal Prediction | LLM: Gemini 2.0 Flash"
    "</div>"
)


def safe_render(fallback_message: str = "Unable to render this panel.") -> Callable[[F], F]:
    """
    Decorator: wrap any rendering function so exceptions become styled cards.

    Args:
        fallback_message: Shown inside the error card on failure.

    Returns:
        Decorated callable that never raises into Streamlit.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                logger.exception("safe_render caught error in %s: %s", func.__name__, exc)
                render_error_card(fallback_message, detail=str(exc))
                return None

        return wrapper  # type: ignore[return-value]

    return decorator


def render_placeholder(message: str = "Run pipeline to generate this data") -> None:
    """Styled empty-state card when a parquet/JSON artefact is missing."""
    st.markdown(
        f'<div class="placeholder-card">'
        f'<div class="placeholder-icon"></div>'
        f'<div class="placeholder-title">{message}</div>'
        f'<div class="placeholder-sub">python run_pipeline.py --mode full</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_error_card(message: str, detail: str = "") -> None:
    """Clean error card — never a raw traceback in the UI."""
    detail_html = f'<div class="error-detail">{_escape(detail[:240])}</div>' if detail else ""
    st.markdown(
        f'<div class="error-card">'
        f'<div class="error-title">Panel unavailable</div>'
        f'<div class="error-msg">{_escape(message)}</div>'
        f"{detail_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def section_spacer() -> None:
    """Exactly 32px between dashboard sections."""
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)


def safe_plotly(fig: Any, **kwargs: Any) -> None:
    """Render a Plotly figure with the shared trading-terminal chart template."""
    try:
        if fig is None:
            render_placeholder("No chart data available")
            return
        layout = dict(PLOTLY_LAYOUT)
        # Preserve caller margins when already set (e.g. taller title room)
        m = fig.layout.margin
        has_margin = m is not None and (
            getattr(m, "l", None) is not None or getattr(m, "t", None) is not None
        )
        apply = {k: v for k, v in layout.items() if k != "margin" or not has_margin}
        # Merge axis/title without wiping caller titles — update_layout merges dicts
        fig.update_layout(**apply)
        # Force left-aligned title styling on any existing title text
        if fig.layout.title and fig.layout.title.text:
            fig.update_layout(
                title=dict(
                    font={"family": "Inter", "color": "#f9fafb", "size": 14},
                    x=0.0,
                    xanchor="left",
                    pad={"l": 0, "t": 0},
                )
            )
        st.plotly_chart(
            fig,
            use_container_width=True,
            config=PLOTLY_CONFIG,
            **{k: v for k, v in kwargs.items() if k not in ("use_container_width", "config")},
        )
    except Exception as exc:
        logger.exception("safe_plotly failed: %s", exc)
        render_error_card("Chart failed to render", str(exc))


def safe_dataframe(df: Any, **kwargs: Any) -> None:
    """Safe st.dataframe wrapper."""
    try:
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            render_placeholder("No table data available")
            return
        st.dataframe(df, use_container_width=True, **kwargs)
    except Exception as exc:
        logger.exception("safe_dataframe failed: %s", exc)
        render_error_card("Table failed to render", str(exc))


def _delta_class_and_text(delta: str) -> tuple[str, str]:
    """Map a delta string to (css_class, display text without arrows)."""
    raw = (delta or "").strip()
    if not raw:
        return "flat", ""
    m = re.search(r"([+-]?\d+\.?\d*)", raw)
    cls = "flat"
    if m:
        try:
            num = float(m.group(1))
            if num > 0:
                cls = "up"
            elif num < 0:
                cls = "down"
        except ValueError:
            cls = "flat"
    # Strip any legacy arrow glyphs
    raw = raw.replace("↑", "").replace("↓", "").strip()
    return cls, raw


def metric_card_html(
    label: str,
    value: str,
    delta: str = "",
    subtext: str = "",
    value_color: str = "#f9fafb",
    large: bool = False,
) -> str:
    """
    KPI card HTML — label / value / delta / subtext only.

    Args:
        label: 11px uppercase grey
        value: 24px (or 36px if large) JetBrains Mono
        delta: signed change — green/red via CSS class
        subtext: 11px grey units/metadata
    """
    size_cls = "metric-value-lg" if large else "metric-value"
    # Back-compat: if caller passed units in `delta` and no subtext, treat as subtext
    # when delta has no numeric sign pattern for a change.
    d_cls, d_txt = _delta_class_and_text(delta)
    # Heuristic: pure unit strings like "EUR/MWh" have no leading +/- number as change
    if delta and not subtext and not re.search(r"[+-]\d", delta) and "vs" not in delta.lower():
        subtext = delta
        d_txt = ""
    delta_html = (
        f'<div class="metric-delta {d_cls}">{_escape(d_txt)}</div>' if d_txt else ""
    )
    sub_html = f'<div class="metric-subtext">{_escape(subtext)}</div>' if subtext else ""
    return (
        f'<div class="metric-card">'
        f'<div class="metric-label">{_escape(label)}</div>'
        f'<div class="{size_cls}" style="color:{value_color}">{_escape(value)}</div>'
        f"{delta_html}"
        f"{sub_html}"
        f"</div>"
    )


def sidebar_metric_html(label: str, value: str) -> str:
    """Compact sidebar row — JetBrains Mono value, right-aligned 13px."""
    return (
        f'<div class="sidebar-metric">'
        f'<span class="lbl">{_escape(label)}</span>'
        f'<span class="val">{_escape(value)}</span>'
        f"</div>"
    )


def load_json_safe(path: Path) -> Dict[str, Any]:
    """Load JSON or return empty dict."""
    try:
        if path.exists():
            import json

            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load JSON %s: %s", path, exc)
    return {}


def load_parquet_safe(path: Path) -> pd.DataFrame:
    """Load parquet or return empty DataFrame."""
    try:
        if path.exists():
            return pd.read_parquet(path)
    except Exception as exc:
        logger.warning("Failed to load parquet %s: %s", path, exc)
    return pd.DataFrame()


def pipeline_step_status(settings: Any) -> Dict[str, bool]:
    """Return which pipeline artefacts exist (for sidebar checklist)."""
    return {
        "Ingest": (settings.data_raw / "ingestion_manifest.json").exists()
        or (settings.data_raw / "prices" / "da_price.parquet").exists(),
        "Clean": settings.master_dataset.exists(),
        "QA": (settings.qa_report / "qa_summary.json").exists(),
        "Regime": (settings.data_processed / "master_with_regimes.parquet").exists(),
        "Features": settings.features_path.exists(),
        "Validate": (settings.forecasts_dir / "walk_forward_results.parquet").exists(),
        "Forecast": (settings.forecasts_dir / "latest_forecast.json").exists(),
        "Backtest": (settings.forecasts_dir / "backtest_stats.json").exists(),
        "Commentary": (settings.forecasts_dir / "commentary_latest.json").exists(),
        "Submission": settings.submission_csv.exists(),
    }


def system_status_dot(
    settings: Any,
    has_forecasts: bool,
    metrics: Optional[Dict[str, Any]] = None,
) -> tuple[str, str, str]:
    """
    Return (css_class, label, color) for system status.

    Prefers walk_forward_metrics.json: if MAE exists and MAE < 50, show LIVE SMARD.
    Otherwise amber SYNTHETIC. Red if pipeline never run.
    """
    metrics = metrics or {}
    mae = metrics.get("MAE")
    metrics_path = Path(settings.forecasts_dir) / "walk_forward_metrics.json"
    if metrics_path.exists() and mae is not None:
        try:
            if float(mae) < 50.0:
                return "status-dot-green", "LIVE DATA — SMARD (smard.de)", "#10b981"
        except (TypeError, ValueError):
            pass
        return "status-dot-orange", "SYNTHETIC DATA", "#f59e0b"
    if not has_forecasts and not settings.master_dataset.exists():
        return "status-dot-red", "PIPELINE NEVER RUN", "#ef4444"
    return "status-dot-orange", "SYNTHETIC DATA", "#f59e0b"


def render_tab_footer() -> None:
    """Footer on every tab."""
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)


def tab_section_header(text: str) -> None:
    """Tab hero header — 18px Inter, primary text, 24px bottom margin."""
    st.markdown(
        f'<div class="tab-section-header">{_escape(text)}</div>',
        unsafe_allow_html=True,
    )
