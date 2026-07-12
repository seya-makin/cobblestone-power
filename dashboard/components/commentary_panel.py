"""Market commentary panel — styled card, hallucination badge, audit log."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from utils.dashboard_helpers import render_placeholder, safe_render, tab_section_header


def _gemini_is_placeholder() -> bool:
    try:
        from config.settings import get_settings

        return bool(get_settings().gemini_key_is_placeholder())
    except Exception:
        return True


@safe_render("Commentary panel unavailable — run pipeline --mode commentary")
def render_commentary_panel(
    commentary: Dict[str, Any],
    logs_dir: Path,
    delivery_view: Dict[str, Any] | None = None,
    signal: Dict[str, Any] | None = None,
) -> None:
    """Today's commentary, model vs market card, searchable audit log."""
    tab_section_header("MARKET COMMENTARY — AI-generated analyst note (Gemini 2.0 Flash)")

    has_text = bool((commentary or {}).get("commentary"))
    if not has_text:
        msg = (
            "Commentary requires GEMINI_API_KEY — add key to .env and run "
            "python run_pipeline.py --mode commentary"
            if _gemini_is_placeholder()
            else "Run pipeline to generate this data"
        )
        st.markdown(
            f'<div class="placeholder-card">'
            f'<div class="placeholder-icon"></div>'
            f'<div class="placeholder-title">{msg}</div>'
            f'<div class="placeholder-sub">Gemini 2.0 Flash · anti-hallucination guard enabled</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        return

    st.divider()
    hallu = commentary.get("hallucination_check", "—")
    if hallu == "PASS" or commentary.get("contains_hallucination_flag") is False:
        st.markdown(
            '<div class="hallu-pass">VERIFIED — No hallucinations detected</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="hallu-fail">WARNING — HALLUCINATION FLAG — Review required</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="commentary-card">{commentary.get("commentary", "")}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Generated at `{commentary.get('generated_at')}` · "
        f"model: `{commentary.get('model')}` · "
        f"words: **{commentary.get('word_count')}**"
    )

    # Model Says vs Market Implies
    st.subheader("Model Says vs Market Implies")
    metrics = commentary.get("input_metrics") or {}
    model_bl = metrics.get("da_baseload_forecast_eur_mwh")
    if model_bl is None and delivery_view:
        model_bl = delivery_view.get("baseload_tomorrow")
    naive_proxy = None
    if metrics.get("forecast_vs_same_day_last_week_eur") is not None and model_bl is not None:
        try:
            naive_proxy = float(model_bl) - float(metrics["forecast_vs_same_day_last_week_eur"])
        except (TypeError, ValueError):
            naive_proxy = None
    if model_bl is None:
        render_placeholder("Baseload forecast not available for comparison")
    else:
        delta = ""
        if naive_proxy is not None:
            try:
                d = float(model_bl) - float(naive_proxy)
                delta = f"{d:+.1f} EUR/MWh vs naive"
            except (TypeError, ValueError):
                delta = ""
        st.markdown(
            f'<div class="compare-card">'
            f'<div class="compare-side"><div class="label">Model fair value (baseload)</div>'
            f'<div class="value">{float(model_bl):.1f}</div>'
            f'<div style="color:#6b7280;font-size:11px;">{delta}</div></div>'
            f'<div class="compare-side"><div class="label">Market / naive imply</div>'
            f'<div class="value">{f"{naive_proxy:.1f}" if naive_proxy is not None else "—"}</div>'
            f'<div style="color:#6b7280;font-size:11px;">Seasonal naive (lag-168)</div></div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    with st.expander("Input metrics JSON"):
        st.json(metrics)
    with st.expander("Raw LLM response"):
        st.code(commentary.get("commentary", ""))

    st.subheader("LLM Prompt Audit Log")
    log_path = logs_dir / "llm_commentary_prompts.jsonl"
    if not log_path.exists():
        render_placeholder("No commentary audit log yet")
        return

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    records: List[Dict[str, Any]] = []
    for x in lines:
        if not x.strip():
            continue
        try:
            records.append(json.loads(x))
        except Exception:
            continue
    if not records:
        render_placeholder("No commentary audit log yet")
        return
    st.dataframe(pd_safe(records), use_container_width=True)


def pd_safe(records: List[Dict[str, Any]]):
    import pandas as pd

    return pd.DataFrame(records)
