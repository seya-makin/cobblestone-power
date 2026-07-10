"""
Cobblestone Power — LLM market commentary with anti-hallucination guard.

Purpose:
    Generate Bloomberg-style daily (and weekly) German power commentaries from
    pipeline-computed metrics only, using gemini-2.0-flash, and verify that
    every number in the text traces back to the input metrics.

Inputs:
    computed_metrics_dict (never invent figures).

Outputs:
    Commentary text; commentary_latest.json; JSONL audit log.

Side Effects:
    Gemini API calls; writes outputs/logs/llm_commentary_prompts.jsonl.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from config.settings import GEMINI_MODEL, get_settings
from src.utils import append_jsonl, utc_now_iso, write_json

logger = logging.getLogger(__name__)

NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


class MarketCommentator:
    """
    Senior-style power market commentary generator.

    Purpose:
        Turn model outputs into trader-readable notes without inventing numbers.

    Inputs:
        Pipeline metrics dict; optional 7-day history for weekly summary.

    Outputs:
        Commentary string + metadata (hallucination flag, word count).

    Side Effects:
        Gemini calls; JSON/JSONL persistence.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.log_path = self.settings.logs / "llm_commentary_prompts.jsonl"
        self._model = None

    def _configure(self, temperature: float = 0.2) -> bool:
        """Configure Gemini; return False if key missing."""
        if self.settings.gemini_key_is_placeholder():
            logger.warning("GEMINI_API_KEY placeholder — using template commentary")
            return False
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.settings.gemini_api_key)
            self._model = genai.GenerativeModel(
                GEMINI_MODEL,
                generation_config=genai.GenerationConfig(temperature=temperature),
            )
            return True
        except Exception as exc:
            logger.error("Gemini configure failed: %s", exc)
            return False

    def generate_daily_commentary(self, computed_metrics_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate ≤180-word daily commentary from pipeline metrics only.

        Args:
            computed_metrics_dict: Strictly pipeline-computed numbers.

        Returns:
            Dict with commentary, metadata, hallucination flag.
        """
        metrics = computed_metrics_dict
        prompt = f"""You are a senior power market analyst at a European proprietary energy trading firm.
Write a daily market commentary for the German day-ahead electricity market.

STRICT RULES:
- Use ONLY the numbers provided below. Do not invent any figures.
- Write in the style of a Bloomberg terminal market note: dense, precise, no filler.
- Use standard energy market terminology: baseload, peak, residual load, merit order,
  spark spread, Dunkelflaute, CCGT, EUA, TTF.
- Maximum 180 words.
- Structure: [1 sentence market summary] [2-3 sentences fundamental drivers]
  [1 sentence risk/signal] [1 sentence model note]
- End with: "Signal: [LONG/SHORT/NEUTRAL] prompt [week/month]."

TODAY'S PIPELINE-COMPUTED METRICS:
{json.dumps(metrics, indent=2)}

Write the commentary:
"""
        t0 = time.perf_counter()
        raw = ""
        if self._configure(0.2):
            try:
                response = self._model.generate_content(prompt)
                raw = (response.text or "").strip()
            except Exception as exc:
                logger.error("Commentary generation failed: %s", exc)
                raw = self._fallback_commentary(metrics)
        else:
            raw = self._fallback_commentary(metrics)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        hallu = self.check_for_hallucinations(raw, metrics)
        word_count = len(raw.split())

        record = {
            "timestamp": utc_now_iso(),
            "model": GEMINI_MODEL,
            "temperature": 0.2,
            "input_metrics_count": len(metrics),
            "prompt_chars": len(prompt),
            "response_chars": len(raw),
            "word_count": word_count,
            "raw_response": raw,
            "generation_time_ms": elapsed_ms,
            "contains_hallucination_flag": hallu,
        }
        append_jsonl(self.log_path, record)

        out = {
            "generated_at": utc_now_iso(),
            "model": GEMINI_MODEL,
            "commentary": raw,
            "word_count": word_count,
            "contains_hallucination_flag": hallu,
            "hallucination_check": "FAIL" if hallu else "PASS",
            "input_metrics": metrics,
        }
        write_json(self.settings.forecasts_dir / "commentary_latest.json", out)
        logger.info(
            "Commentary generated — %s words | hallucination=%s",
            word_count,
            "FAIL" if hallu else "PASS",
        )
        return out

    def check_for_hallucinations(self, commentary_text: str, metrics_dict: Dict[str, Any]) -> bool:
        """
        Extract numbers from commentary and cross-check against metrics (±0.5).

        Args:
            commentary_text: LLM output.
            metrics_dict: Source-of-truth metrics.

        Returns:
            True if any number cannot be traced (hallucination flag).
        """
        allowed: List[float] = []
        for v in metrics_dict.values():
            if isinstance(v, bool):
                allowed.extend([0.0, 1.0])
            elif isinstance(v, (int, float)):
                allowed.append(float(v))
            elif isinstance(v, str):
                for m in NUMBER_RE.findall(v):
                    allowed.append(float(m))

        # Also allow common integers like year fragments from dates
        for v in metrics_dict.values():
            if isinstance(v, str) and re.match(r"\d{4}-\d{2}-\d{2}", v):
                for part in re.findall(r"\d+", v):
                    allowed.append(float(part))

        found = [float(x) for x in NUMBER_RE.findall(commentary_text)]
        # Filter out pure structural numbers that appear in "Signal:" lines rarely
        for num in found:
            if any(abs(num - a) <= 0.5 for a in allowed):
                continue
            # Allow small integers 0-3 for regime references; confidence levels; common horizons
            if num in {0.0, 1.0, 2.0, 3.0, 7.0, 24.0, 30.0, 80.0, 90.0, 95.0, 100.0, 168.0}:
                continue
            logger.warning("Untraced number in commentary: %s", num)
            return True
        return False

    def generate_weekly_summary(self, daily_metrics_list: List[Dict[str, Any]]) -> str:
        """
        Synthesise a ~300-word weekly note from 7 days of metrics/commentaries.

        Args:
            daily_metrics_list: List of daily metric dicts (and optional commentary).

        Returns:
            Weekly summary text.
        """
        prompt = f"""You are a senior power market analyst. Write a 300-word weekly synthesis
for German day-ahead power covering: price range, dominant regime, forecast accuracy,
and prompt curve view. Use ONLY the numbers in the JSON below.

WEEKLY PIPELINE DATA:
{json.dumps(daily_metrics_list, indent=2, default=str)}
"""
        if self._configure(0.2):
            try:
                response = self._model.generate_content(prompt)
                text = (response.text or "").strip()
            except Exception as exc:
                logger.error("Weekly summary failed: %s", exc)
                text = self._fallback_weekly(daily_metrics_list)
        else:
            text = self._fallback_weekly(daily_metrics_list)

        append_jsonl(
            self.log_path,
            {
                "timestamp": utc_now_iso(),
                "model": GEMINI_MODEL,
                "type": "weekly_summary",
                "raw_response": text,
                "n_days": len(daily_metrics_list),
            },
        )
        write_json(
            self.settings.forecasts_dir / "commentary_weekly.json",
            {"generated_at": utc_now_iso(), "summary": text},
        )
        return text

    @staticmethod
    def _fallback_commentary(m: Dict[str, Any]) -> str:
        """Deterministic template when Gemini is unavailable."""
        regime = m.get("dominant_regime", 2)
        base = m.get("da_baseload_forecast_eur_mwh", float("nan"))
        peak = m.get("da_peak_forecast_eur_mwh", float("nan"))
        lo = m.get("conformal_90_low", float("nan"))
        hi = m.get("conformal_90_high", float("nan"))
        resid = m.get("residual_load_forecast_mw", float("nan"))
        wind = m.get("wind_forecast_mw", float("nan"))
        solar = m.get("solar_forecast_mw", float("nan"))
        pen = m.get("renewable_penetration_pct", float("nan"))
        dunkel = m.get("dunkelflaute_risk_score", 0.0)
        neg = m.get("negative_price_risk_score", 0.0)
        skill = m.get("model_skill_vs_naive_pct", float("nan"))
        top = m.get("top_shap_feature", "residual_load")
        direction = "NEUTRAL"
        if dunkel and float(dunkel) > 0.4:
            direction = "LONG"
        elif neg and float(neg) > 0.4:
            direction = "SHORT"
        return (
            f"DE DA fair value prints baseload {base} EUR/MWh / peak {peak} EUR/MWh "
            f"with Conformal 90% PI [{lo}, {hi}]. "
            f"Residual load {resid} MW with wind {wind} MW and solar {solar} MW "
            f"implies renewable penetration {pen}%. "
            f"Dominant regime {regime}; Dunkelflaute risk {dunkel}, negative-price risk {neg}. "
            f"Model note: skill vs naive {skill}%; top SHAP driver {top}. "
            f"Signal: {direction} prompt day."
        )

    @staticmethod
    def _fallback_weekly(days: List[Dict[str, Any]]) -> str:
        """Template weekly summary."""
        bases = [d.get("da_baseload_forecast_eur_mwh") for d in days if d.get("da_baseload_forecast_eur_mwh") is not None]
        if not bases:
            return "Insufficient metrics for weekly summary."
        return (
            f"Weekly DE DA fair-value range {min(bases):.1f}–{max(bases):.1f} EUR/MWh baseload "
            f"across {len(days)} sessions. Regime mix and conformal widths tracked in pipeline "
            f"outputs; see daily notes for Dunkelflaute and negative-price risk evolution."
        )
