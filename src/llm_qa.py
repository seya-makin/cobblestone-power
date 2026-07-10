"""
Cobblestone Power — LLM-proposed data validation rules.

Purpose:
    Use gemini-2.0-flash to propose 20 electricity-market-specific QA rules,
    execute them safely against the master DataFrame, and audit all prompts.

Inputs:
    Schema description, sample rows, dataset stats; GEMINI_API_KEY.

Outputs:
    List of rule dicts; violation counts; JSONL audit log.

Side Effects:
    Gemini API calls; writes outputs/logs/llm_qa_prompts.jsonl.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import GEMINI_MODEL, get_settings
from src.utils import append_jsonl, safe_eval_condition, utc_now_iso, write_json

logger = logging.getLogger(__name__)


FALLBACK_RULES: List[Dict[str, Any]] = [
    {
        "rule_id": "R001",
        "description": "DA price within physical bounds [-500, 3000] EUR/MWh",
        "python_condition": "(df['da_price'] < -500) | (df['da_price'] > 3000)",
        "violation_means": "Price outside EPEX physical/technical bounds",
        "severity": "ERROR",
        "physical_rationale": "German DA prices bounded by market rules and technical limits",
        "expected_violation_rate_pct": 0.0,
    },
    {
        "rule_id": "R002",
        "description": "Load never below 20,000 MW",
        "python_condition": "df['da_load'] < 20000",
        "violation_means": "Implausibly low German system load",
        "severity": "ERROR",
        "physical_rationale": "German load floor historically ~25 GW; <20 GW indicates corruption",
        "expected_violation_rate_pct": 0.0,
    },
    {
        "rule_id": "R003",
        "description": "Load never above 100,000 MW",
        "python_condition": "df['da_load'] > 100000",
        "violation_means": "Implausibly high German system load",
        "severity": "ERROR",
        "physical_rationale": "Peak German load historically <90 GW",
        "expected_violation_rate_pct": 0.0,
    },
    {
        "rule_id": "R004",
        "description": "Solar zero when Berlin local hour outside 4-21",
        "python_condition": "(df['europe_berlin_hour'] < 4) | (df['europe_berlin_hour'] > 21) if 'europe_berlin_hour' in df.columns else (df.index.hour < 4) | (df.index.hour > 21)",
        "violation_means": "Night-time solar generation",
        "severity": "WARNING",
        "physical_rationale": "Solar must be ~0 between sunset and sunrise at lat 51.2N",
        "expected_violation_rate_pct": 0.1,
    },
    {
        "rule_id": "R005",
        "description": "Night solar must be near zero",
        "python_condition": "((df.index.hour < 4) | (df.index.hour > 21)) & (df['da_solar'] > 500)",
        "violation_means": "Material solar at night",
        "severity": "ERROR",
        "physical_rationale": "Physics: no irradiance at night",
        "expected_violation_rate_pct": 0.0,
    },
    {
        "rule_id": "R006",
        "description": "Wind below installed capacity ~76 GW",
        "python_condition": "df['da_wind'] > 76000",
        "violation_means": "Wind exceeds installed capacity",
        "severity": "ERROR",
        "physical_rationale": "~68 GW onshore + ~8 GW offshore (2024)",
        "expected_violation_rate_pct": 0.0,
    },
    {
        "rule_id": "R007",
        "description": "DE nuclear zero after phase-out",
        "python_condition": "(df.index >= '2023-04-15') & (df['nuclear_avail_de'] > 10) if 'nuclear_avail_de' in df.columns else pd.Series(False, index=df.index)",
        "violation_means": "Non-zero DE nuclear after 2023-04-15",
        "severity": "ERROR",
        "physical_rationale": "German nuclear phase-out completed 15 Apr 2023",
        "expected_violation_rate_pct": 0.0,
    },
    {
        "rule_id": "R008",
        "description": "Negative prices should coincide with high renewables",
        "python_condition": "(df['da_price'] < 0) & ((df['da_wind'] + df['da_solar']) / df['da_load'] < 0.4)",
        "violation_means": "Negative price without high renewable penetration",
        "severity": "WARNING",
        "physical_rationale": "Negative prices correlate with renewable glut",
        "expected_violation_rate_pct": 1.0,
    },
    {
        "rule_id": "R009",
        "description": "Dunkelflaute hours should not have very low prices",
        "python_condition": "((df['da_wind'] + df['da_solar']) / df['da_load'] < 0.10) & (df['da_price'] < 20)",
        "violation_means": "Low price during renewable drought",
        "severity": "WARNING",
        "physical_rationale": "Dunkelflaute typically pushes prices >100 EUR/MWh",
        "expected_violation_rate_pct": 0.5,
    },
    {
        "rule_id": "R010",
        "description": "Hour-on-hour price change flag consistency",
        "python_condition": "(df['da_price'].diff().abs() > 500) & (~df['hour_spike']) if 'hour_spike' in df.columns else df['da_price'].diff().abs() > 500",
        "violation_means": "Large spike without hour_spike flag",
        "severity": "WARNING",
        "physical_rationale": "Spikes >500 EUR/MWh/h should be flagged",
        "expected_violation_rate_pct": 0.1,
    },
    {
        "rule_id": "R011",
        "description": "Residual load finite",
        "python_condition": "~np.isfinite(df['da_load'] - df['da_wind'] - df['da_solar'])",
        "violation_means": "Non-finite residual load",
        "severity": "ERROR",
        "physical_rationale": "All fundamentals must be finite for modelling",
        "expected_violation_rate_pct": 0.0,
    },
    {
        "rule_id": "R012",
        "description": "Renewable penetration in [0, 1.5]",
        "python_condition": "((df['da_wind'] + df['da_solar']) / df['da_load'] < 0) | ((df['da_wind'] + df['da_solar']) / df['da_load'] > 1.5)",
        "violation_means": "Implausible renewable penetration",
        "severity": "WARNING",
        "physical_rationale": "Penetration can exceed 100% on export-heavy hours but not 150%+",
        "expected_violation_rate_pct": 0.2,
    },
    {
        "rule_id": "R013",
        "description": "FR nuclear availability positive",
        "python_condition": "df['nuclear_avail_fr'] < 1000 if 'nuclear_avail_fr' in df.columns else pd.Series(False, index=df.index)",
        "violation_means": "FR nuclear near zero — check outage feed",
        "severity": "WARNING",
        "physical_rationale": "French fleet ~61 GW; <1 GW is extreme",
        "expected_violation_rate_pct": 0.0,
    },
    {
        "rule_id": "R014",
        "description": "No duplicate UTC timestamps",
        "python_condition": "pd.Series(df.index.duplicated(), index=df.index)",
        "violation_means": "Duplicate index entries",
        "severity": "ERROR",
        "physical_rationale": "Hourly panel must be unique in UTC",
        "expected_violation_rate_pct": 0.0,
    },
    {
        "rule_id": "R015",
        "description": "Wind non-negative",
        "python_condition": "df['da_wind'] < 0",
        "violation_means": "Negative wind generation",
        "severity": "ERROR",
        "physical_rationale": "Generation cannot be negative",
        "expected_violation_rate_pct": 0.0,
    },
    {
        "rule_id": "R016",
        "description": "Solar non-negative",
        "python_condition": "df['da_solar'] < 0",
        "violation_means": "Negative solar generation",
        "severity": "ERROR",
        "physical_rationale": "Generation cannot be negative",
        "expected_violation_rate_pct": 0.0,
    },
    {
        "rule_id": "R017",
        "description": "Weekend negative price enrichment",
        "python_condition": "(df['da_price'] < 0) & (df.index.dayofweek < 5) & ((df['da_wind'] + df['da_solar']) / df['da_load'] < 0.5)",
        "violation_means": "Weekday negative price without strong renewables",
        "severity": "WARNING",
        "physical_rationale": "Most negative hours are weekend renewable gluts",
        "expected_violation_rate_pct": 2.0,
    },
    {
        "rule_id": "R018",
        "description": "TTF gas price positive when present",
        "python_condition": "(df['ttf_gas_price'] <= 0) if 'ttf_gas_price' in df.columns else pd.Series(False, index=df.index)",
        "violation_means": "Non-positive TTF",
        "severity": "ERROR",
        "physical_rationale": "Gas commodity price must be positive",
        "expected_violation_rate_pct": 0.0,
    },
    {
        "rule_id": "R019",
        "description": "EUA carbon price positive",
        "python_condition": "(df['eua_carbon_price'] <= 0) if 'eua_carbon_price' in df.columns else pd.Series(False, index=df.index)",
        "violation_means": "Non-positive EUA",
        "severity": "ERROR",
        "physical_rationale": "Carbon allowance price must be positive",
        "expected_violation_rate_pct": 0.0,
    },
    {
        "rule_id": "R020",
        "description": "Post phase-out flag consistency",
        "python_condition": "(df.index >= '2023-04-15') & (~df['post_nuclear_phaseout']) if 'post_nuclear_phaseout' in df.columns else pd.Series(False, index=df.index)",
        "violation_means": "Phase-out flag false after 2023-04-15",
        "severity": "ERROR",
        "physical_rationale": "Structural break column must align with calendar",
        "expected_violation_rate_pct": 0.0,
    },
]


class LLMQualityAssurance:
    """
    Propose and execute market-specific validation rules via Gemini.

    Purpose:
        Catch physically impossible or suspicious German power data.

    Inputs:
        Master DataFrame; schema/stats JSON for the LLM prompt.

    Outputs:
        rules list; execution results dict.

    Side Effects:
        Gemini calls; JSONL audit log; optional rules JSON on disk.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.log_path = self.settings.logs / "llm_qa_prompts.jsonl"
        self._model = None

    def _configure(self) -> bool:
        """Configure Gemini client; return False if key missing."""
        if self.settings.gemini_key_is_placeholder():
            logger.warning("GEMINI_API_KEY placeholder — using fallback QA rules")
            return False
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.settings.gemini_api_key)
            self._model = genai.GenerativeModel(
                GEMINI_MODEL,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            return True
        except Exception as exc:
            logger.error("Gemini configure failed: %s", exc)
            return False

    def propose_validation_rules(
        self,
        schema_description: str,
        sample_rows_json: str,
        dataset_stats_json: str,
    ) -> List[Dict[str, Any]]:
        """
        Ask Gemini for exactly 20 executable validation rules.

        Args:
            schema_description: Column/type description.
            sample_rows_json: ~10 sample rows as JSON.
            dataset_stats_json: Summary statistics JSON.

        Returns:
            List of 20 rule dicts.

        Example:
            >>> rules = LLMQualityAssurance().propose_validation_rules(s, rows, stats)
        """
        prompt = f"""You are a senior data engineer specialising in European electricity market data pipelines.
You are reviewing a German day-ahead electricity price dataset for a proprietary trading firm.

DATASET SCHEMA:
{schema_description}

SAMPLE ROWS (10 rows, JSON format):
{sample_rows_json}

DATASET STATISTICS:
{dataset_stats_json}

Propose exactly 20 concrete, executable data validation rules.

Requirements:
1. Specific to electricity market data — no generic statistical rules
2. Physically grounded in how the German power market actually works
3. Expressible as a Python boolean condition on a pandas DataFrame
4. Distinguish ERROR (data corruption) from WARNING (anomaly worth investigating)

Domain knowledge to apply:
- German DA prices: physically bounded [-500, 3000] EUR/MWh
- German load: typically 30,000-85,000 MW, never below 20,000 MW
- Solar generation must be zero between sunset and sunrise (Germany: lat 51.2N)
- Wind cannot exceed installed capacity (~68 GW onshore, ~8 GW offshore as of 2024)
- Negative prices correlate with: high wind+solar, low load, weekend, spring/summer
- Dunkelflaute: wind+solar < 10% of load for 24+ hours → prices typically > 100 EUR/MWh
- German nuclear phase-out April 15 2023: DE nuclear must be 0 after this date

Return ONLY a JSON array:
[
  {{
    "rule_id": "R001",
    "description": "...",
    "python_condition": "...",
    "violation_means": "...",
    "severity": "ERROR" or "WARNING",
    "physical_rationale": "...",
    "expected_violation_rate_pct": 0.0
  }}
]
"""
        t0 = time.perf_counter()
        rules: List[Dict[str, Any]] = FALLBACK_RULES
        raw = ""
        parse_success = True

        if self._configure():
            try:
                response = self._model.generate_content(prompt)
                raw = response.text or ""
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and "rules" in parsed:
                    parsed = parsed["rules"]
                if isinstance(parsed, list) and len(parsed) >= 10:
                    rules = parsed[:20]
                else:
                    parse_success = False
                    rules = FALLBACK_RULES
            except Exception as exc:
                logger.error("LLM QA propose failed: %s", exc)
                raw = traceback.format_exc()
                parse_success = False
                rules = FALLBACK_RULES
        else:
            raw = json.dumps(FALLBACK_RULES)
            parse_success = True

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        append_jsonl(
            self.log_path,
            {
                "timestamp": utc_now_iso(),
                "model": GEMINI_MODEL,
                "temperature": 0.1,
                "prompt_chars": len(prompt),
                "response_chars": len(raw),
                "rules_proposed": len(rules),
                "parse_success": parse_success,
                "raw_response": raw[:50_000],
                "parsed_rules": rules,
                "generation_time_ms": elapsed_ms,
            },
        )
        write_json(self.settings.qa_report / "llm_qa_rules.json", {"rules": rules})
        return rules

    def execute_rules(self, df: pd.DataFrame, rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute each rule condition safely; never crash the pipeline.

        Args:
            df: Master DataFrame.
            rules: Rule dicts with python_condition.

        Returns:
            Summary with per-rule violation counts and flagged indices.
        """
        results: List[Dict[str, Any]] = []
        total_violations = 0
        executed = 0

        for rule in rules:
            rid = rule.get("rule_id", "?")
            cond = rule.get("python_condition", "False")
            try:
                # Special-case night solar rule R004 — condition returns mask of night hours;
                # for R004 we check solar on those hours separately if needed.
                mask = safe_eval_condition(cond, df)
                if rid == "R004":
                    # Reinterpret: night hours with solar > 100 MW
                    night = mask
                    mask = night & (df["da_solar"] > 100) if "da_solar" in df.columns else night & False
                n_viol = int(mask.sum())
                idx = df.index[mask][:50].astype(str).tolist()
                executed += 1
                total_violations += n_viol
                results.append(
                    {
                        "rule_id": rid,
                        "severity": rule.get("severity", "WARNING"),
                        "violations": n_viol,
                        "sample_indices": idx,
                        "error": None,
                    }
                )
            except Exception as exc:
                logger.error("Rule %s failed: %s\n%s", rid, exc, traceback.format_exc())
                results.append(
                    {
                        "rule_id": rid,
                        "severity": rule.get("severity", "WARNING"),
                        "violations": 0,
                        "sample_indices": [],
                        "error": str(exc),
                    }
                )

        summary = {
            "rules_executed": executed,
            "rules_proposed": len(rules),
            "total_violations": total_violations,
            "results": results,
        }
        write_json(self.settings.qa_report / "llm_qa_results.json", summary)
        logger.info("LLM QA executed %s/%s rules | violations=%s", executed, len(rules), total_violations)
        return summary

    def run(self, df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        End-to-end: propose rules from data sample, then execute.

        Args:
            df: Master DataFrame.

        Returns:
            (rules, execution_summary).
        """
        schema = {c: str(df[c].dtype) for c in df.columns}
        sample = df.head(10).reset_index().to_json(orient="records", date_format="iso")
        stats = df.describe(include="all").to_json()
        rules = self.propose_validation_rules(json.dumps(schema), sample or "[]", stats or "{}")
        summary = self.execute_rules(df, rules)
        return rules, summary
