"""
Cobblestone Power — LLM-assisted ENTSO-E ingestion config generation.

Purpose:
    Use gemini-2.0-flash to convert ENTSO-E field documentation into a
    structured JSON ingestion configuration (column names, units, codes).

Inputs:
    data/raw/entsoe_field_docs.txt; GEMINI_API_KEY from settings.

Outputs:
    Structured ingestion config dict; JSONL audit log.

Side Effects:
    Calls Gemini API when key is present; writes outputs/logs/llm_config_generation.jsonl.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import GEMINI_MODEL, get_settings
from src.utils import append_jsonl, utc_now_iso, write_json

logger = logging.getLogger(__name__)

FALLBACK_CONFIG: Dict[str, Any] = {
    "fields": [
        {
            "name": "da_price",
            "document_type": "A44",
            "unit": "EUR/MWh",
            "resolution": "PT60M",
            "domain": "BZN|DE-LU",
            "folder": "prices",
        },
        {
            "name": "da_load",
            "document_type": "A65",
            "unit": "MW",
            "resolution": "PT60M",
            "domain": "CTY|DE",
            "folder": "load",
        },
        {
            "name": "da_wind",
            "document_type": "A69",
            "unit": "MW",
            "resolution": "PT60M",
            "domain": "CTY|DE",
            "psr_types": ["B18", "B19"],
            "folder": "wind",
        },
        {
            "name": "da_solar",
            "document_type": "A69",
            "unit": "MW",
            "resolution": "PT60M",
            "domain": "CTY|DE",
            "psr_types": ["B16"],
            "folder": "solar",
        },
        {
            "name": "nuclear_avail_fr",
            "document_type": "A80",
            "unit": "MW",
            "resolution": "PT60M",
            "domain": "CTY|FR",
            "folder": "nuclear",
        },
        {
            "name": "net_exports",
            "document_type": "A11",
            "unit": "MW",
            "resolution": "PT60M",
            "domain": "cross_border",
            "folder": "flows",
        },
        {
            "name": "actual_generation",
            "document_type": "A75",
            "unit": "MW",
            "resolution": "PT60M",
            "domain": "CTY|DE",
            "folder": "fuels",
        },
    ],
    "source": "fallback",
}


class LLMConfigGenerator:
    """
    Convert ENTSO-E field docs into structured ingestion config via Gemini.

    Purpose:
        Produce column/unit expectations for the ingestion layer.

    Inputs:
        Path to entsoe_field_docs.txt; Gemini API key.

    Outputs:
        Dict with `fields` list of ingestion field definitions.

    Side Effects:
        Network call to Gemini; appends audit log JSONL.
    """

    def __init__(self, docs_path: Optional[Path] = None) -> None:
        """
        Args:
            docs_path: Optional override for field documentation path.
        """
        self.settings = get_settings()
        self.docs_path = docs_path or self.settings.entsoe_field_docs
        self.log_path = self.settings.logs / "llm_config_generation.jsonl"
        self._model = None

    def _configure_model(self) -> bool:
        """
        Configure google.generativeai client.

        Returns:
            True if model is ready; False if key missing (use fallback).
        """
        if self.settings.gemini_key_is_placeholder():
            logger.warning("GEMINI_API_KEY placeholder — using fallback ingestion config")
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
            os.environ["GEMINI_API_KEY"] = self.settings.gemini_api_key
            return True
        except Exception as exc:
            logger.error("Failed to configure Gemini: %s — using fallback", exc)
            return False

    def generate(self) -> Dict[str, Any]:
        """
        Read field docs and produce structured ingestion config.

        Returns:
            Config dict with `fields` list.

        Raises:
            FileNotFoundError: If docs file is missing and no fallback desired.
            Example:
                >>> cfg = LLMConfigGenerator().generate()
                >>> assert "fields" in cfg
        """
        if not self.docs_path.exists():
            logger.warning("Field docs missing at %s — using fallback config", self.docs_path)
            return FALLBACK_CONFIG

        docs_text = self.docs_path.read_text(encoding="utf-8")
        prompt = self._build_prompt(docs_text)

        if not self._configure_model():
            record = {
                "timestamp": utc_now_iso(),
                "model": GEMINI_MODEL,
                "temperature": 0.1,
                "prompt_chars": len(prompt),
                "response_chars": 0,
                "parse_success": True,
                "source": "fallback",
                "raw_response": json.dumps(FALLBACK_CONFIG),
                "parsed_config": FALLBACK_CONFIG,
                "generation_time_ms": 0,
            }
            append_jsonl(self.log_path, record)
            out = self.settings.data_processed / "ingestion_config.json"
            write_json(out, FALLBACK_CONFIG)
            return FALLBACK_CONFIG

        t0 = time.perf_counter()
        try:
            response = self._model.generate_content(prompt)
            raw = response.text or ""
            parsed = json.loads(raw)
            if "fields" not in parsed:
                parsed = {"fields": parsed if isinstance(parsed, list) else FALLBACK_CONFIG["fields"]}
            parse_success = True
        except Exception as exc:
            logger.error("LLM config generation failed: %s", exc)
            raw = str(exc)
            parsed = FALLBACK_CONFIG
            parse_success = False

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        record = {
            "timestamp": utc_now_iso(),
            "model": GEMINI_MODEL,
            "temperature": 0.1,
            "prompt_chars": len(prompt),
            "response_chars": len(raw),
            "parse_success": parse_success,
            "raw_response": raw[:50_000],
            "parsed_config": parsed,
            "generation_time_ms": elapsed_ms,
        }
        append_jsonl(self.log_path, record)
        out = self.settings.data_processed / "ingestion_config.json"
        write_json(out, parsed)
        logger.info(
            "LLM config generation complete — %s fields | %s ms | parse=%s",
            len(parsed.get("fields", [])),
            elapsed_ms,
            parse_success,
        )
        return parsed

    @staticmethod
    def _build_prompt(docs_text: str) -> str:
        """Build the Gemini prompt for field → JSON conversion."""
        return f"""You are a senior data engineer specialising in ENTSO-E Transparency Platform data.

Convert the following ENTSO-E field documentation into a structured JSON ingestion configuration.

DOCUMENTATION:
{docs_text}

Return ONLY valid JSON with this schema:
{{
  "fields": [
    {{
      "name": "column_name",
      "document_type": "Axx",
      "unit": "EUR/MWh or MW",
      "resolution": "PT60M",
      "domain": "BZN|DE-LU or CTY|DE",
      "psr_types": ["B18"],
      "folder": "prices|load|wind|solar|nuclear|flows|fuels",
      "notes": "optional"
    }}
  ]
}}
"""
