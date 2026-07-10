"""
Cobblestone Power — global configuration.

Purpose:
    Centralise all environment variables, ENTSO-E codes, filesystem paths,
    and model hyperparameters for the German day-ahead forecasting pipeline.

Inputs:
    Environment variables / .env file (via pydantic-settings).

Outputs:
    Validated Settings instance; path helpers; typed ENTSO-E constants.

Side Effects:
    Creates output directories on validate_all(); emits warnings for missing keys.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root (cobblestone-power/)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# ENTSO-E document type codes
# ---------------------------------------------------------------------------
DOC_DAY_AHEAD_PRICES: str = "A44"  # Day-ahead Prices [12.1.D]
DOC_LOAD_FORECAST: str = "A65"  # Day-ahead Total Load Forecast [6.1.B]
DOC_WIND_SOLAR_FORECAST: str = "A69"  # Day-ahead Generation Forecasts Wind/Solar [14.1.C]
DOC_UNAVAILABILITY: str = "A80"  # Unavailability of Production Units [15.1.A]
DOC_SCHEDULED_EXCHANGES: str = "A09"  # Scheduled Commercial Exchanges
DOC_PHYSICAL_FLOWS: str = "A11"  # Cross-Border Physical Flows [12.1.G]
DOC_ACTUAL_GENERATION: str = "A75"  # Actual Generation per Production Type [16.1.B&C]

# ---------------------------------------------------------------------------
# Bidding zones / control areas
# ---------------------------------------------------------------------------
DE_LU: str = "10Y1001A1001A82H"  # BZN|DE-LU post 2018-10-01
DE_AT_LU: str = "10Y1001A1001A63L"  # BZN|DE-AT-LU pre 2018-10-01
CTY_DE: str = "10Y1001A1001A83F"  # Country Germany
CTY_FR: str = "10YFR-RTE------C"  # Country France

ZONE_SPLIT_DATE: date = date(2018, 10, 1)
DE_NUCLEAR_PHASEOUT_DATE: date = date(2023, 4, 15)
UKRAINE_WAR_DATE: date = date(2022, 2, 24)
EPEX_OUTAGE_DATE: date = date(2024, 6, 26)

# Neighbour bidding zones for cross-border flows
NEIGHBOUR_ZONES: Dict[str, str] = {
    "FR": "10YFR-RTE------C",
    "NL": "10YNL----------L",
    "AT": "10YAT-APG------L",
    "CH": "10YCH-SWISSGRIDZ",
    "CZ": "10YCZ-CEPS-----N",
    "DK1": "10YDK-1--------W",
    "DK2": "10YDK-2--------M",
    "PL": "10YPL-AREA-----S",
}

# PSR types for generation
PSR_WIND_ONSHORE: str = "B18"
PSR_WIND_OFFSHORE: str = "B19"
PSR_SOLAR: str = "B16"
PSR_NUCLEAR: str = "B14"
PSR_GAS: str = "B04"
PSR_HARD_COAL: str = "B05"
PSR_LIGNITE: str = "B06"
PSR_HYDRO: str = "B11"
PSR_BIOMASS: str = "B01"
PSR_OTHER: str = "B20"

# Physical constants
GERMANY_LAT_DEG: float = 51.2
GERMANY_LON_DEG: float = 10.5
PRICE_FLOOR_EUR_MWH: float = -500.0
PRICE_CEILING_EUR_MWH: float = 3000.0
LOAD_FLOOR_MW: float = 20_000.0
LOAD_CEILING_MW: float = 100_000.0
THERMAL_BASELOAD_MW: float = 25_000.0
WIND_ONSHORE_CAPACITY_MW: float = 68_000.0
WIND_OFFSHORE_CAPACITY_MW: float = 8_000.0
HOUR_SPIKE_THRESHOLD_EUR: float = 500.0
RANDOM_SEED: int = 42
PIPELINE_VERSION: str = "1.0.0"
ENTSOE_PLACEHOLDER_KEY: str = "placeholder_add_when_received"
GEMINI_MODEL: str = "gemini-2.0-flash"
FIGURE_DPI: int = 300


@dataclass
class XGBoostHyperparameters:
    """Default XGBoost hyperparameters for point forecasting (EUR/MWh)."""

    n_estimators: int = 1200
    max_depth: int = 6
    learning_rate: float = 0.04
    subsample: float = 0.80
    colsample_bytree: float = 0.75
    min_child_weight: int = 6
    reg_alpha: float = 0.15
    reg_lambda: float = 1.2
    objective: str = "reg:squarederror"
    tree_method: str = "hist"
    early_stopping_rounds: int = 60
    eval_metric: List[str] = field(default_factory=lambda: ["rmse", "mae"])
    random_state: int = RANDOM_SEED

    def to_dict(self) -> Dict[str, Any]:
        """Return hyperparameters as a plain dict for XGBRegressor."""
        return asdict(self)


class Settings(BaseSettings):
    """
    Application settings loaded from environment / .env.

    Purpose:
        Validate and expose all runtime configuration for the pipeline.

    Inputs:
        ENTSOE_API_KEY, GEMINI_API_KEY, date range, market codes, log level.

    Outputs:
        Typed settings with pathlib Path attributes for all data/output dirs.

    Side Effects:
        Reads .env from project root; may emit warnings on validate_all().
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    entsoe_api_key: str = Field(default=ENTSOE_PLACEHOLDER_KEY, alias="ENTSOE_API_KEY")
    gemini_api_key: str = Field(default="your_gemini_api_key_here", alias="GEMINI_API_KEY")
    start_date: date = Field(default=date(2022, 1, 1), alias="START_DATE")
    end_date: date = Field(default=date(2024, 12, 31), alias="END_DATE")
    test_start: date = Field(default=date(2024, 1, 1), alias="TEST_START")
    market: str = Field(default="DE_LU", alias="MARKET")
    bidding_zone: str = Field(default=DE_LU, alias="BIDDING_ZONE")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    dashboard_port: int = Field(default=8501, alias="DASHBOARD_PORT")

    # Paths
    project_root: Path = PROJECT_ROOT
    data_raw: Path = PROJECT_ROOT / "data" / "raw"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    outputs: Path = PROJECT_ROOT / "outputs"
    figures: Path = PROJECT_ROOT / "outputs" / "figures"
    qa_report: Path = PROJECT_ROOT / "outputs" / "qa_report"
    logs: Path = PROJECT_ROOT / "outputs" / "logs"
    models_dir: Path = PROJECT_ROOT / "outputs" / "models"
    forecasts_dir: Path = PROJECT_ROOT / "outputs" / "forecasts"
    walk_forward_splits: Path = PROJECT_ROOT / "data" / "processed" / "walk_forward_splits"
    master_dataset: Path = PROJECT_ROOT / "data" / "processed" / "master_dataset.parquet"
    features_path: Path = PROJECT_ROOT / "data" / "processed" / "features.parquet"
    features_manifest: Path = PROJECT_ROOT / "data" / "processed" / "features_manifest.json"
    submission_csv: Path = PROJECT_ROOT / "submission.csv"
    entsoe_field_docs: Path = PROJECT_ROOT / "data" / "raw" / "entsoe_field_docs.txt"

    xgb_params: XGBoostHyperparameters = Field(default_factory=XGBoostHyperparameters)

    @field_validator("start_date", "end_date", "test_start", mode="before")
    @classmethod
    def parse_date(cls, v: Any) -> Any:
        """Parse ISO date strings from env into date objects."""
        if isinstance(v, str):
            return date.fromisoformat(v)
        return v

    def entsoe_key_is_placeholder(self) -> bool:
        """Return True if ENTSO-E API key is unset or still the placeholder."""
        key = (self.entsoe_api_key or "").strip()
        return (
            not key
            or key == ENTSOE_PLACEHOLDER_KEY
            or key.startswith("placeholder")
        )

    def gemini_key_is_placeholder(self) -> bool:
        """Return True if Gemini API key is unset or still the placeholder."""
        key = (self.gemini_api_key or "").strip()
        return (
            not key
            or key == "your_gemini_api_key_here"
            or key.startswith("your_")
        )

    def bidding_zone_for_date(self, d: date) -> str:
        """
        Return the correct DE bidding zone EIC for a calendar date.

        Args:
            d: Calendar date (timezone-naive date is fine).

        Returns:
            DE_AT_LU before 2018-10-01, else DE_LU.
        """
        return DE_AT_LU if d < ZONE_SPLIT_DATE else DE_LU

    def validate_all(self) -> bool:
        """
        Validate configuration and ensure directories exist.

        Returns:
            True if configuration is usable (pipeline may still skip ingestion).

        Side Effects:
            Creates data/output directories; logs warnings for missing API keys.
        """
        for path in [
            self.data_raw,
            self.data_raw / "prices",
            self.data_raw / "load",
            self.data_raw / "wind",
            self.data_raw / "solar",
            self.data_raw / "nuclear",
            self.data_raw / "flows",
            self.data_raw / "fuels",
            self.data_processed,
            self.walk_forward_splits,
            self.outputs,
            self.figures / "eda",
            self.figures / "validation",
            self.figures / "forecasts",
            self.figures / "regime",
            self.qa_report,
            self.logs,
            self.models_dir / "xgboost_quantiles",
            self.forecasts_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        if self.entsoe_key_is_placeholder():
            msg = (
                "ENTSOE_API_KEY is placeholder — skipping ingestion. "
                "Add your key to .env when received from transparency.entsoe.eu. "
                "All other pipeline steps will run normally."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
            logger.warning(msg)

        if self.gemini_key_is_placeholder():
            msg = (
                "GEMINI_API_KEY is placeholder — LLM modules will use fallback rules. "
                "Get a free key at aistudio.google.com."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
            logger.warning(msg)

        if self.start_date >= self.end_date:
            raise ValueError(f"START_DATE ({self.start_date}) must be before END_DATE ({self.end_date})")

        if self.test_start < self.start_date or self.test_start > self.end_date:
            raise ValueError(
                f"TEST_START ({self.test_start}) must fall within "
                f"[{self.start_date}, {self.end_date}]"
            )

        logger.info("Configuration validated | version=%s", PIPELINE_VERSION)
        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise settings for pipeline-start logging (secrets redacted).

        Returns:
            Dict of config values with API keys masked.
        """
        return {
            "pipeline_version": PIPELINE_VERSION,
            "start_date": str(self.start_date),
            "end_date": str(self.end_date),
            "test_start": str(self.test_start),
            "market": self.market,
            "bidding_zone": self.bidding_zone,
            "log_level": self.log_level,
            "dashboard_port": self.dashboard_port,
            "entsoe_api_key_set": not self.entsoe_key_is_placeholder(),
            "gemini_api_key_set": not self.gemini_key_is_placeholder(),
            "gemini_model": GEMINI_MODEL,
            "project_root": str(self.project_root),
            "xgb_params": self.xgb_params.to_dict(),
        }


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Return a process-wide Settings singleton.

    Returns:
        Cached Settings instance.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
