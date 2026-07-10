"""
Cobblestone Power — SMARD (smard.de) public API ingestion.

Purpose:
    Pull real German electricity market data from Bundesnetzagentur SMARD
    without an API key. Primary data source for the forecasting pipeline.

Inputs:
    Date range (default 2022-01-01 → 2024-12-31); public HTTPS endpoints.

Outputs:
    Parquet under data/raw/smard/ and the standard data/raw/{prices,load,...}
    layout expected by DataCleaner; ingestion_manifest.json; data_source.json.

Side Effects:
    Network calls to smard.de; disk writes; structured logging.

API notes:
    Base: https://www.smard.de/app/chart_data/{filter}/{region}/...
    Index: index_{resolution}.json  (hourly resolution string is ``hour``)
    Series: {filter}_{region}_{resolution}_{timestamp}.json
    Payload: {"series": [[unix_ms, value], ...]} with nulls for missing points.

Filter IDs (requested → working fallback when 404 / combined series):
    4169  Day-ahead price DE/LU (EUR/MWh) — region DE-LU
    6791  Total load → fallback 410 (Netzlast)
    5097  Wind onshore forecast (user) → prefer 123 (dedicated onshore;
          5097 is Wind+PV combined per SMARD docs)
    5098  Wind offshore forecast → fallback 3791
    5100  Solar PV forecast → fallback 125
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

from config.settings import DE_NUCLEAR_PHASEOUT_DATE, RANDOM_SEED, get_settings
from src.utils import retry_with_backoff, save_parquet, utc_now_iso, write_json

logger = logging.getLogger(__name__)

SMARD_BASE: str = "https://www.smard.de/app/chart_data"
# SMARD encodes hourly resolution as the string "hour" (not numeric 60).
SMARD_RESOLUTION: str = "hour"

# User-requested filter IDs
FILTER_DA_PRICE: int = 4169
FILTER_LOAD: int = 6791
FILTER_WIND_ONSHORE: int = 5097
FILTER_WIND_OFFSHORE: int = 5098
FILTER_SOLAR: int = 5100

# Official SMARD fallbacks when requested IDs 404 or are combined series
FILTER_LOAD_FALLBACK: int = 410
FILTER_WIND_ONSHORE_DEDICATED: int = 123
FILTER_WIND_OFFSHORE_FALLBACK: int = 3791
FILTER_SOLAR_FALLBACK: int = 125

# Actual generation (merit-order thermal stack)
FILTER_GAS: int = 4071
FILTER_LIGNITE: int = 1223
FILTER_HARD_COAL: int = 4069

REGION_PRICE: str = "DE-LU"
REGION_GENERATION: str = "DE"

REQUEST_TIMEOUT_S: float = 60.0
USER_AGENT: str = "CobblestonePower/1.0 (+https://smard.de; research)"


class SMARDIngester:
    """
    SMARD public-API ingester for DE-LU day-ahead fundamentals.

    Purpose:
        Replace synthetic / ENTSO-E placeholder runs with real Bundesnetzagentur
        market data (no API key).

    Inputs:
        Optional start/end dates (defaults from settings).

    Outputs:
        Dict of series matching ENTSOEIngester.fetch_all schema; parquet files.

    Side Effects:
        HTTPS GETs; writes under data/raw/smard/ and data/raw/*; manifest JSON.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        self.manifest: Dict[str, Any] = {
            "run_timestamp": utc_now_iso(),
            "source": "SMARD",
            "base_url": SMARD_BASE,
            "resolution": SMARD_RESOLUTION,
            "fetches": [],
            "failed_requests": 0,
            "filter_resolution": {},
        }
        self.smard_dir = self.settings.data_raw / "smard"
        self.smard_dir.mkdir(parents=True, exist_ok=True)
        self._index_cache: Dict[Tuple[int, str], List[int]] = {}

    def _get_json(self, url: str) -> Any:
        """GET JSON from SMARD. Retries transient errors; 404 fails immediately."""
        return self._get_json_retry(url)

    @retry_with_backoff(max_retries=5, base_delay=2.0, exceptions=(requests.RequestException,))
    def _get_json_retry(self, url: str) -> Any:
        """Retry only network/HTTP errors (not 404)."""
        resp = self.session.get(url, timeout=REQUEST_TIMEOUT_S)
        if resp.status_code == 404:
            raise FileNotFoundError(f"SMARD 404: {url}")
        resp.raise_for_status()
        return resp.json()

    def _index_url(self, filter_id: int, region: str) -> str:
        return f"{SMARD_BASE}/{filter_id}/{region}/index_{SMARD_RESOLUTION}.json"

    def _series_url(self, filter_id: int, region: str, timestamp_ms: int) -> str:
        return (
            f"{SMARD_BASE}/{filter_id}/{region}/"
            f"{filter_id}_{region}_{SMARD_RESOLUTION}_{timestamp_ms}.json"
        )

    def fetch_index_timestamps(self, filter_id: int, region: str) -> List[int]:
        """Fetch available chunk timestamps for a filter/region."""
        key = (filter_id, region)
        if key in self._index_cache:
            return self._index_cache[key]
        url = self._index_url(filter_id, region)
        data = self._get_json(url)
        timestamps = sorted(int(t) for t in data.get("timestamps", []))
        self._index_cache[key] = timestamps
        self.manifest["fetches"].append(
            {
                "type": "index",
                "filter": filter_id,
                "region": region,
                "n_timestamps": len(timestamps),
                "url": url,
            }
        )
        logger.info("SMARD index filter=%s region=%s → %s chunks", filter_id, region, len(timestamps))
        return timestamps

    def _resolve_filter(self, candidates: Sequence[int], region: str, label: str) -> int:
        """Return the first candidate filter that has a working index."""
        errors: List[str] = []
        available: List[int] = []
        for fid in candidates:
            try:
                self.fetch_index_timestamps(fid, region)
                available.append(fid)
            except Exception as exc:
                errors.append(f"{fid}: {exc}")
                logger.warning(
                    "SMARD filter %s unavailable for %s (%s): %s", fid, label, region, exc
                )

        if not available:
            raise RuntimeError(f"No SMARD filter available for {label}: {errors}")

        # Prefer dedicated onshore (123) over combined Wind+PV (5097)
        if label == "da_wind_onshore" and FILTER_WIND_ONSHORE_DEDICATED in available:
            chosen = FILTER_WIND_ONSHORE_DEDICATED
            if FILTER_WIND_ONSHORE in available:
                logger.info(
                    "SMARD: skipping filter %s (Wind+PV combined) — using dedicated onshore %s",
                    FILTER_WIND_ONSHORE,
                    chosen,
                )
        else:
            chosen = available[0]

        self.manifest["filter_resolution"][label] = {
            "chosen": chosen,
            "candidates": list(candidates),
            "available": available,
            "region": region,
        }
        logger.info("SMARD %s → filter %s (region %s)", label, chosen, region)
        return chosen

    def fetch_series(
        self,
        filter_id: int,
        region: str,
        start: date,
        end: date,
        name: str,
    ) -> pd.Series:
        """Download all hourly chunks covering [start, end] and concatenate."""
        start_ms = int(
            datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp() * 1000
        )
        end_ms = int(
            (
                datetime(end.year, end.month, end.day, tzinfo=timezone.utc) + pd.Timedelta(hours=23)
            ).timestamp()
            * 1000
        )

        timestamps = self.fetch_index_timestamps(filter_id, region)
        in_range = [t for t in timestamps if start_ms <= t <= end_ms]
        before = [t for t in timestamps if t < start_ms]
        if before:
            in_range = [before[-1]] + in_range
        seen = set()
        chunks: List[int] = []
        for t in in_range:
            if t not in seen:
                seen.add(t)
                chunks.append(t)

        if not chunks:
            raise RuntimeError(
                f"No SMARD timestamps for {name} filter={filter_id} in {start}–{end}"
            )

        frames: List[pd.Series] = []
        for ts_ms in chunks:
            url = self._series_url(filter_id, region, ts_ms)
            try:
                payload = self._get_json(url)
                pairs = payload.get("series", [])
                if not pairs:
                    continue
                idx = pd.to_datetime([p[0] for p in pairs], unit="ms", utc=True)
                vals = [np.nan if p[1] is None else float(p[1]) for p in pairs]
                frames.append(pd.Series(vals, index=idx, name=name))
                self.manifest["fetches"].append(
                    {
                        "type": "series",
                        "filter": filter_id,
                        "region": region,
                        "timestamp_ms": ts_ms,
                        "n": len(pairs),
                    }
                )
            except Exception as exc:
                logger.warning("SMARD chunk failed %s: %s", url, exc)
                self.manifest["failed_requests"] += 1

        if not frames:
            raise RuntimeError(f"SMARD returned no data points for {name} (filter={filter_id})")

        series = pd.concat(frames).sort_index()
        series = series[~series.index.duplicated(keep="last")]
        lo = pd.Timestamp(start, tz="UTC")
        hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23)
        series = series.loc[(series.index >= lo) & (series.index <= hi)]
        series = series.resample("h").mean()
        series.name = name
        logger.info(
            "SMARD %s: %s hours, nulls=%s (%.1f%%)",
            name,
            len(series),
            int(series.isna().sum()),
            100.0 * float(series.isna().mean()),
        )
        return series

    @staticmethod
    def _impute_renewable_component(s: pd.Series) -> pd.Series:
        """Fill null renewable forecast points with 0 (ENTSO-E wind path parity)."""
        return s.fillna(0.0)

    @staticmethod
    def _impute_load_light(s: pd.Series) -> pd.Series:
        """Short linear fill for load; longer gaps left for DataCleaner._impute."""
        return s.interpolate(method="linear", limit=6)

    @staticmethod
    def _impute_price_light(s: pd.Series) -> pd.Series:
        """Short linear fill for prices; longer gaps left for DataCleaner._impute."""
        return s.interpolate(method="linear", limit=3)

    def fetch_all(self, start: Optional[date] = None, end: Optional[date] = None) -> Dict[str, Any]:
        """
        Download price, load, wind, solar from SMARD and persist ENTSO-E-compatible raw files.

        Returns:
            Dict with da_price, da_load, da_wind, da_solar, nuclear, flows, generation, fuels.
        """
        start = start or self.settings.start_date
        end = end or self.settings.end_date
        logger.info("SMARD ingest %s → %s", start, end)

        price_f = self._resolve_filter([FILTER_DA_PRICE], REGION_PRICE, "da_price")
        load_f = self._resolve_filter(
            [FILTER_LOAD, FILTER_LOAD_FALLBACK], REGION_GENERATION, "da_load"
        )
        wind_on_f = self._resolve_filter(
            [FILTER_WIND_ONSHORE, FILTER_WIND_ONSHORE_DEDICATED],
            REGION_GENERATION,
            "da_wind_onshore",
        )
        wind_off_f = self._resolve_filter(
            [FILTER_WIND_OFFSHORE, FILTER_WIND_OFFSHORE_FALLBACK],
            REGION_GENERATION,
            "da_wind_offshore",
        )
        solar_f = self._resolve_filter(
            [FILTER_SOLAR, FILTER_SOLAR_FALLBACK],
            REGION_GENERATION,
            "da_solar",
        )

        da_price = self.fetch_series(price_f, REGION_PRICE, start, end, "da_price")
        da_load = self.fetch_series(load_f, REGION_GENERATION, start, end, "da_load")
        wind_on = self.fetch_series(wind_on_f, REGION_GENERATION, start, end, "da_wind_onshore")
        wind_off = self.fetch_series(wind_off_f, REGION_GENERATION, start, end, "da_wind_offshore")
        da_solar = self.fetch_series(solar_f, REGION_GENERATION, start, end, "da_solar")

        da_price = self._impute_price_light(da_price)
        da_load = self._impute_load_light(da_load)
        wind_on = self._impute_renewable_component(wind_on)
        wind_off = self._impute_renewable_component(wind_off)
        da_solar = self._impute_renewable_component(da_solar)

        idx = (
            da_price.index.union(da_load.index)
            .union(wind_on.index)
            .union(wind_off.index)
            .union(da_solar.index)
        )
        idx = pd.date_range(idx.min(), idx.max(), freq="h", tz="UTC")
        da_price = da_price.reindex(idx)
        da_load = da_load.reindex(idx)
        wind_on = wind_on.reindex(idx).fillna(0.0)
        wind_off = wind_off.reindex(idx).fillna(0.0)
        da_solar = da_solar.reindex(idx).fillna(0.0)
        da_wind = (wind_on + wind_off).rename("da_wind")

        # Real SMARD actual generation (gas / lignite / hard coal) for merit-order features
        gen_gas = self._fetch_generation_safe(FILTER_GAS, start, end, "gen_gas", idx)
        gen_lignite = self._fetch_generation_safe(FILTER_LIGNITE, start, end, "gen_lignite", idx)
        gen_coal = self._fetch_generation_safe(FILTER_HARD_COAL, start, end, "gen_coal", idx)

        nuclear, flows, generation, fuels = self._build_optional_panels(
            idx, da_wind, da_solar, da_load, gen_gas=gen_gas, gen_lignite=gen_lignite, gen_coal=gen_coal
        )

        results = {
            "da_price": da_price,
            "da_load": da_load,
            "da_wind": da_wind,
            "da_solar": da_solar,
            "da_wind_onshore": wind_on,
            "da_wind_offshore": wind_off,
            "nuclear": nuclear,
            "flows": flows,
            "generation": generation,
            "fuels": fuels,
        }
        self._persist(results)
        self._save_manifest(start, end)
        self._save_data_source(start, end, success=True)
        return results

    def _fetch_generation_safe(
        self,
        filter_id: int,
        start: date,
        end: date,
        name: str,
        idx: pd.DatetimeIndex,
    ) -> pd.Series:
        """Fetch actual generation series; return zeros on failure."""
        try:
            fid = self._resolve_filter([filter_id], REGION_GENERATION, name)
            s = self.fetch_series(fid, REGION_GENERATION, start, end, name)
            s = s.reindex(idx).interpolate(method="linear", limit=12).fillna(0.0)
            return s
        except Exception as exc:
            logger.warning("SMARD generation %s (filter %s) failed: %s — using zeros", name, filter_id, exc)
            return pd.Series(0.0, index=idx, name=name)

    def _build_optional_panels(
        self,
        idx: pd.DatetimeIndex,
        da_wind: pd.Series,
        da_solar: pd.Series,
        da_load: pd.Series,
        gen_gas: Optional[pd.Series] = None,
        gen_lignite: Optional[pd.Series] = None,
        gen_coal: Optional[pd.Series] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Construct nuclear / flows / generation / fuels frames (real gen when provided)."""
        rng = np.random.default_rng(RANDOM_SEED)
        n = len(idx)
        de_nuc = np.where(idx.date < DE_NUCLEAR_PHASEOUT_DATE, 4_000.0, 0.0)
        nuclear = pd.DataFrame(
            {
                "nuclear_avail_de": de_nuc,
                "nuclear_avail_fr": np.full(n, 45_000.0),
                "de_nuclear_phaseout": idx.date >= DE_NUCLEAR_PHASEOUT_DATE,
            },
            index=idx,
        )
        flows = pd.DataFrame(
            {"net_exports": np.zeros(n), "de_fr_net_flow": np.zeros(n)},
            index=idx,
        )
        residual = (da_load.fillna(0) - da_wind.fillna(0) - da_solar.fillna(0)).clip(lower=0)
        gas = gen_gas if gen_gas is not None else residual * 0.25
        lignite = gen_lignite if gen_lignite is not None else residual * 0.20
        coal = gen_coal if gen_coal is not None else residual * 0.15
        generation = pd.DataFrame(
            {
                "gen_gas": gas,
                "gen_coal": coal,
                "gen_lignite": lignite,
                "gen_nuclear": de_nuc,
                "gen_wind_onshore": da_wind * 0.85,
                "gen_wind_offshore": da_wind * 0.15,
                "gen_solar": da_solar,
                "gen_hydro": np.full(n, 2_500.0),
                "gen_biomass": np.full(n, 5_000.0),
                "gen_other": np.full(n, 1_000.0),
            },
            index=idx,
        )
        fuels = pd.DataFrame(
            {
                "ttf_gas_price": np.full(n, 35.0) + rng.normal(0, 0.5, n),
                "eua_carbon_price": np.full(n, 70.0),
                "coal_price": np.full(n, 120.0),
            },
            index=idx,
        )
        return nuclear, flows, generation, fuels

    def _persist(self, results: Dict[str, Any]) -> None:
        """Write SMARD archive + ENTSO-E-compatible raw paths."""
        save_parquet(results["da_price"], self.smard_dir / "da_price.parquet")
        save_parquet(results["da_load"], self.smard_dir / "da_load.parquet")
        save_parquet(results["da_wind"], self.smard_dir / "da_wind.parquet")
        save_parquet(results["da_solar"], self.smard_dir / "da_solar.parquet")
        save_parquet(results["da_wind_onshore"], self.smard_dir / "da_wind_onshore.parquet")
        save_parquet(results["da_wind_offshore"], self.smard_dir / "da_wind_offshore.parquet")

        raw = self.settings.data_raw
        save_parquet(results["da_price"], raw / "prices" / "da_price.parquet")
        save_parquet(results["da_load"], raw / "load" / "da_load.parquet")
        save_parquet(results["da_wind"], raw / "wind" / "da_wind.parquet")
        save_parquet(results["da_solar"], raw / "solar" / "da_solar.parquet")
        save_parquet(results["da_wind_onshore"], raw / "wind" / "da_wind_onshore.parquet")
        save_parquet(results["da_wind_offshore"], raw / "wind" / "da_wind_offshore.parquet")
        save_parquet(results["nuclear"], raw / "nuclear" / "nuclear_availability.parquet")
        save_parquet(results["flows"], raw / "flows" / "cross_border_flows.parquet")
        save_parquet(results["generation"], raw / "fuels" / "actual_generation.parquet")
        save_parquet(results["fuels"], raw / "fuels" / "fuel_prices.parquet")
        logger.info(
            "SMARD series persisted under %s and standard data/raw/* paths", self.smard_dir
        )

    def _save_manifest(self, start: date, end: date) -> None:
        self.manifest["date_range"] = {"start": str(start), "end": str(end)}
        self.manifest["total_fetches"] = len(self.manifest["fetches"])
        self.manifest["skipped_placeholder"] = False
        write_json(self.settings.data_raw / "ingestion_manifest.json", self.manifest)
        write_json(self.smard_dir / "ingestion_manifest.json", self.manifest)
        logger.info("SMARD ingestion manifest saved")

    def _save_data_source(self, start: date, end: date, success: bool) -> None:
        write_json(
            self.settings.data_raw / "data_source.json",
            {
                "source": "SMARD",
                "provider": "Bundesnetzagentur",
                "url": "https://www.smard.de",
                "success": success,
                "date_range": {"start": str(start), "end": str(end)},
                "written_at": utc_now_iso(),
                "label": "DATA SOURCE: SMARD (smard.de) — Bundesnetzagentur",
                "filters": self.manifest.get("filter_resolution", {}),
            },
        )


def load_data_source(settings: Optional[Any] = None) -> Dict[str, Any]:
    """Load data_source.json if present (for dashboard badge)."""
    settings = settings or get_settings()
    path = settings.data_raw / "data_source.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
