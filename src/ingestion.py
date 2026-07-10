"""
Cobblestone Power — ENTSO-E Transparency Platform data ingestion.

Purpose:
    Fetch German day-ahead prices, load, renewables, nuclear availability,
    cross-border flows, and actual generation; persist raw parquet files.

Inputs:
    Date range; ENTSOE_API_KEY (skips gracefully if placeholder).

Outputs:
    Raw parquet series under data/raw/*; ingestion_manifest.json.

Side Effects:
    Network calls to ENTSO-E; writes parquet + manifest; may synthesise
    demo data when API key is unavailable so downstream steps can run.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import (
    CTY_DE,
    CTY_FR,
    DE_AT_LU,
    DE_LU,
    DE_NUCLEAR_PHASEOUT_DATE,
    EPEX_OUTAGE_DATE,
    NEIGHBOUR_ZONES,
    PSR_BIOMASS,
    PSR_GAS,
    PSR_HARD_COAL,
    PSR_HYDRO,
    PSR_LIGNITE,
    PSR_NUCLEAR,
    PSR_OTHER,
    PSR_SOLAR,
    PSR_WIND_OFFSHORE,
    PSR_WIND_ONSHORE,
    RANDOM_SEED,
    ZONE_SPLIT_DATE,
    get_settings,
)
from src.utils import retry_with_backoff, save_parquet, utc_now_iso, write_json

logger = logging.getLogger(__name__)

# Installed capacity proxies (MW) for nuclear availability
FR_NUCLEAR_INSTALLED_MW: float = 61_000.0
DE_NUCLEAR_INSTALLED_PRE_PHASEOUT_MW: float = 8_000.0


class ENTSOEIngester:
    """
    Rate-limit-aware ENTSO-E data fetcher for the DE-LU market.

    Purpose:
        Ingest all fundamental series required for fair-value forecasting.

    Inputs:
        start/end dates; Settings (API key, paths).

    Outputs:
        Hourly Series/DataFrames in UTC; parquet under data/raw/.

    Side Effects:
        HTTP requests via entsoe-py; disk writes; structured logging.
    """

    def __init__(self) -> None:
        """Initialise client; defer API construction until first real fetch."""
        self.settings = get_settings()
        self._client: Any = None
        self.manifest: Dict[str, Any] = {
            "run_timestamp": utc_now_iso(),
            "fetches": [],
            "failed_requests": 0,
            "skipped_placeholder": False,
        }

    def _api_available(self) -> bool:
        """Return False if ENTSOE key is placeholder — skip ingestion."""
        if self.settings.entsoe_key_is_placeholder():
            logger.warning(
                "ENTSOE_API_KEY is placeholder — skipping live ingestion. "
                "Add your key to .env when received from transparency.entsoe.eu"
            )
            self.manifest["skipped_placeholder"] = True
            return False
        return True

    def _get_client(self) -> Any:
        """
        Lazily construct EntsoePandasClient.

        Returns:
            Configured client instance.

        Raises:
            ImportError / RuntimeError if client cannot be created.
        """
        if self._client is None:
            from entsoe import EntsoePandasClient

            self._client = EntsoePandasClient(api_key=self.settings.entsoe_api_key)
        return self._client

    def _zone_for(self, d: date) -> str:
        """Return bidding zone EIC for calendar date (DE zone split)."""
        return DE_AT_LU if d < ZONE_SPLIT_DATE else DE_LU

    def _record_fetch(
        self,
        name: str,
        start: date,
        end: date,
        rows: int,
        path: Optional[Path],
        gaps: int = 0,
        notes: str = "",
    ) -> None:
        """Append a fetch record to the ingestion manifest."""
        self.manifest["fetches"].append(
            {
                "name": name,
                "start": str(start),
                "end": str(end),
                "rows": rows,
                "path": str(path) if path else None,
                "gaps": gaps,
                "notes": notes,
                "timestamp": utc_now_iso(),
            }
        )

    @retry_with_backoff(max_retries=5, base_delay=2.0)
    def _query(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Execute an ENTSO-E client method with exponential backoff.

        Args:
            fn: Bound client method.
            *args / **kwargs: Passed through.

        Returns:
            API response (typically Series/DataFrame).
        """
        logger.info("ENTSO-E request: %s args=%s kwargs_keys=%s", getattr(fn, "__name__", str(fn)), args[:2], list(kwargs.keys()))
        return fn(*args, **kwargs)

    def fetch_day_ahead_prices(self, start: date, end: date) -> Optional[pd.Series]:
        """
        Fetch Day-ahead Prices [12.1.D] (document type A44).

        Bidding zone: BZN|DE-LU post Oct 2018, BZN|DE-AT-LU pre Oct 2018.
        Handles the zone split automatically. Flags EPEX outage 2024-06-25/26.

        Args:
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            Hourly Series EUR/MWh with UTC index, or None if key placeholder.
        """
        if not self._api_available():
            return None

        client = self._get_client()
        pieces: List[pd.Series] = []

        # Split at zone boundary if range crosses it
        if start < ZONE_SPLIT_DATE <= end:
            ranges = [(start, ZONE_SPLIT_DATE - timedelta(days=1), DE_AT_LU), (ZONE_SPLIT_DATE, end, DE_LU)]
        elif end < ZONE_SPLIT_DATE:
            ranges = [(start, end, DE_AT_LU)]
        else:
            ranges = [(start, end, DE_LU)]

        for s, e, zone in ranges:
            ts_start = pd.Timestamp(s, tz="UTC")
            ts_end = pd.Timestamp(e, tz="UTC") + pd.Timedelta(days=1)
            raw = self._query(client.query_day_ahead_prices, zone, start=ts_start, end=ts_end)
            if isinstance(raw, pd.Series):
                pieces.append(raw)

        if not pieces:
            self.manifest["failed_requests"] += 1
            return None

        series = pd.concat(pieces).sort_index()
        series = series[~series.index.duplicated(keep="first")]
        if series.index.tz is None:
            series.index = series.index.tz_localize("UTC")
        else:
            series.index = series.index.tz_convert("UTC")
        series = series.resample("h").mean()
        series.name = "da_price"

        # Flag EPEX technical outage 25-26 June 2024 (€2,325/MWh spike)
        outage_mask = (series.index.date >= date(2024, 6, 25)) & (series.index.date <= EPEX_OUTAGE_DATE)
        if outage_mask.any():
            logger.warning(
                "EPEX technical outage 25-26 June 2024 detected — %s hours flagged",
                int(outage_mask.sum()),
            )

        path = self.settings.data_raw / "prices" / "da_price.parquet"
        save_parquet(series, path)
        self._record_fetch("da_price", start, end, len(series), path, notes="A44 Day-ahead Prices [12.1.D]")
        return series

    def fetch_load_forecast(self, start: date, end: date) -> Optional[pd.Series]:
        """
        Fetch Day-ahead Total Load Forecast [6.1.B] (A65) for CTY|DE.

        Aggregates 15-minute data to hourly mean when present.

        Args:
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            Hourly load Series in MW, or None if key placeholder.
        """
        if not self._api_available():
            return None

        client = self._get_client()
        ts_start = pd.Timestamp(start, tz="UTC")
        ts_end = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
        raw = self._query(client.query_load_forecast, CTY_DE, start=ts_start, end=ts_end)
        series = self._to_hourly_series(raw, "da_load")
        path = self.settings.data_raw / "load" / "da_load.parquet"
        save_parquet(series, path)
        self._record_fetch("da_load", start, end, len(series), path, notes="A65 Load Forecast [6.1.B]")
        return series

    def fetch_wind_forecast(self, start: date, end: date) -> Optional[pd.Series]:
        """
        Fetch Day-ahead Wind Generation Forecast [14.1.C] (A69).

        Fetches onshore (B18) and offshore (B19) separately, sums to total wind.
        Offshore sparse before 2020 — zero-imputed with logging.

        Args:
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            Hourly total wind Series in MW, or None if key placeholder.
        """
        if not self._api_available():
            return None

        client = self._get_client()
        ts_start = pd.Timestamp(start, tz="UTC")
        ts_end = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)

        onshore = self._query(
            client.query_wind_and_solar_forecast,
            CTY_DE,
            start=ts_start,
            end=ts_end,
            psr_type=PSR_WIND_ONSHORE,
        )
        offshore = self._query(
            client.query_wind_and_solar_forecast,
            CTY_DE,
            start=ts_start,
            end=ts_end,
            psr_type=PSR_WIND_OFFSHORE,
        )
        s_on = self._to_hourly_series(onshore, "da_wind_onshore").fillna(0.0)
        s_off = self._to_hourly_series(offshore, "da_wind_offshore")
        if s_off.isna().mean() > 0.5:
            logger.info("Offshore wind sparse — imputing missing with 0 (documented pre-2020 gap)")
            s_off = s_off.fillna(0.0)
        else:
            s_off = s_off.fillna(0.0)

        total = (s_on + s_off).rename("da_wind")
        path = self.settings.data_raw / "wind" / "da_wind.parquet"
        save_parquet(total, path)
        save_parquet(s_on, self.settings.data_raw / "wind" / "da_wind_onshore.parquet")
        save_parquet(s_off, self.settings.data_raw / "wind" / "da_wind_offshore.parquet")
        self._record_fetch("da_wind", start, end, len(total), path, notes="A69 Wind B18+B19 [14.1.C]")
        return total

    def fetch_solar_forecast(self, start: date, end: date) -> Optional[pd.Series]:
        """
        Fetch Day-ahead Solar Generation Forecast [14.1.C] (A69, PSR B16).

        Aggregates 15-minute resolution to hourly mean when present.

        Args:
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            Hourly solar Series in MW, or None if key placeholder.
        """
        if not self._api_available():
            return None

        client = self._get_client()
        ts_start = pd.Timestamp(start, tz="UTC")
        ts_end = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
        raw = self._query(
            client.query_wind_and_solar_forecast,
            CTY_DE,
            start=ts_start,
            end=ts_end,
            psr_type=PSR_SOLAR,
        )
        series = self._to_hourly_series(raw, "da_solar")
        path = self.settings.data_raw / "solar" / "da_solar.parquet"
        save_parquet(series, path)
        self._record_fetch("da_solar", start, end, len(series), path, notes="A69 Solar B16 [14.1.C]")
        return series

    def fetch_nuclear_availability(self, start: date, end: date) -> Optional[pd.DataFrame]:
        """
        Fetch Unavailability of Production Units [15.1.A] (A80) for DE and FR.

        Proxy: installed capacity minus reported unavailability.
        After 2023-04-15 DE nuclear availability is 0 (phase-out flag).

        Args:
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            DataFrame with nuclear_avail_de, nuclear_avail_fr, de_nuclear_phaseout.
        """
        if not self._api_available():
            return None

        client = self._get_client()
        ts_start = pd.Timestamp(start, tz="UTC")
        ts_end = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)

        def _avail(country: str, installed: float) -> pd.Series:
            try:
                raw = self._query(
                    client.query_unavailability_of_generation_units,
                    country,
                    start=ts_start,
                    end=ts_end,
                )
                # Aggregate unavailable MW to hourly if DataFrame of outages
                if isinstance(raw, pd.DataFrame) and not raw.empty:
                    unavail = raw.resample("h").sum(numeric_only=True).iloc[:, 0]
                    return (installed - unavail).clip(lower=0.0)
            except Exception as exc:
                logger.warning("Nuclear unavailability fetch failed for %s: %s — using installed proxy", country, exc)
            idx = pd.date_range(ts_start, ts_end - pd.Timedelta(hours=1), freq="h", tz="UTC")
            return pd.Series(installed, index=idx)

        de = _avail(CTY_DE, DE_NUCLEAR_INSTALLED_PRE_PHASEOUT_MW).rename("nuclear_avail_de")
        fr = _avail(CTY_FR, FR_NUCLEAR_INSTALLED_MW).rename("nuclear_avail_fr")
        df = pd.concat([de, fr], axis=1).sort_index()
        df["de_nuclear_phaseout"] = df.index.date >= DE_NUCLEAR_PHASEOUT_DATE
        df.loc[df["de_nuclear_phaseout"], "nuclear_avail_de"] = 0.0

        path = self.settings.data_raw / "nuclear" / "nuclear_availability.parquet"
        save_parquet(df, path)
        self._record_fetch("nuclear_availability", start, end, len(df), path, notes="A80 Unavailability [15.1.A]")
        return df

    def fetch_cross_border_flows(self, start: date, end: date) -> Optional[pd.DataFrame]:
        """
        Fetch Cross-Border Physical Flows [12.1.G] for DE neighbours.

        Computes net_exports = sum(DE→X) - sum(X→DE) and de_fr_net_flow.

        Args:
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            DataFrame with net_exports, de_fr_net_flow (MW), or None.
        """
        if not self._api_available():
            return None

        client = self._get_client()
        ts_start = pd.Timestamp(start, tz="UTC")
        ts_end = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
        zone = DE_LU if end >= ZONE_SPLIT_DATE else DE_AT_LU

        exports = []
        imports = []
        de_fr_export = None
        de_fr_import = None

        for name, eic in NEIGHBOUR_ZONES.items():
            try:
                out_flow = self._query(
                    client.query_crossborder_flows,
                    zone,
                    eic,
                    start=ts_start,
                    end=ts_end,
                )
                in_flow = self._query(
                    client.query_crossborder_flows,
                    eic,
                    zone,
                    start=ts_start,
                    end=ts_end,
                )
                s_out = self._to_hourly_series(out_flow, f"de_to_{name}").fillna(0.0)
                s_in = self._to_hourly_series(in_flow, f"{name}_to_de").fillna(0.0)
                exports.append(s_out)
                imports.append(s_in)
                if name == "FR":
                    de_fr_export, de_fr_import = s_out, s_in
            except Exception as exc:
                logger.warning("Flow fetch failed DE↔%s: %s", name, exc)
                self.manifest["failed_requests"] += 1

        if not exports:
            return None

        net = sum(exports) - sum(imports)  # type: ignore[arg-type]
        net = net.rename("net_exports")
        if de_fr_export is not None and de_fr_import is not None:
            de_fr = (de_fr_export - de_fr_import).rename("de_fr_net_flow")
        else:
            de_fr = pd.Series(0.0, index=net.index, name="de_fr_net_flow")

        df = pd.concat([net, de_fr], axis=1)
        path = self.settings.data_raw / "flows" / "cross_border_flows.parquet"
        save_parquet(df, path)
        self._record_fetch("cross_border_flows", start, end, len(df), path, notes="12.1.G Physical Flows")
        return df

    def fetch_actual_generation_by_type(self, start: date, end: date) -> Optional[pd.DataFrame]:
        """
        Fetch Actual Generation per Production Type [16.1.B&C].

        Columns: gas, coal, lignite, nuclear, wind onshore/offshore, solar,
        hydro, biomass, other — used for regime detection.

        Args:
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            Wide DataFrame MW by type, or None.
        """
        if not self._api_available():
            return None

        client = self._get_client()
        ts_start = pd.Timestamp(start, tz="UTC")
        ts_end = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)

        psr_map = {
            "gen_gas": PSR_GAS,
            "gen_coal": PSR_HARD_COAL,
            "gen_lignite": PSR_LIGNITE,
            "gen_nuclear": PSR_NUCLEAR,
            "gen_wind_onshore": PSR_WIND_ONSHORE,
            "gen_wind_offshore": PSR_WIND_OFFSHORE,
            "gen_solar": PSR_SOLAR,
            "gen_hydro": PSR_HYDRO,
            "gen_biomass": PSR_BIOMASS,
            "gen_other": PSR_OTHER,
        }
        cols: Dict[str, pd.Series] = {}
        for col, psr in psr_map.items():
            try:
                raw = self._query(
                    client.query_generation,
                    CTY_DE,
                    start=ts_start,
                    end=ts_end,
                    psr_type=psr,
                )
                cols[col] = self._to_hourly_series(raw, col).fillna(0.0)
            except Exception as exc:
                logger.warning("Generation fetch %s failed: %s", col, exc)
                self.manifest["failed_requests"] += 1

        if not cols:
            return None
        df = pd.DataFrame(cols).sort_index()
        path = self.settings.data_raw / "fuels" / "actual_generation.parquet"
        save_parquet(df, path)
        self._record_fetch("actual_generation", start, end, len(df), path, notes="16.1.B&C Actual Generation")
        return df

    def fetch_all(self, start: Optional[date] = None, end: Optional[date] = None) -> Dict[str, Any]:
        """
        Orchestrate all ENTSO-E fetches and write ingestion_manifest.json.

        If API key is placeholder, generates synthetic demo dataset so the
        rest of the pipeline can execute end-to-end.

        Args:
            start: Optional start (default settings.start_date).
            end: Optional end (default settings.end_date).

        Returns:
            Dict of series/frames keyed by dataset name (may be synthetic).
        """
        start = start or self.settings.start_date
        end = end or self.settings.end_date
        results: Dict[str, Any] = {}

        if not self._api_available():
            logger.warning("Generating synthetic fundamental dataset for offline pipeline run")
            results = self._generate_synthetic(start, end)
            self._save_manifest(start, end)
            return results

        fetchers = [
            ("da_price", self.fetch_day_ahead_prices),
            ("da_load", self.fetch_load_forecast),
            ("da_wind", self.fetch_wind_forecast),
            ("da_solar", self.fetch_solar_forecast),
            ("nuclear", self.fetch_nuclear_availability),
            ("flows", self.fetch_cross_border_flows),
            ("generation", self.fetch_actual_generation_by_type),
        ]
        for name, fn in fetchers:
            try:
                results[name] = fn(start, end)
            except Exception as exc:
                logger.exception("Fetch %s failed: %s", name, exc)
                self.manifest["failed_requests"] += 1
                results[name] = None

        # If critical series missing, fall back to synthetic
        if results.get("da_price") is None or results.get("da_load") is None:
            logger.warning("Critical series missing — falling back to synthetic data")
            results = self._generate_synthetic(start, end)

        self._save_manifest(start, end)
        return results

    def _save_manifest(self, start: date, end: date) -> None:
        """Persist ingestion_manifest.json under data/raw/."""
        self.manifest["date_range"] = {"start": str(start), "end": str(end)}
        self.manifest["total_fetches"] = len(self.manifest["fetches"])
        path = self.settings.data_raw / "ingestion_manifest.json"
        write_json(path, self.manifest)
        logger.info("Ingestion manifest saved → %s", path)

    def _to_hourly_series(self, raw: Any, name: str) -> pd.Series:
        """
        Normalise ENTSO-E response to hourly UTC Series.

        Args:
            raw: Series or DataFrame from entsoe-py.
            name: Output series name.

        Returns:
            Hourly mean Series.
        """
        if raw is None:
            return pd.Series(dtype=float, name=name)
        if isinstance(raw, pd.DataFrame):
            if raw.empty:
                return pd.Series(dtype=float, name=name)
            series = raw.iloc[:, 0]
        else:
            series = raw
        series = series.copy()
        series.name = name
        if not isinstance(series.index, pd.DatetimeIndex):
            series.index = pd.to_datetime(series.index, utc=True)
        if series.index.tz is None:
            series.index = series.index.tz_localize("UTC")
        else:
            series.index = series.index.tz_convert("UTC")
        return series.resample("h").mean()

    def _generate_synthetic(self, start: date, end: date) -> Dict[str, Any]:
        """
        Build a physically plausible synthetic DE power dataset.

        Embeds negative-price weekends, Nov/Dec 2024 Dunkelflaute spikes,
        nuclear phase-out, and Ukraine-war structural levels so regime
        detection and modelling can be demonstrated offline.

        Args:
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            Dict of synthetic series matching live ingestion schema.
        """
        rng = np.random.default_rng(RANDOM_SEED)
        idx = pd.date_range(
            pd.Timestamp(start, tz="UTC"),
            pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23),
            freq="h",
            tz="UTC",
        )
        n = len(idx)
        hour = idx.hour.to_numpy(dtype=float)
        dow = idx.dayofweek.to_numpy(dtype=int)
        month = idx.month.to_numpy(dtype=int)
        doy = idx.dayofyear.to_numpy(dtype=float)

        # Load: diurnal + weekly + seasonal
        load = np.clip(
            52_000
            + 8_000 * np.sin(2 * np.pi * (hour - 7) / 24)
            + 3_000 * (dow < 5).astype(float)
            - 4_000 * (dow >= 5).astype(float)
            + 5_000 * np.sin(2 * np.pi * (doy - 15) / 365)
            + rng.normal(0, 800, n),
            22_000,
            85_000,
        )

        # Wind: higher in winter, stochastic
        wind = np.clip(
            12_000
            + 8_000 * np.sin(2 * np.pi * (doy + 90) / 365)
            + rng.normal(0, 4_000, n),
            200,
            55_000,
        )

        # Solar: daylight envelope
        solar_env = np.maximum(0, np.sin(np.pi * (hour - 5) / 14))
        solar_env = np.where((hour >= 5) & (hour <= 20), solar_env, 0.0)
        solar_season = np.maximum(0.15, np.sin(np.pi * (doy - 80) / 200))
        solar = np.clip(solar_env * solar_season * 38_000 + rng.normal(0, 500, n), 0, 45_000)

        # Dunkelflaute Nov 2-7 and Dec 12-14 2024
        dunk_mask = np.zeros(n, dtype=bool)
        for d0, d1 in [(date(2024, 11, 2), date(2024, 11, 7)), (date(2024, 12, 12), date(2024, 12, 14))]:
            dunk_mask |= (idx.date >= d0) & (idx.date <= d1)
        wind = np.where(dunk_mask, wind * 0.08, wind)
        solar = np.where(dunk_mask, solar * 0.15, solar)

        residual = load - wind - solar
        # Price from residual load merit-order proxy
        price = (
            40
            + 0.0025 * residual
            + 15 * np.sin(2 * np.pi * hour / 24)
            + rng.normal(0, 12, n)
        )
        # Negative prices: high renewables + weekend + summer
        glut = (wind + solar) / load > 0.75
        weekend = dow >= 5
        summer = (month >= 4) & (month <= 9)
        neg_mask = glut & weekend & summer
        price = np.where(neg_mask, rng.uniform(-180, 5, n), price)
        # Dunkelflaute spikes
        price = np.where(dunk_mask, 200 + 0.01 * residual + rng.normal(0, 80, n), price)
        price = np.where(dunk_mask & (hour >= 17) & (hour <= 20), price + rng.uniform(200, 600, n), price)
        # Post Ukraine war level shift
        war = idx.date >= date(2022, 2, 24)
        price = np.where(war & (idx.date < date(2023, 1, 1)), price + 80, price)
        # EPEX outage spike
        epex = (idx.date >= date(2024, 6, 25)) & (idx.date <= EPEX_OUTAGE_DATE)
        price = np.where(epex, np.minimum(price + 1500, 2325), price)
        price = np.clip(np.asarray(price, dtype=float), -500, 3000)

        da_price = pd.Series(price, index=idx, name="da_price")
        da_load = pd.Series(load, index=idx, name="da_load")
        da_wind = pd.Series(wind, index=idx, name="da_wind")
        da_solar = pd.Series(solar, index=idx, name="da_solar")

        fr_nuc = np.full(n, 45_000.0) + rng.normal(0, 1_500, n)
        de_nuc = np.where(idx.date < DE_NUCLEAR_PHASEOUT_DATE, 4_000.0, 0.0)
        nuclear = pd.DataFrame(
            {
                "nuclear_avail_de": de_nuc,
                "nuclear_avail_fr": fr_nuc,
                "de_nuclear_phaseout": idx.date >= DE_NUCLEAR_PHASEOUT_DATE,
            },
            index=idx,
        )

        net_exports = pd.Series(rng.normal(2_000, 3_000, n), index=idx, name="net_exports")
        de_fr = pd.Series(rng.normal(500, 1_500, n), index=idx, name="de_fr_net_flow")
        flows = pd.DataFrame({"net_exports": net_exports, "de_fr_net_flow": de_fr}, index=idx)

        generation = pd.DataFrame(
            {
                "gen_gas": np.maximum(0, residual * 0.25 + rng.normal(0, 500, n)),
                "gen_coal": np.maximum(0, residual * 0.15 + rng.normal(0, 300, n)),
                "gen_lignite": np.maximum(0, residual * 0.20 + rng.normal(0, 300, n)),
                "gen_nuclear": de_nuc,
                "gen_wind_onshore": da_wind * 0.85,
                "gen_wind_offshore": da_wind * 0.15,
                "gen_solar": da_solar,
                "gen_hydro": rng.uniform(1_500, 4_000, n),
                "gen_biomass": rng.uniform(4_000, 6_000, n),
                "gen_other": rng.uniform(500, 2_000, n),
            },
            index=idx,
        )

        # Fuel prices (synthetic TTF / EUA / coal)
        ttf = 30 + 40 * (idx.date >= date(2022, 2, 24)).astype(float) + rng.normal(0, 3, n)
        ttf = pd.Series(np.maximum(5, ttf), index=idx, name="ttf_gas_price")
        eua = pd.Series(70 + rng.normal(0, 5, n), index=idx, name="eua_carbon_price")
        coal = pd.Series(120 + rng.normal(0, 8, n), index=idx, name="coal_price")
        fuels = pd.DataFrame({"ttf_gas_price": ttf, "eua_carbon_price": eua, "coal_price": coal}, index=idx)

        # Persist
        save_parquet(da_price, self.settings.data_raw / "prices" / "da_price.parquet")
        save_parquet(da_load, self.settings.data_raw / "load" / "da_load.parquet")
        save_parquet(da_wind, self.settings.data_raw / "wind" / "da_wind.parquet")
        save_parquet(da_solar, self.settings.data_raw / "solar" / "da_solar.parquet")
        save_parquet(nuclear, self.settings.data_raw / "nuclear" / "nuclear_availability.parquet")
        save_parquet(flows, self.settings.data_raw / "flows" / "cross_border_flows.parquet")
        save_parquet(generation, self.settings.data_raw / "fuels" / "actual_generation.parquet")
        save_parquet(fuels, self.settings.data_raw / "fuels" / "fuel_prices.parquet")

        self._record_fetch("synthetic_bundle", start, end, n, self.settings.data_raw, notes="offline synthetic")
        logger.info("Synthetic dataset written — %s hours | %s → %s", n, start, end)

        return {
            "da_price": da_price,
            "da_load": da_load,
            "da_wind": da_wind,
            "da_solar": da_solar,
            "nuclear": nuclear,
            "flows": flows,
            "generation": generation,
            "fuels": fuels,
        }
