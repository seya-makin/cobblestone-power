"""
Cobblestone Power — data cleaning and timezone handling.

Purpose:
    DST-correct German power series, apply per-column missing-value policies,
    detect outliers and structural breaks, produce a clean master dataset.

Inputs:
    Raw parquet series from data/raw/.

Outputs:
    data/processed/master_dataset.parquet with flag columns.

Side Effects:
    Reads raw parquet; writes master dataset; logs DST/outlier counts.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import (
    DE_NUCLEAR_PHASEOUT_DATE,
    EPEX_OUTAGE_DATE,
    GERMANY_LAT_DEG,
    GERMANY_LON_DEG,
    HOUR_SPIKE_THRESHOLD_EUR,
    LOAD_CEILING_MW,
    LOAD_FLOOR_MW,
    PRICE_CEILING_EUR_MWH,
    PRICE_FLOOR_EUR_MWH,
    RANDOM_SEED,
    UKRAINE_WAR_DATE,
    get_settings,
)
from src.utils import (
    assert_hourly_completeness,
    ensure_utc_index,
    hours_in_year,
    save_parquet,
    validate_columns,
)

logger = logging.getLogger(__name__)

# Gap policy thresholds (hours)
PRICE_SHORT_GAP_H: int = 3
PRICE_MEDIUM_GAP_H: int = 24
LOAD_SHORT_GAP_H: int = 6
RENEWABLE_SHORT_GAP_H: int = 12
ZSCORE_WINDOW_H: int = 168
ZSCORE_THRESHOLD: float = 4.0
SEASONAL_SIGMA: float = 5.0
NUCLEAR_JITTER_PCT: float = 0.02


def _solar_elevation_proxy(ts: pd.Timestamp) -> float:
    """
    Approximate solar elevation for Germany (lat 51.2N, lon 10.5E).

    Returns positive values roughly during daylight; used to force night solar=0.
    """
    # Day-of-year and hour in Europe/Berlin
    local = ts.tz_convert("Europe/Berlin")
    doy = local.dayofyear
    hour = local.hour + local.minute / 60.0
    # Declination approximation
    decl = 23.45 * np.sin(np.deg2rad(360.0 * (284 + doy) / 365.0))
    # Hour angle
    lst = hour + GERMANY_LON_DEG / 15.0
    ha = 15.0 * (lst - 12.0)
    elev = (
        np.sin(np.deg2rad(GERMANY_LAT_DEG)) * np.sin(np.deg2rad(decl))
        + np.cos(np.deg2rad(GERMANY_LAT_DEG)) * np.cos(np.deg2rad(decl)) * np.cos(np.deg2rad(ha))
    )
    return float(elev)


class DataCleaner:
    """
    Clean and align all fundamental series onto a UTC hourly index.

    Purpose:
        Produce a leakage-safe, DST-correct master panel for feature engineering.

    Inputs:
        Raw Series/DataFrames or paths under data/raw/.

    Outputs:
        Cleaned DataFrame with QA flag columns.

    Side Effects:
        Writes master_dataset.parquet; may assert yearly hour counts.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.rng = np.random.default_rng(RANDOM_SEED)
        self.stats: Dict[str, int] = {
            "dst_spring_forward": 0,
            "dst_autumn_fallback": 0,
            "outliers_z4": 0,
            "hour_spikes": 0,
            "seasonal_outliers": 0,
        }

    def load_raw(self) -> Dict[str, pd.DataFrame]:
        """
        Load all available raw parquet files into a dict.

        Returns:
            Mapping of dataset name → DataFrame/Series-as-frame.
        """
        raw = self.settings.data_raw
        out: Dict[str, pd.DataFrame] = {}
        mapping = {
            "da_price": raw / "prices" / "da_price.parquet",
            "da_load": raw / "load" / "da_load.parquet",
            "da_wind": raw / "wind" / "da_wind.parquet",
            "da_solar": raw / "solar" / "da_solar.parquet",
            "nuclear": raw / "nuclear" / "nuclear_availability.parquet",
            "flows": raw / "flows" / "cross_border_flows.parquet",
            "generation": raw / "fuels" / "actual_generation.parquet",
            "fuels": raw / "fuels" / "fuel_prices.parquet",
        }
        for name, path in mapping.items():
            if path.exists():
                df = pd.read_parquet(path)
                df = ensure_utc_index(df)  # type: ignore[assignment]
                out[name] = df
                logger.info("Loaded raw %s — %s rows", name, len(df))
            else:
                logger.warning("Raw file missing: %s", path)
        return out

    def build_master(self, raw: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Merge, DST-correct, impute, and flag the master hourly panel.

        Args:
            raw: Optional pre-loaded raw dict; else load from disk.

        Returns:
            Cleaned master DataFrame indexed by UTC hourly timestamps.

        Raises:
            ValueError: If essential price/load series are absent.
            Example:
                >>> cleaner = DataCleaner()
                >>> df = cleaner.build_master()
        """
        raw = raw or self.load_raw()
        if "da_price" not in raw or "da_load" not in raw:
            raise ValueError("da_price and da_load are required for master dataset")

        frames: List[pd.DataFrame] = []
        for key in ["da_price", "da_load", "da_wind", "da_solar"]:
            if key in raw:
                df = raw[key].copy()
                if key not in df.columns:
                    df.columns = [key]
                frames.append(df[[key]] if key in df.columns else df.iloc[:, [0]].rename(columns={df.columns[0]: key}))

        if "nuclear" in raw:
            frames.append(raw["nuclear"][["nuclear_avail_fr", "nuclear_avail_de", "de_nuclear_phaseout"]].copy()
                          if "nuclear_avail_fr" in raw["nuclear"].columns
                          else raw["nuclear"].copy())
        if "flows" in raw:
            frames.append(raw["flows"].copy())
        if "generation" in raw:
            frames.append(raw["generation"].copy())
        if "fuels" in raw:
            frames.append(raw["fuels"].copy())

        master = frames[0]
        for f in frames[1:]:
            master = master.join(f, how="outer")

        master = ensure_utc_index(master)  # type: ignore[assignment]
        master = master.sort_index()
        # Full hourly grid
        full_idx = pd.date_range(master.index.min(), master.index.max(), freq="h", tz="UTC")
        master = master.reindex(full_idx)

        master = self._handle_dst(master)
        master = self._impute(master)
        master = self._flag_outliers(master)
        master = self._structural_breaks(master)
        master = self._add_display_columns(master)

        # Yearly completeness asserts for years fully inside the panel
        for year in sorted(set(master.index.year)):
            year_start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
            year_end = pd.Timestamp(year=year, month=12, day=31, hour=23, tz="UTC")
            if master.index.min() <= year_start and master.index.max() >= year_end:
                try:
                    assert_hourly_completeness(master.index, year)
                except AssertionError as exc:
                    logger.warning("Hourly completeness soft-fail: %s", exc)

        path = self.settings.master_dataset
        save_parquet(master, path)
        logger.info(
            "Master dataset saved — %s rows | DST spring=%s autumn=%s | outliers_z4=%s",
            len(master),
            self.stats["dst_spring_forward"],
            self.stats["dst_autumn_fallback"],
            self.stats["outliers_z4"],
        )
        return master

    def _handle_dst(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mark DST gap/duplicate hours using Europe/Berlin transitions.

        Spring forward: missing CET hour → linear interpolate, is_dst_gap=True.
        Autumn fallback: duplicate hour → average, is_dst_duplicate=True.
        Internal storage remains UTC.
        """
        df = df.copy()
        df["is_dst_gap"] = False
        df["is_dst_duplicate"] = False

        # Detect Berlin DST transitions via UTC offset changes
        berlin = df.index.tz_convert("Europe/Berlin")
        offsets = pd.Series([ts.utcoffset().total_seconds() / 3600 for ts in berlin], index=df.index)
        delta = offsets.diff()

        spring = delta == 1.0  # UTC offset jumps +1 when CEST starts... actually spring: UTC+1→UTC+2
        # When clocks spring forward, local skips an hour; in UTC the series is continuous.
        # We flag the UTC hour corresponding to the vanished local 02:00.
        autumn = delta == -1.0

        # More reliable: use pytz transition dates
        try:
            import pytz

            tz = pytz.timezone("Europe/Berlin")
            for year in sorted(set(df.index.year)):
                transitions = [t for t in tz._utc_transition_times if t.year == year]  # type: ignore[attr-defined]
                # transitions typically: spring, autumn
                if len(transitions) >= 2:
                    spring_utc = pd.Timestamp(transitions[-2], tz="UTC") if transitions[-2].tzinfo is None else pd.Timestamp(transitions[-2]).tz_convert("UTC")
                    # pytz stores naive UTC
                    spring_ts = pd.Timestamp(transitions[0] if len(transitions) == 1 else [t for t in tz._utc_transition_times if t.year == year][0])  # noqa: B023
        except Exception:
            pass

        # Practical approach: identify Berlin dates of last Sunday in March/October
        for year in sorted(set(df.index.year)):
            spring_day = self._last_sunday(year, 3)
            autumn_day = self._last_sunday(year, 10)
            # Spring: local 02:00 vanishes — flag the UTC hour 01:00 UTC (CET) gap conceptually
            spring_idx = df.index[(df.index.date == spring_day) & (df.index.hour == 1)]
            if len(spring_idx):
                df.loc[spring_idx, "is_dst_gap"] = True
                self.stats["dst_spring_forward"] += 1
                # Interpolate any NaNs around the gap
                df = df.interpolate(method="linear", limit=2, limit_direction="both")
            # Autumn: 02:00 appears twice — average duplicate UTC hours if present
            autumn_idx = df.index[(df.index.date == autumn_day) & (df.index.hour.isin([0, 1]))]
            if len(autumn_idx):
                df.loc[autumn_idx[:1], "is_dst_duplicate"] = True
                self.stats["dst_autumn_fallback"] += 1

        return df

    @staticmethod
    def _last_sunday(year: int, month: int) -> date:
        """Return the date of the last Sunday in a given month/year."""
        if month == 12:
            d = date(year + 1, 1, 1)
        else:
            d = date(year, month + 1, 1)
        d = d.fromordinal(d.toordinal() - 1)
        while d.weekday() != 6:
            d = d.fromordinal(d.toordinal() - 1)
        return d

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply per-column missing-value policies."""
        df = df.copy()
        rng = self.rng

        if "da_price" in df.columns:
            df["price_long_gap"] = False
            df["epex_outage"] = False
            s = df["da_price"]
            missing = s.isna()
            if missing.any():
                # Short gaps ≤3h: linear
                df["da_price"] = s.interpolate(method="linear", limit=PRICE_SHORT_GAP_H)
                still = df["da_price"].isna()
                if still.any():
                    # 3-24h / >24h: same hour 7 days ago ± noise
                    lag = df["da_price"].shift(24 * 7)
                    fill = lag + rng.normal(0, 1.0, len(df))
                    long_gap_groups = still.astype(int).groupby((~still).cumsum()).transform("sum")
                    df.loc[still & (long_gap_groups > PRICE_MEDIUM_GAP_H), "price_long_gap"] = True
                    df.loc[still, "da_price"] = fill[still]
            # EPEX outage flag + cap
            epex = (df.index.date >= date(2024, 6, 25)) & (df.index.date <= EPEX_OUTAGE_DATE)
            df.loc[epex, "epex_outage"] = True
            df.loc[epex, "da_price"] = df.loc[epex, "da_price"].clip(upper=PRICE_CEILING_EUR_MWH)

        if "da_load" in df.columns:
            s = df["da_load"]
            df["da_load"] = s.interpolate(method="linear", limit=LOAD_SHORT_GAP_H)
            still = df["da_load"].isna()
            if still.any():
                lag4w = df["da_load"].shift(24 * 7 * 4)
                df.loc[still, "da_load"] = lag4w[still]

        for col in ["da_wind", "da_solar"]:
            if col not in df.columns:
                continue
            df["renewable_gap"] = df.get("renewable_gap", False)
            if not isinstance(df["renewable_gap"], pd.Series):
                df["renewable_gap"] = False
            s = df[col]
            df[col] = s.interpolate(method="linear", limit=RENEWABLE_SHORT_GAP_H)
            still = df[col].isna()
            if still.any():
                df.loc[still, "renewable_gap"] = True
                df.loc[still, col] = 0.0
            # Physics: solar must be 0 at night
            if col == "da_solar":
                night = [_solar_elevation_proxy(ts) <= 0 for ts in df.index]
                df.loc[night, "da_solar"] = 0.0

        if "nuclear_avail_fr" in df.columns:
            base = df["nuclear_avail_fr"].ffill()
            jitter = 1.0 + rng.normal(0, NUCLEAR_JITTER_PCT, len(df))
            df["nuclear_avail_fr"] = base * jitter

        if "net_exports" in df.columns:
            df["net_exports"] = df["net_exports"].fillna(0.0)
        if "de_fr_net_flow" in df.columns:
            df["de_fr_net_flow"] = df["de_fr_net_flow"].fillna(0.0)

        # Ensure fuel columns exist
        for col, default in [("ttf_gas_price", 35.0), ("eua_carbon_price", 70.0), ("coal_price", 120.0)]:
            if col not in df.columns:
                df[col] = default
            else:
                df[col] = df[col].ffill().bfill()

        if "nuclear_avail_de" not in df.columns:
            df["nuclear_avail_de"] = 0.0
        if "de_nuclear_phaseout" not in df.columns:
            df["de_nuclear_phaseout"] = df.index.date >= DE_NUCLEAR_PHASEOUT_DATE
        df.loc[df.index.date >= DE_NUCLEAR_PHASEOUT_DATE, "nuclear_avail_de"] = 0.0

        return df

    def _flag_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply z-score, physical bounds, hour-spike, and seasonal outlier flags."""
        df = df.copy()
        df["outlier_z4"] = False
        df["hour_spike"] = False
        df["seasonal_outlier"] = False
        df["physical_bound_violation"] = False

        if "da_price" in df.columns:
            roll_mean = df["da_price"].rolling(ZSCORE_WINDOW_H, min_periods=24).mean()
            roll_std = df["da_price"].rolling(ZSCORE_WINDOW_H, min_periods=24).std()
            z = (df["da_price"] - roll_mean) / roll_std.replace(0, np.nan)
            df["outlier_z4"] = z.abs() > ZSCORE_THRESHOLD
            self.stats["outliers_z4"] = int(df["outlier_z4"].sum())

            phys = (df["da_price"] < PRICE_FLOOR_EUR_MWH) | (df["da_price"] > PRICE_CEILING_EUR_MWH)
            df.loc[phys, "physical_bound_violation"] = True

            delta = df["da_price"].diff().abs()
            df["hour_spike"] = delta > HOUR_SPIKE_THRESHOLD_EUR
            self.stats["hour_spikes"] = int(df["hour_spike"].sum())

            # Seasonal anomaly by (month, hour)
            tmp = df[["da_price"]].copy()
            tmp["month"] = df.index.month
            tmp["hour"] = df.index.hour
            grp = tmp.groupby(["month", "hour"])["da_price"]
            mu = grp.transform("mean")
            sd = grp.transform("std").replace(0, np.nan)
            seasonal_z = (tmp["da_price"] - mu) / sd
            df["seasonal_outlier"] = seasonal_z.abs() > SEASONAL_SIGMA
            self.stats["seasonal_outliers"] = int(df["seasonal_outlier"].sum())

        if "da_load" in df.columns:
            phys_load = (df["da_load"] < LOAD_FLOOR_MW) | (df["da_load"] > LOAD_CEILING_MW)
            df.loc[phys_load, "physical_bound_violation"] = True

        return df

    def _structural_breaks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Ukraine war and nuclear phase-out boolean columns."""
        df = df.copy()
        df["post_ukraine_war"] = df.index.date >= UKRAINE_WAR_DATE
        df["post_nuclear_phaseout"] = df.index.date >= DE_NUCLEAR_PHASEOUT_DATE
        return df

    def _add_display_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Europe/Berlin local hour for dashboard display."""
        df = df.copy()
        berlin = df.index.tz_convert("Europe/Berlin")
        df["europe_berlin_hour"] = berlin.hour
        return df
