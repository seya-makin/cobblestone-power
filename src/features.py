"""
Cobblestone Power — feature engineering (67 features, leakage-safe).

Purpose:
    Transform the regime-labelled master panel into a modelling feature matrix
    with documented physical rationale and an explicit leakage assertion.

Inputs:
    Regime-augmented master DataFrame.

Outputs:
    features.parquet; features_manifest.json; leakage_check.json.

Side Effects:
    Writes processed artefacts; raises DataLeakageError on |corr|>0.95 with future target.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import holidays
import numpy as np
import pandas as pd

from config.settings import DE_NUCLEAR_PHASEOUT_DATE, THERMAL_BASELOAD_MW, UKRAINE_WAR_DATE, get_settings
from src.utils import (
    cyclic_encode,
    list_required_feature_groups,
    save_parquet,
    validate_columns,
    write_json,
)

logger = logging.getLogger(__name__)

TARGET_FEATURE_COUNT: int = 77
LEAKAGE_CORR_THRESHOLD: float = 0.95


class DataLeakageError(Exception):
    """Raised when a feature correlates too strongly with the future target."""


class FeatureEngineer:
    """
    Build the full 67-feature matrix for German DA price forecasting.

    Purpose:
        Encode fundamentals, fuels, calendar, lags, rolling stats, interactions,
        regimes, and structural breaks without look-ahead bias.

    Inputs:
        DataFrame with price, load, wind, solar, fuels, regime columns.

    Outputs:
        Feature DataFrame + manifest.

    Side Effects:
        Writes parquet/JSON under data/processed/ and outputs/logs/.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.manifest: Dict[str, Any] = {"features": [], "groups": list_required_feature_groups()}
        self.de_holidays = holidays.country_holidays("DE")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer all feature groups A–H.

        Args:
            df: Regime-labelled master panel.

        Returns:
            Feature DataFrame aligned to df.index (NaNs at warm-up edges).

        Raises:
            ValueError: If required base columns missing.
            DataLeakageError: If leakage check fails.
        """
        validate_columns(df, ["da_price", "da_load", "da_wind", "da_solar"], "FeatureEngineer")
        self.manifest["features"] = []
        out = df.copy()

        out = self._group_a_fundamental(out)
        out = self._group_b_fuel(out)
        out = self._group_c_calendar(out)
        out = self._group_d_lags(out)
        out = self._group_e_rolling(out)
        out = self._group_f_interactions(out)
        out = self._group_g_regime(out)
        out = self._group_h_structural(out)

        feature_cols = self._feature_column_list(out)
        feat = out[feature_cols].copy()
        feat = feat.replace([np.inf, -np.inf], np.nan)

        self.assert_no_leakage(feat, out["da_price"])
        self._finalise_manifest(feat)
        save_parquet(feat, self.settings.features_path)
        panel = feat.join(out[["da_price"]], how="left")
        save_parquet(panel, self.settings.data_processed / "features_with_target.parquet")
        write_json(self.settings.features_manifest, self.manifest)

        logger.info(
            "Features engineered — %s columns | %s rows | target count goal=%s",
            len(feature_cols),
            len(feat),
            TARGET_FEATURE_COUNT,
        )
        return feat

    def _register(self, name: str, group: str, rationale: str, formula: str, series: pd.Series) -> None:
        """Append feature metadata to the manifest."""
        self.manifest["features"].append(
            {
                "name": name,
                "group": group,
                "physical_rationale": rationale,
                "formula": formula,
                "min": float(series.min()) if series.notna().any() else None,
                "max": float(series.max()) if series.notna().any() else None,
                "mean": float(series.mean()) if series.notna().any() else None,
                "fuel_data_dependency": name
                in {
                    "ttf_gas_price",
                    "eua_carbon_price",
                    "clean_spark_spread",
                    "dark_spread",
                    "gas_to_coal_switch_price",
                    "ttf_7d_change",
                    "eua_7d_change",
                    "gas_x_residual_load",
                },
            }
        )

    def _group_a_fundamental(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group A — 11 supply/demand fundamentals."""
        out = df.copy()
        out["residual_load"] = out["da_load"] - out["da_wind"] - out["da_solar"]
        out["residual_load_normalised"] = out["residual_load"] / out["da_load"].replace(0, np.nan)
        out["renewable_penetration"] = (out["da_wind"] + out["da_solar"]) / out["da_load"].replace(0, np.nan)
        out["wind_solar_combined"] = out["da_wind"] + out["da_solar"]
        out["thermal_gap"] = (out["residual_load"] - THERMAL_BASELOAD_MW).clip(lower=0)
        if "net_exports" not in out.columns:
            out["net_exports"] = 0.0
        out["net_import_ratio"] = out["net_exports"] / out["da_load"].replace(0, np.nan)
        out["de_fr_flow"] = out["de_fr_net_flow"] if "de_fr_net_flow" in out.columns else 0.0
        out["fr_nuclear_avail"] = out["nuclear_avail_fr"] if "nuclear_avail_fr" in out.columns else 45_000.0

        hour = out.index.hour
        solar_peak = out["renewable_penetration"].where((hour >= 10) & (hour <= 16))
        out["solar_share_of_peak_hours"] = solar_peak.rolling(24 * 7, min_periods=24).mean()
        out["wind_ramp_rate"] = out["da_wind"].diff(1)
        out["solar_ramp_rate"] = out["da_solar"].diff(1)

        # Merit-order thermal mix (real SMARD actual generation when available)
        gas = out["gen_gas"] if "gen_gas" in out.columns else pd.Series(0.0, index=out.index)
        lignite = out["gen_lignite"] if "gen_lignite" in out.columns else pd.Series(0.0, index=out.index)
        out["actual_gas_generation_mw"] = gas.astype(float)
        out["actual_lignite_generation_mw"] = lignite.astype(float)
        out["gas_plus_lignite_mw"] = out["actual_gas_generation_mw"] + out["actual_lignite_generation_mw"]
        out["thermal_share"] = (
            out["gas_plus_lignite_mw"] / out["da_load"].replace(0, np.nan)
        ).fillna(0.0)

        specs = [
            ("residual_load", "Load net of wind/solar — primary merit-order driver", "da_load - da_wind - da_solar"),
            ("residual_load_normalised", "Scale-free residual tightness", "residual_load / da_load"),
            ("renewable_penetration", "Share of demand met by wind+solar", "(wind+solar)/load"),
            ("wind_solar_combined", "Absolute renewable output", "da_wind + da_solar"),
            ("thermal_gap", "Residual above thermal baseload", "max(0, residual - thermal_baseload)"),
            ("net_import_ratio", "Export intensity vs load", "net_exports / da_load"),
            ("de_fr_flow", "DE–FR net flow (FR nuclear coupling)", "de_fr_net_flow"),
            ("fr_nuclear_avail", "French nuclear availability", "nuclear_avail_fr"),
            ("solar_share_of_peak_hours", "7d mean solar pen in hours 10-16", "roll7d mean"),
            ("wind_ramp_rate", "Hourly wind change", "diff(da_wind)"),
            ("solar_ramp_rate", "Hourly solar change", "diff(da_solar)"),
            ("actual_gas_generation_mw", "SMARD actual gas generation — merit order", "gen_gas"),
            ("actual_lignite_generation_mw", "SMARD actual lignite generation", "gen_lignite"),
            ("gas_plus_lignite_mw", "Thermal stack (gas+lignite)", "gas + lignite"),
            ("thermal_share", "Thermal share of load", "(gas+lignite)/load"),
        ]
        for name, rationale, formula in specs:
            self._register(name, "A_fundamental", rationale, formula, out[name])
        return out

    def _group_b_fuel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group B — 7 fuel/carbon features."""
        out = df.copy()
        for col, default in [("ttf_gas_price", 35.0), ("eua_carbon_price", 70.0), ("coal_price", 120.0)]:
            if col not in out.columns:
                out[col] = default
            out[col] = out[col].ffill().bfill()

        price_l24 = out["da_price"].shift(24)
        out["clean_spark_spread"] = price_l24 - (out["ttf_gas_price"] * 0.40) - (out["eua_carbon_price"] * 0.38)
        out["dark_spread"] = price_l24 - (out["coal_price"] * 0.32) - (out["eua_carbon_price"] * 0.85)
        out["gas_to_coal_switch_price"] = (out["ttf_gas_price"] * 0.40 + out["eua_carbon_price"] * 0.38) - (
            out["coal_price"] * 0.32 + out["eua_carbon_price"] * 0.85
        )
        out["ttf_7d_change"] = out["ttf_gas_price"] - out["ttf_gas_price"].shift(24 * 7)
        out["eua_7d_change"] = out["eua_carbon_price"] - out["eua_carbon_price"].shift(24 * 7)

        specs = [
            ("ttf_gas_price", "TTF gas sets CCGT SRMC", "spot TTF"),
            ("eua_carbon_price", "EUA carbon cost in merit order", "spot EUA"),
            ("clean_spark_spread", "Gas peaker margin proxy (lagged price)", "price_l24 - 0.4*TTF - 0.38*EUA"),
            ("dark_spread", "Coal margin proxy (lagged price)", "price_l24 - 0.32*coal - 0.85*EUA"),
            ("gas_to_coal_switch_price", "Relative gas vs coal SRMC", "gas_srmc - coal_srmc"),
            ("ttf_7d_change", "Weekly gas momentum", "TTF - TTF_7d"),
            ("eua_7d_change", "Weekly carbon momentum", "EUA - EUA_7d"),
        ]
        for name, rationale, formula in specs:
            self._register(name, "B_fuel_carbon", rationale, formula, out[name])
        return out

    def _group_c_calendar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group C — 16 cyclically encoded calendar features."""
        out = df.copy()
        hour = out.index.hour.astype(float)
        dow = out.index.dayofweek.astype(float)
        month = out.index.month.astype(float)
        woy = out.index.isocalendar().week.astype(float).to_numpy()

        out["hour_sin"], out["hour_cos"] = cyclic_encode(hour.to_numpy(), 24)
        out["dow_sin"], out["dow_cos"] = cyclic_encode(dow.to_numpy(), 7)
        out["month_sin"], out["month_cos"] = cyclic_encode(month.to_numpy(), 12)
        out["woy_sin"], out["woy_cos"] = cyclic_encode(woy, 52)

        out["is_weekend"] = (out.index.dayofweek >= 5).astype(float)
        out["is_german_holiday"] = pd.Series(
            [1.0 if ts.date() in self.de_holidays else 0.0 for ts in out.index],
            index=out.index,
        )
        out["days_since_year_start"] = out.index.dayofyear.astype(float)
        out["season"] = ((out.index.month % 12) // 3).astype(float)
        out["is_peak_hour"] = (
            (out.index.hour >= 8) & (out.index.hour < 20) & (out.index.dayofweek < 5)
        ).astype(float)
        out["quarter_hour_of_day"] = (out.index.hour * 4).astype(float)
        out["hour_of_day"] = hour
        out["week_of_year"] = woy

        specs = [
            ("hour_sin", "Diurnal cycle", "sin(2π hour/24)"),
            ("hour_cos", "Diurnal cycle", "cos(2π hour/24)"),
            ("dow_sin", "Weekly cycle", "sin(2π dow/7)"),
            ("dow_cos", "Weekly cycle", "cos(2π dow/7)"),
            ("month_sin", "Annual cycle", "sin(2π month/12)"),
            ("month_cos", "Annual cycle", "cos(2π month/12)"),
            ("woy_sin", "Week-of-year cycle", "sin(2π woy/52)"),
            ("woy_cos", "Week-of-year cycle", "cos(2π woy/52)"),
            ("is_weekend", "Weekend demand/renewable pattern", "dow>=5"),
            ("is_german_holiday", "Holiday load reduction", "DE holiday calendar"),
            ("days_since_year_start", "Intra-year position", "dayofyear"),
            ("season", "Meteorological season 0-3", "month mapping"),
            ("is_peak_hour", "EPEX peak window Mon-Fri 8-20", "boolean"),
            ("quarter_hour_of_day", "Fine diurnal index", "hour*4"),
            ("hour_of_day", "Raw hour for interactions", "hour"),
            ("week_of_year", "ISO week number", "isocalendar week"),
        ]
        for name, rationale, formula in specs:
            self._register(name, "C_calendar", rationale, formula, out[name])
        return out

    def _group_d_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group D — lag features, all shifted ≥ 24h (leakage-safe)."""
        out = df.copy()
        out["price_lag_24h"] = out["da_price"].shift(24)
        out["price_lag_48h"] = out["da_price"].shift(48)
        out["price_lag_72h"] = out["da_price"].shift(72)
        out["price_lag_168h"] = out["da_price"].shift(168)
        out["price_lag_336h"] = out["da_price"].shift(336)
        out["price_lag_672h"] = out["da_price"].shift(672)  # same hour 4 weeks ago
        # True annual lag — do NOT fill with shorter lags (that diluted corr to ~0.38).
        # XGBoost handles NaN natively; warm-up rows are dropped via y/feature masks
        # that only require core columns, or left as NaN for tree default direction.
        out["price_lag_8736h"] = out["da_price"].shift(8736)  # same hour ~1 year ago
        out["price_lag_672h"] = out["price_lag_672h"].fillna(out["price_lag_336h"]).fillna(
            out["price_lag_168h"]
        )
        out["residual_load_lag_24h"] = out["residual_load"].shift(24)
        out["residual_load_lag_168h"] = out["residual_load"].shift(168)
        out["wind_lag_24h"] = out["da_wind"].shift(24)

        for name, formula in [
            ("price_lag_24h", "price.shift(24)"),
            ("price_lag_48h", "price.shift(48)"),
            ("price_lag_72h", "price.shift(72)"),
            ("price_lag_168h", "price.shift(168)"),
            ("price_lag_336h", "price.shift(336)"),
            ("price_lag_672h", "price.shift(672)"),
            ("price_lag_8736h", "price.shift(8736)"),
            ("residual_load_lag_24h", "residual.shift(24)"),
            ("residual_load_lag_168h", "residual.shift(168)"),
            ("wind_lag_24h", "wind.shift(24)"),
        ]:
            self._register(name, "D_lags", "Autoregressive / persistence structure", formula, out[name])
        return out

    def _group_e_rolling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group E — 10 rolling statistics, all shifted 24h to prevent leakage."""
        out = df.copy()
        price = out["da_price"]
        out["price_roll7d_mean"] = price.rolling(24 * 7, min_periods=24).mean().shift(24)
        out["price_roll7d_std"] = price.rolling(24 * 7, min_periods=24).std().shift(24)
        out["price_roll7d_max"] = price.rolling(24 * 7, min_periods=24).max().shift(24)
        out["price_roll7d_min"] = price.rolling(24 * 7, min_periods=24).min().shift(24)
        out["price_roll28d_mean"] = price.rolling(24 * 28, min_periods=24).mean().shift(24)
        out["price_roll28d_std"] = price.rolling(24 * 28, min_periods=24).std().shift(24)
        out["residual_load_roll7d_mean"] = out["residual_load"].rolling(24 * 7, min_periods=24).mean().shift(24)
        out["renewable_pen_roll7d_mean"] = (
            out["renewable_penetration"].rolling(24 * 7, min_periods=24).mean().shift(24)
        )
        out["price_roll7d_skewness"] = price.rolling(24 * 7, min_periods=48).skew().shift(24)
        out["negative_price_freq_7d"] = (price < 0).astype(float).rolling(24 * 7, min_periods=24).mean().shift(24)

        for name in [
            "price_roll7d_mean",
            "price_roll7d_std",
            "price_roll7d_max",
            "price_roll7d_min",
            "price_roll28d_mean",
            "price_roll28d_std",
            "residual_load_roll7d_mean",
            "renewable_pen_roll7d_mean",
            "price_roll7d_skewness",
            "negative_price_freq_7d",
        ]:
            self._register(name, "E_rolling", "Recent distribution shape (shifted 24h)", f"{name}", out[name])
        return out

    def _group_f_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group F — physically motivated interactions (incl. summer neg-price)."""
        out = df.copy()
        month_in_summer = out.index.month.isin([4, 5, 6, 7, 8, 9]).astype(float)
        out["month_in_summer"] = month_in_summer
        out["gas_x_residual_load"] = out["ttf_gas_price"] * out["residual_load"]
        out["wind_x_offpeak"] = out["da_wind"] * (1.0 - out["is_peak_hour"])
        out["solar_x_summer_weekend"] = out["da_solar"] * out["is_weekend"] * month_in_summer
        # Explicit summer glut feature for NegativePriceClassifier
        # (negatives cluster in solar hours 10-16 on summer weekends)
        out["summer_solar_weekend"] = out["da_solar"] * out["is_weekend"] * month_in_summer
        out["residual_load_x_hour"] = out["residual_load"] * out["hour_of_day"]
        out["fr_nuclear_x_de_load"] = out["fr_nuclear_avail"] * out["da_load"]
        out["renewable_pen_x_weekend"] = out["renewable_penetration"] * out["is_weekend"]

        for name, rationale in [
            ("month_in_summer", "Apr–Sep summer season flag"),
            ("gas_x_residual_load", "Gas price matters more when residual is high"),
            ("wind_x_offpeak", "Wind cannibalisation in off-peak"),
            ("solar_x_summer_weekend", "Classic negative-price setup"),
            (
                "summer_solar_weekend",
                "da_solar × is_weekend × month_in_summer — summer neg-price setup",
            ),
            ("residual_load_x_hour", "Diurnal residual interaction"),
            ("fr_nuclear_x_de_load", "FR nuclear relief when DE tight"),
            ("renewable_pen_x_weekend", "Weekend renewable glut"),
        ]:
            self._register(name, "F_interactions", rationale, name, out[name])
        return out

    def _group_g_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group G — regime features (hard label, probabilities, Dunkelflaute, solar)."""
        out = df.copy()
        for col, default in [
            ("price_regime", 2),
            ("regime_probability_0", 0.0),
            ("regime_probability_1", 0.0),
            ("regime_probability_2", 1.0),
            ("regime_probability_3", 0.0),
            ("dunkelflaute_day_index", 0),
            ("dunkelflaute_severity", 0),
            ("solar_cannibal_risk", 0.0),
        ]:
            if col not in out.columns:
                out[col] = default

        for name, rationale in [
            ("price_regime", "Hard regime label 0-3"),
            ("regime_probability_0", "P(negative/glut)"),
            ("regime_probability_1", "P(low)"),
            ("regime_probability_2", "P(normal)"),
            ("regime_probability_3", "P(high/Dunkelflaute)"),
            ("dunkelflaute_day_index", "Days into drought event"),
            ("dunkelflaute_severity", "Drought severity 0-3"),
            ("solar_cannibal_risk", "Solar oversupply risk 0-1"),
        ]:
            self._register(name, "G_regime", rationale, name, out[name])
        return out

    def _group_h_structural(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group H — 3 structural break features."""
        out = df.copy()
        if "post_ukraine_war" not in out.columns:
            out["post_ukraine_war"] = (out.index.date >= UKRAINE_WAR_DATE).astype(float)
        else:
            out["post_ukraine_war"] = out["post_ukraine_war"].astype(float)
        if "post_nuclear_phaseout" not in out.columns:
            out["post_nuclear_phaseout"] = (out.index.date >= DE_NUCLEAR_PHASEOUT_DATE).astype(float)
        else:
            out["post_nuclear_phaseout"] = out["post_nuclear_phaseout"].astype(float)

        war_ts = pd.Timestamp(UKRAINE_WAR_DATE, tz="UTC")
        years = (out.index - war_ts).total_seconds() / (365.25 * 24 * 3600)
        out["years_since_ukraine_war"] = np.maximum(0.0, np.asarray(years, dtype=float))

        for name, rationale in [
            ("post_ukraine_war", "Post-2022-02-24 energy crisis regime"),
            ("post_nuclear_phaseout", "Post-2023-04-15 zero DE nuclear"),
            ("years_since_ukraine_war", "Time since structural shock"),
        ]:
            self._register(name, "H_structural", rationale, name, out[name])
        return out

    def _feature_column_list(self, df: pd.DataFrame) -> List[str]:
        """Return ordered unique modelling feature columns."""
        names = [f["name"] for f in self.manifest["features"]]
        seen = set()
        cols: List[str] = []
        for n in names:
            if n not in seen and n in df.columns:
                seen.add(n)
                cols.append(n)
        return cols

    def assert_no_leakage(self, feature_df: pd.DataFrame, target_series: pd.Series) -> None:
        """
        Fail if any feature correlates >0.95 with target.shift(-1).

        Args:
            feature_df: Feature matrix.
            target_series: DA price series.

        Raises:
            DataLeakageError: On suspicious future correlation.
        """
        future = target_series.shift(-1)
        results: Dict[str, float] = {}
        offenders: List[str] = []
        for col in feature_df.columns:
            s = feature_df[col]
            mask = s.notna() & future.notna()
            if mask.sum() < 100:
                results[col] = float("nan")
                continue
            corr = float(s[mask].corr(future[mask]))
            results[col] = corr
            if abs(corr) > LEAKAGE_CORR_THRESHOLD:
                offenders.append(col)

        write_json(
            self.settings.logs / "leakage_check.json",
            {
                "threshold": LEAKAGE_CORR_THRESHOLD,
                "correlations": results,
                "offenders": offenders,
                "pass": len(offenders) == 0,
            },
        )
        if offenders:
            raise DataLeakageError(
                f"Potential leakage: features {offenders} correlate >{LEAKAGE_CORR_THRESHOLD} with future target"
            )
        logger.info("Leakage check PASS — %s features tested", len(results))

    def _finalise_manifest(self, feat: pd.DataFrame) -> None:
        """Update manifest stats from final feature frame."""
        by_name = {f["name"]: f for f in self.manifest["features"]}
        for col in feat.columns:
            if col in by_name:
                s = feat[col]
                by_name[col]["min"] = float(s.min()) if s.notna().any() else None
                by_name[col]["max"] = float(s.max()) if s.notna().any() else None
                by_name[col]["mean"] = float(s.mean()) if s.notna().any() else None
        self.manifest["feature_count"] = len(feat.columns)
        self.manifest["feature_names"] = list(feat.columns)
