"""
Cobblestone Power — forecasting models (Naive, Ridge, XGBoost + quantiles).

Purpose:
    Provide tiered baselines and the main XGBoost point/quantile forecasters
    with Optuna tuning and SHAP explanations.

Inputs:
    Feature matrices and target Series (EUR/MWh).

Outputs:
    Fitted models; predictions; best_params.json; SHAP figures.

Side Effects:
    Writes model artefacts under outputs/models/; figures under validation/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from config.settings import RANDOM_SEED, get_settings
from src.utils import save_figure, write_json

logger = logging.getLogger(__name__)

RIDGE_FEATURES: List[str] = [
    "residual_load",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "price_lag_168h",
    "renewable_penetration",
]

DEFAULT_QUANTILES: List[float] = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
QUANTILE_COL_MAP: Dict[float, str] = {
    0.05: "q05",
    0.10: "q10",
    0.25: "q25",
    0.50: "q50",
    0.75: "q75",
    0.90: "q90",
    0.95: "q95",
}

# Two-stage extreme-event system (heavy-tailed DE prices)
# Fixed 120 EUR was calibrated on synthetic data; real 2022 crisis prices
# averaged ~€235/MWh so a fixed cut labelled most hours as extreme.
# Use the 85th percentile of each walk-forward training window instead.
EXTREME_PRICE_PERCENTILE: float = 85.0
EXTREME_PRICE_HIGH_EUR: float = 120.0  # fallback only when percentile cannot be computed
EXTREME_PRICE_LOW_EUR: float = 0.0
EXTREME_PROB_THRESHOLD: float = 0.6  # route to extreme regressor when P(extreme) > 0.6
EXTREME_FLOOR_PROB: float = 0.7
EXTREME_FLOOR_PRED_EUR: float = 100.0
EXTREME_FLOOR_MULTIPLIER: float = 2.5
MIN_EXTREME_TRAIN_SAMPLES: int = 50
CLASSIFIER_FEATURES: List[str] = [
    "dunkelflaute_day_index",
    "dunkelflaute_severity",
    "renewable_penetration",
    "residual_load",
    "wind_ramp_rate",
    "solar_cannibal_risk",
    "is_weekend",
    "month",
]

# Dedicated negative-price probability module (DE DA ~5.2% of hours in 2024)
NEGATIVE_PRICE_FEATURES: List[str] = [
    "renewable_penetration",
    "wind_solar_combined",
    "is_weekend",
    "solar_x_summer_weekend",
    "summer_solar_weekend",
    "month_in_summer",
    "hour_of_day",
    "month",
    "negative_price_freq_7d",
    "wind_x_offpeak",
]
NEGATIVE_PRICE_FEATURE_NAME: str = "negative_price_probability"
NEGATIVE_PRICE_RISK_THRESHOLD: float = 0.3
# Real 2024: 457/8784 ≈ 5.2% negatives → scale_pos_weight = 94.8/5.2 ≈ 18.2
NEGATIVE_PRICE_DEFAULT_SCALE_POS_WEIGHT: float = 94.8 / 5.2

# Signed log1p transform disabled — was masking the extreme-threshold bug
USE_PRICE_TRANSFORM: bool = False

# Exponential time-weighting for walk-forward XGB fits.
# weight = exp(0.001 × days_from_train_start) → ~2–2.7× more weight on
# post-crisis (2023/24) samples vs early-2022 crisis hours.
TIME_WEIGHT_DECAY: float = 0.001


def exponential_time_weights(
    index: pd.DatetimeIndex,
    rate: float = TIME_WEIGHT_DECAY,
    origin: Optional[pd.Timestamp] = None,
) -> np.ndarray:
    """
    Sample weights that up-weight recent hours.

    ``weight = exp(rate × days_from_origin)`` with origin = first timestamp
    in ``index`` (or ``origin`` if given).

    Args:
        index: Training DatetimeIndex.
        rate: Exponential rate per day (default 0.001).
        origin: Optional fixed origin; defaults to ``index.min()``.

    Returns:
        Float array of weights, same length as ``index``.
    """
    if len(index) == 0:
        return np.array([], dtype=float)
    idx = pd.DatetimeIndex(index)
    start = pd.Timestamp(origin) if origin is not None else idx.min()
    if start.tzinfo is None and idx.tz is not None:
        start = start.tz_localize(idx.tz)
    elif start.tzinfo is not None and idx.tz is not None:
        start = start.tz_convert(idx.tz)
    days = (idx - start).total_seconds() / 86400.0
    return np.exp(rate * np.asarray(days, dtype=float))


def transform_price(y: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    """y_transformed = sign(y) * log(1 + |y|)."""
    if isinstance(y, pd.Series):
        return pd.Series(np.sign(y.values) * np.log1p(np.abs(y.values)), index=y.index, name=y.name)
    arr = np.asarray(y, dtype=float)
    return np.sign(arr) * np.log1p(np.abs(arr))


def inverse_transform_price(y_t: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    """Inverse of signed log1p."""
    if isinstance(y_t, pd.Series):
        return pd.Series(
            np.sign(y_t.values) * np.expm1(np.abs(y_t.values)), index=y_t.index, name=y_t.name
        )
    arr = np.asarray(y_t, dtype=float)
    return np.sign(arr) * np.expm1(np.abs(arr))



class SeasonalNaiveBaseline:
    """
    Seasonal naive: forecast = same-hour price 7 days ago.

    Purpose:
        MAE denominator for skill scores.

    Inputs:
        Historical price Series.

    Outputs:
        Forecast Series aligned to requested index.
    """

    def __init__(self, lag_hours: int = 168) -> None:
        self.lag_hours = lag_hours
        self._history: Optional[pd.Series] = None

    def fit(self, y: pd.Series) -> "SeasonalNaiveBaseline":
        """Store history for lag lookup."""
        self._history = y.copy()
        return self

    def predict(self, index: pd.DatetimeIndex) -> pd.Series:
        """
        Predict seasonal naive values for timestamps in `index`.

        Args:
            index: Forecast timestamps (UTC).

        Returns:
            Series of lagged prices (NaN where history insufficient).
        """
        if self._history is None:
            raise RuntimeError("SeasonalNaiveBaseline must be fit before predict")
        hist = self._history
        vals = []
        for ts in index:
            key = ts - pd.Timedelta(hours=self.lag_hours)
            vals.append(float(hist.loc[key]) if key in hist.index else np.nan)
        return pd.Series(vals, index=index, name="naive")


class RidgeBaseline:
    """
    Ridge regression baseline with StandardScaler on a fixed feature set.

    Purpose:
        Linear skill benchmark; coefficients logged for interpretability.
    """

    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = alpha
        self.model = Ridge(alpha=alpha, random_state=RANDOM_SEED)
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.coefficients_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeBaseline":
        """
        Fit Ridge on available RIDGE_FEATURES intersection.

        Args:
            X: Feature frame.
            y: Target prices.
        """
        cols = [c for c in RIDGE_FEATURES if c in X.columns]
        if not cols:
            raise ValueError("No Ridge features present in X")
        self.feature_names = cols
        mask = X[cols].notna().all(axis=1) & y.notna()
        Xs = self.scaler.fit_transform(X.loc[mask, cols])
        self.model.fit(Xs, y.loc[mask])
        self.coefficients_ = dict(zip(cols, self.model.coef_.tolist()))
        logger.info("Ridge coefficients: %s", self.coefficients_)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict with trained Ridge; NaN rows where features missing."""
        cols = self.feature_names
        out = pd.Series(np.nan, index=X.index, name="ridge")
        mask = X[cols].notna().all(axis=1)
        if mask.any():
            Xs = self.scaler.transform(X.loc[mask, cols])
            out.loc[mask] = self.model.predict(Xs)
        return out


class XGBoostPointForecaster:
    """
    Main XGBoost point forecaster with Optuna tuning, quantiles, and SHAP.

    Purpose:
        Production point estimate for DE day-ahead prices.

    Inputs:
        Full feature matrix; hyperparameters from settings.

    Outputs:
        Predictions; quantile frame; SHAP importances; saved model JSON.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.settings = get_settings()
        base = self.settings.xgb_params.to_dict()
        if params:
            base.update(params)
        # early_stopping_rounds is fit kwarg in recent xgboost
        self.early_stopping_rounds = int(base.pop("early_stopping_rounds", 60))
        self.params = base
        self.model: Optional[XGBRegressor] = None
        self.quantile_models: Dict[float, XGBRegressor] = {}
        self.feature_names_: List[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "XGBoostPointForecaster":
        """
        Fit point XGBoost with optional early stopping on a validation set.

        Args:
            X_train / y_train: Training data.
            X_val / y_val: Optional validation for early stopping.
        """
        self.feature_names_ = list(X_train.columns)
        mask = X_train.notna().all(axis=1) & y_train.notna()
        Xt = X_train.loc[mask]
        yt = y_train.loc[mask]

        fit_params: Dict[str, Any] = {}
        model_params = {**self.params}
        eval_metric = model_params.pop("eval_metric", ["rmse", "mae"])

        self.model = XGBRegressor(**model_params, eval_metric=eval_metric)
        if X_val is not None and y_val is not None:
            vm = X_val.notna().all(axis=1) & y_val.notna()
            fit_params["eval_set"] = [(X_val.loc[vm], y_val.loc[vm])]
            fit_params["verbose"] = False
            try:
                self.model.fit(
                    Xt,
                    yt,
                    early_stopping_rounds=self.early_stopping_rounds,
                    **fit_params,
                )
            except TypeError:
                # xgboost 2.x may want early_stopping_rounds in constructor
                self.model = XGBRegressor(
                    **model_params,
                    eval_metric=eval_metric,
                    early_stopping_rounds=self.early_stopping_rounds,
                )
                self.model.fit(Xt, yt, eval_set=fit_params["eval_set"], verbose=False)
        else:
            self.model.fit(Xt, yt, verbose=False)

        path = self.settings.models_dir / "xgboost_point.json"
        self.model.save_model(str(path))
        logger.info("XGBoost point model saved → %s", path)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Point prediction; NaN where features incomplete."""
        if self.model is None:
            raise RuntimeError("Model not fitted")
        cols = self.feature_names_
        out = pd.Series(np.nan, index=X.index, name="y_pred")
        mask = X[cols].notna().all(axis=1)
        if mask.any():
            out.loc[mask] = self.model.predict(X.loc[mask, cols])
        return out

    def hyperparameter_tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_trials: int = 50,
        walk_forward: bool = True,
    ) -> Dict[str, Any]:
        """
        Optuna TPE search over full XGB hyperparameter space.

        Objective minimises walk-forward MAE on real data (monthly folds on
        the last year of ``X_train``) when ``walk_forward=True``.

        Args:
            X_train / y_train: Tuning panel (typically all history before test year).
            n_trials: Number of Optuna trials (default 50).
            walk_forward: Use multi-fold walk-forward MAE (else 80/20 split).

        Returns:
            Best parameter dict (also saved to best_params.json).
        """
        import optuna
        from optuna.samplers import TPESampler

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Build walk-forward folds: 4 × 28-day forecast windows in the last year
        folds: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        if walk_forward and len(X_train) > 24 * 400:
            end = X_train.index.max()
            for k in range(4):
                f_end = end - pd.Timedelta(days=28 * k)
                f_start = f_end - pd.Timedelta(days=27) + pd.Timedelta(hours=1)
                # Need ≥400 days history before fold
                if f_start - pd.Timedelta(days=400) >= X_train.index.min():
                    folds.append((f_start.floor("h"), f_end.floor("h")))
            folds = list(reversed(folds))
        if not folds:
            split = int(len(X_train) * 0.8)
            folds = [(X_train.index[split], X_train.index[-1])]

        def objective(trial: optuna.Trial) -> float:
            params = {
                **{k: v for k, v in self.params.items() if k not in ("eval_metric", "early_stopping_rounds")},
                "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
                "max_depth": trial.suggest_int("max_depth", 4, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
                "min_child_weight": trial.suggest_int("min_child_weight", 3, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.05, 0.5),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
            }
            maes: List[float] = []
            for f_start, f_end in folds:
                tr_mask = X_train.index < f_start
                te_mask = (X_train.index >= f_start) & (X_train.index <= f_end)
                X_tr, y_tr = X_train.loc[tr_mask], y_train.loc[tr_mask]
                X_te, y_te = X_train.loc[te_mask], y_train.loc[te_mask]
                m = X_tr.notna().all(axis=1) & y_tr.notna()
                mv = X_te.notna().all(axis=1) & y_te.notna()
                if m.sum() < 1000 or mv.sum() < 24:
                    continue
                y_fit = transform_price(y_tr.loc[m]) if USE_PRICE_TRANSFORM else y_tr.loc[m]
                model = XGBRegressor(**params, eval_metric=["rmse", "mae"])
                # Hold out last 10% of train for early stopping
                cut = int(m.sum() * 0.9)
                X_fit = X_tr.loc[m].iloc[:cut]
                y_fit_es = y_fit.iloc[:cut]
                X_es = X_tr.loc[m].iloc[cut:]
                y_es = y_fit.iloc[cut:]
                try:
                    model.fit(
                        X_fit,
                        y_fit_es,
                        eval_set=[(X_es, y_es)],
                        verbose=False,
                    )
                except TypeError:
                    model.fit(X_fit, y_fit_es, verbose=False)
                pred_t = model.predict(X_te.loc[mv])
                pred = inverse_transform_price(pred_t) if USE_PRICE_TRANSFORM else pred_t
                maes.append(float(np.mean(np.abs(pred - y_te.loc[mv].values))))
            return float(np.mean(maes)) if maes else 1e6

        study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=RANDOM_SEED))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        best = study.best_params
        self.params.update(best)
        write_json(
            self.settings.models_dir / "best_params.json",
            {
                "best_params": best,
                "best_mae": study.best_value,
                "n_trials": n_trials,
                "n_folds": len(folds),
                "search": "walk_forward_mae",
            },
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot([t.value for t in study.trials if t.value is not None], marker="o", ms=3)
        ax.set_xlabel("Trial")
        ax.set_ylabel("Walk-forward MAE")
        ax.set_title("Optuna hyperparameter search (real SMARD data)")
        save_figure(fig, self.settings.figures / "validation" / "optuna_history.png")
        plt.close(fig)
        logger.info("Optuna best walk-forward MAE=%.3f params=%s", study.best_value, best)
        return best

    def predict_quantiles(
        self,
        X: pd.DataFrame,
        y_train: Optional[pd.Series] = None,
        X_train: Optional[pd.DataFrame] = None,
        quantiles: Optional[Sequence[float]] = None,
    ) -> pd.DataFrame:
        """
        Fit/predict separate quantile XGBRegressors (reg:quantileerror).

        Args:
            X: Features to predict.
            y_train / X_train: If provided, (re)fit quantile models.
            quantiles: Quantile levels.

        Returns:
            DataFrame with q05..q95 columns.
        """
        quantiles = list(quantiles or DEFAULT_QUANTILES)
        if X_train is not None and y_train is not None:
            self.quantile_models = {}
            mask = X_train.notna().all(axis=1) & y_train.notna()
            for q in quantiles:
                params = {
                    **{k: v for k, v in self.params.items() if k not in ("objective", "eval_metric")},
                    "objective": "reg:quantileerror",
                    "quantile_alpha": q,
                    "n_estimators": min(int(self.params.get("n_estimators", 800)), 800),
                }
                m = XGBRegressor(**params)
                m.fit(X_train.loc[mask], y_train.loc[mask], verbose=False)
                self.quantile_models[q] = m
                qpath = self.settings.models_dir / "xgboost_quantiles" / f"q{int(q*100):02d}.json"
                qpath.parent.mkdir(parents=True, exist_ok=True)
                m.save_model(str(qpath))

        cols = self.feature_names_ or list(X.columns)
        out = pd.DataFrame(index=X.index)
        mask = X[cols].notna().all(axis=1)
        for q in quantiles:
            name = QUANTILE_COL_MAP.get(q, f"q{int(q*100):02d}")
            out[name] = np.nan
            model = self.quantile_models.get(q)
            if model is not None and mask.any():
                out.loc[mask, name] = model.predict(X.loc[mask, cols])
        return out

    def explain(self, X_sample: pd.DataFrame, n_samples: int = 200) -> Dict[str, float]:
        """
        SHAP via pred_contribs; save beeswarm and top-5 dependence plots.

        Args:
            X_sample: Feature sample.
            n_samples: Max rows for SHAP.

        Returns:
            {feature: mean_|shap|} sorted descending.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted")
        cols = self.feature_names_
        sample = X_sample[cols].dropna().iloc[:n_samples]
        if sample.empty:
            return {}

        # pred_contribs via xgboost booster
        booster = self.model.get_booster()
        dm = __import__("xgboost").DMatrix(sample)
        contribs = booster.predict(dm, pred_contribs=True)
        # last column is bias
        shap_vals = contribs[:, :-1]
        mean_abs = np.abs(shap_vals).mean(axis=0)
        importance = {c: float(v) for c, v in sorted(zip(cols, mean_abs), key=lambda x: -x[1])}

        # Beeswarm-like summary
        fig, ax = plt.subplots(figsize=(8, 6))
        top = list(importance.keys())[:20]
        top_idx = [cols.index(c) for c in top]
        for i, ti in enumerate(reversed(top_idx)):
            ax.scatter(
                shap_vals[:, ti],
                np.full(len(sample), i) + np.random.default_rng(RANDOM_SEED).normal(0, 0.1, len(sample)),
                c=sample.iloc[:, ti],
                cmap="coolwarm",
                s=8,
                alpha=0.6,
            )
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(list(reversed(top)), fontsize=8)
        ax.set_xlabel("SHAP value (EUR/MWh)")
        ax.set_title("SHAP summary (pred_contribs)")
        save_figure(fig, self.settings.figures / "validation" / "shap_summary.png")
        plt.close(fig)

        # Dependence for top 5
        for feat in top[:5]:
            fi = cols.index(feat)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(sample[feat], shap_vals[:, fi], s=8, alpha=0.5, c="#00d4ff")
            ax.set_xlabel(feat)
            ax.set_ylabel("SHAP")
            ax.set_title(f"SHAP dependence — {feat}")
            save_figure(fig, self.settings.figures / "validation" / f"shap_dependence_{feat}.png")
            plt.close(fig)

        write_json(self.settings.models_dir / "shap_importance.json", importance)
        return importance


# QRA base-model variants (Nowotarski & Weron, 2018)
QRA_VARIANT_SPECS: Dict[str, Dict[str, Any]] = {
    "conservative": {"max_depth": 4},
    "balanced": {},  # current / default params
    "aggressive": {"max_depth": 8},
}
QRA_VARIANT_ORDER: List[str] = ["conservative", "balanced", "aggressive"]


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    """Pinball (quantile) loss at level tau."""
    e = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.where(e >= 0, tau * e, (tau - 1.0) * e)))


class QuantileRegressionAveraging:
    """
    Quantile Regression Averaging (QRA) — Nowotarski & Weron (2018).

    Purpose:
        Improve probabilistic calibration by combining three XGBoost quantile
        models (conservative / balanced / aggressive) with non-negative
        weights that sum to one, fitted on a calibration window by minimising
        pinball loss.

    Method:
        1. Train 3 ``reg:quantileerror`` XGBRegressors per quantile level.
        2. On the calibration set, solve
           ``min_w  pinball(y, w·q)  s.t.  w ≥ 0,  Σw = 1``.
        3. Final quantile forecast = weighted average of the three base outputs.

    Inputs:
        Feature matrices; target Series; base hyperparams.

    Outputs:
        Averaged quantile DataFrame; per-quantile weights; single-model
        (balanced) forecasts for before/after comparison.

    Side Effects:
        Writes ``outputs/models/qra_weights.json``.
    """

    def __init__(
        self,
        base_params: Optional[Dict[str, Any]] = None,
        quantiles: Optional[Sequence[float]] = None,
        fast: bool = False,
    ) -> None:
        """
        Args:
            base_params: Shared XGB hyperparameters (balanced variant).
            quantiles: Quantile levels to model.
            fast: Fewer trees for walk-forward demo runs.
        """
        self.settings = get_settings()
        self.fast = fast
        params = dict(base_params or self.settings.xgb_params.to_dict())
        params.pop("early_stopping_rounds", None)
        params.pop("eval_metric", None)
        params.pop("objective", None)
        if fast:
            params["n_estimators"] = min(int(params.get("n_estimators", 200)), 150)
        self.base_params = params
        self.quantiles = list(quantiles or DEFAULT_QUANTILES)
        # models[q][variant] = XGBRegressor
        self.models: Dict[float, Dict[str, XGBRegressor]] = {}
        # weights[q] = np.array shape (3,) aligned to QRA_VARIANT_ORDER
        self.weights: Dict[float, np.ndarray] = {}
        self.feature_names_: List[str] = []
        self._fitted: bool = False

    def _variant_params(self, variant: str, tau: float) -> Dict[str, Any]:
        """Build XGB params for a named variant at quantile tau."""
        p = {
            **self.base_params,
            **QRA_VARIANT_SPECS[variant],
            "objective": "reg:quantileerror",
            "quantile_alpha": tau,
            "random_state": RANDOM_SEED,
            "tree_method": self.base_params.get("tree_method", "hist"),
        }
        p["n_estimators"] = min(int(p.get("n_estimators", 800)), 800 if not self.fast else 150)
        return p

    def fit_base_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> "QuantileRegressionAveraging":
        """
        Train the three base quantile models per level on the training window.

        Args:
            X_train / y_train: Training features and prices.

        Returns:
            self
        """
        self.feature_names_ = list(X_train.columns)
        # Allow NaN features (annual-lag warm-up); only require finite target
        mask = y_train.notna() & np.isfinite(y_train.to_numpy(dtype=float))
        Xt, yt = X_train.loc[mask], y_train.loc[mask]
        if len(Xt) < 24:
            raise ValueError(f"QRA needs ≥24 finite training rows, got {len(Xt)}")
        tw = exponential_time_weights(Xt.index)
        self.models = {}

        for tau in self.quantiles:
            self.models[tau] = {}
            for variant in QRA_VARIANT_ORDER:
                model = XGBRegressor(**self._variant_params(variant, tau))
                model.fit(Xt, yt, sample_weight=tw, verbose=False)
                self.models[tau][variant] = model
                qdir = self.settings.models_dir / "xgboost_quantiles" / "qra"
                qdir.mkdir(parents=True, exist_ok=True)
                model.save_model(str(qdir / f"q{int(tau * 100):02d}_{variant}.json"))

        logger.info(
            "QRA base models fitted — %s quantiles × %s variants on %s rows",
            len(self.quantiles),
            len(QRA_VARIANT_ORDER),
            len(Xt),
        )
        return self

    def _base_predictions(self, X: pd.DataFrame, tau: float) -> np.ndarray:
        """Return (n, 3) array of base-model predictions for quantile tau."""
        cols = self.feature_names_
        n = len(X)
        preds = np.full((n, len(QRA_VARIANT_ORDER)), np.nan)
        if tau not in self.models or n == 0:
            return preds
        valid = X[cols].notna().all(axis=1)
        if not valid.any():
            return preds
        idx = np.flatnonzero(valid.to_numpy())
        Xp = X.loc[valid, cols]
        for j, variant in enumerate(QRA_VARIANT_ORDER):
            preds[idx, j] = self.models[tau][variant].predict(Xp)
        return preds

    def calibrate_weights(
        self,
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
    ) -> "QuantileRegressionAveraging":
        """
        Fit non-negative, sum-to-one QRA weights on the calibration window.

        Minimises pinball loss of ``w · q_base`` subject to ``w ≥ 0``, ``Σw = 1``
        via SLSQP (Nowotarski & Weron constrained averaging).

        Args:
            X_cal / y_cal: Calibration features and actuals.

        Returns:
            self
        """
        from scipy.optimize import minimize

        if not self.models:
            raise RuntimeError("Call fit_base_models() before calibrate_weights()")

        mask = X_cal[self.feature_names_].notna().all(axis=1) & y_cal.notna()
        y = y_cal.loc[mask].to_numpy(dtype=float)
        self.weights = {}

        for tau in self.quantiles:
            base = self._base_predictions(X_cal.loc[mask], tau)
            # Drop rows with any NaN base forecast
            ok = np.isfinite(base).all(axis=1) & np.isfinite(y)
            B = base[ok]
            yy = y[ok]
            if len(yy) < 10:
                self.weights[tau] = np.ones(3) / 3.0
                continue

            def objective(w: np.ndarray) -> float:
                return pinball_loss(yy, B @ w, tau)

            cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
            bounds = [(0.0, 1.0)] * 3
            x0 = np.ones(3) / 3.0
            result = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"maxiter": 200, "ftol": 1e-9},
            )
            if result.success:
                w = np.clip(result.x, 0.0, 1.0)
                w = w / w.sum() if w.sum() > 0 else x0
            else:
                logger.warning("QRA weight opt failed for q=%.2f — equal weights", tau)
                w = x0
            self.weights[tau] = w
            logger.info(
                "QRA weights q=%.2f — cons=%.3f bal=%.3f agg=%.3f | cal_pinball=%.4f",
                tau,
                w[0],
                w[1],
                w[2],
                pinball_loss(yy, B @ w, tau),
            )

        write_json(
            self.settings.models_dir / "qra_weights.json",
            {
                f"q{int(t * 100):02d}": {
                    variant: float(self.weights[t][j])
                    for j, variant in enumerate(QRA_VARIANT_ORDER)
                }
                for t in self.quantiles
                if t in self.weights
            },
        )
        self._fitted = True
        return self

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
    ) -> "QuantileRegressionAveraging":
        """
        End-to-end: fit base models on train, calibrate QRA weights on cal.

        Args:
            X_train / y_train: Expanding train window (excl. calibration).
            X_cal / y_cal: Last ~90 days for weight estimation.

        Returns:
            self
        """
        self.fit_base_models(X_train, y_train)
        self.calibrate_weights(X_cal, y_cal)
        return self

    def predict(
        self,
        X: pd.DataFrame,
        return_single: bool = False,
    ) -> pd.DataFrame:
        """
        QRA-averaged quantile forecasts.

        Args:
            X: Feature frame.
            return_single: If True, also include balanced-only columns
                ``q10_single``, ``q50_single``, ``q90_single`` (before QRA).

        Returns:
            DataFrame with ``q05``…``q95`` (QRA) and optionally ``*_single``.
        """
        if not self._fitted:
            raise RuntimeError("QRA must be fit() before predict()")

        out = pd.DataFrame(index=X.index)
        for tau in self.quantiles:
            name = QUANTILE_COL_MAP.get(tau, f"q{int(tau * 100):02d}")
            base = self._base_predictions(X, tau)
            w = self.weights.get(tau, np.ones(3) / 3.0)
            avg = base @ w
            out[name] = avg
            if return_single:
                # Balanced variant = index 1 in QRA_VARIANT_ORDER
                out[f"{name}_single"] = base[:, 1]

        # Enforce non-crossing quantiles (isotonic along quantile axis)
        q_cols = [QUANTILE_COL_MAP[t] for t in sorted(self.quantiles) if QUANTILE_COL_MAP[t] in out.columns]
        if len(q_cols) >= 2:
            arr = out[q_cols].to_numpy()
            # Forward pass: each quantile >= previous
            for j in range(1, arr.shape[1]):
                arr[:, j] = np.fmax(arr[:, j], arr[:, j - 1])
            out[q_cols] = arr

        return out

    def pinball_comparison(
        self,
        y_true: pd.Series,
        qra_df: pd.DataFrame,
        levels: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """
        Before (balanced single) vs after (QRA) pinball at selected levels.

        Args:
            y_true: Actuals.
            qra_df: Output of ``predict(..., return_single=True)``.
            levels: Quantiles to compare (default 0.10, 0.50, 0.90).

        Returns:
            Dict with before/after/delta per level.
        """
        levels = list(levels or [0.10, 0.50, 0.90])
        aligned = pd.concat([y_true.rename("y"), qra_df], axis=1).dropna()
        report: Dict[str, Any] = {"method": "QRA (Nowotarski & Weron, 2018)", "levels": {}}
        for tau in levels:
            col = QUANTILE_COL_MAP.get(tau, f"q{int(tau * 100):02d}")
            single_col = f"{col}_single"
            if col not in aligned.columns:
                continue
            after = pinball_loss(aligned["y"].values, aligned[col].values, tau)
            before = (
                pinball_loss(aligned["y"].values, aligned[single_col].values, tau)
                if single_col in aligned.columns
                else None
            )
            entry = {
                "pinball_after_qra": after,
                "pinball_before_single": before,
                "improvement_pct": (
                    100.0 * (before - after) / before if before and before > 0 else None
                ),
            }
            report["levels"][f"q{int(tau * 100):02d}"] = entry
        return report


def is_extreme_price(
    y: pd.Series,
    high_eur: Optional[float] = None,
) -> pd.Series:
    """
    Label extreme hours: high spikes or negative prices.

    Args:
        y: Price series (EUR/MWh).
        high_eur: Upper threshold. Defaults to ``EXTREME_PRICE_HIGH_EUR``
            (fallback). Prefer the training-window 85th percentile.

    Extreme := price > high_eur OR price < 0 EUR/MWh.
    """
    hi = float(EXTREME_PRICE_HIGH_EUR if high_eur is None else high_eur)
    return (y > hi) | (y < EXTREME_PRICE_LOW_EUR)


def compute_extreme_threshold(
    y_train: pd.Series,
    percentile: float = EXTREME_PRICE_PERCENTILE,
    lookback_days: int = 365,
) -> float:
    """
    Dynamic extreme cut for logging / test-hour labelling: ``percentile`` of
    the last ``lookback_days`` of training prices (current regime).
    """
    vals = y_train.dropna().astype(float)
    if vals.empty:
        return float(EXTREME_PRICE_HIGH_EUR)
    if lookback_days and len(vals) > lookback_days * 24:
        cutoff = vals.index.max() - pd.Timedelta(days=lookback_days)
        recent = vals.loc[vals.index >= cutoff]
        if len(recent) >= 24 * 90:
            vals = recent
    return float(np.nanpercentile(vals.to_numpy(), percentile))


def label_extreme_prices(
    y_train: pd.Series,
    percentile: float = EXTREME_PRICE_PERCENTILE,
) -> tuple[pd.Series, float]:
    """
    Label extreme hours with a **regime-local** percentile.

    Within each calendar year of ``y_train``, mark the top
    ``(100 - percentile)%`` plus all negative prices. This keeps ~15% of
    hours extreme in 2022 *and* in 2024, instead of applying a single cut
    that either (a) flags most of 2022 as extreme (fixed €120) or
    (b) flags most of 2022 when a 2023/24 p85 is applied globally.

    Returns:
        (boolean extreme mask aligned to y_train, latest-year threshold for logging)
    """
    yt = y_train.astype(float)
    parts: List[pd.Series] = []
    latest_thr = float(EXTREME_PRICE_HIGH_EUR)
    for _, g in yt.groupby(yt.index.year):
        g = g.dropna()
        if g.empty:
            continue
        thr = float(np.nanpercentile(g.to_numpy(), percentile))
        latest_thr = thr
        parts.append((g > thr) | (g < EXTREME_PRICE_LOW_EUR))
    if not parts:
        return pd.Series(False, index=y_train.index), latest_thr
    extreme = pd.concat(parts).reindex(y_train.index).fillna(False).astype(bool)
    return extreme, latest_thr


class NegativePriceClassifier:
    """
    Dedicated binary XGBoost classifier for German DA negative prices.

    Standard squared-error regression underestimates negative-price frequency
    because the loss is symmetric. This module predicts P(price < 0) with
    class-imbalance weighting (~93:7) and supplies ``negative_price_probability``
    as a feature to the main regressor.

    Features (leakage-safe, from the engineered panel):
        renewable_penetration, wind_solar_combined, is_weekend,
        solar_x_summer_weekend, hour_of_day, month, negative_price_freq_7d,
        wind_x_offpeak.

    Side Effects:
        Saves ``xgboost_negative_price_classifier.json`` under outputs/models/.
    """

    def __init__(self, fast: bool = False) -> None:
        """
        Args:
            fast: Fewer trees for demo walk-forward runs.
        """
        self.settings = get_settings()
        self.fast = fast
        self.model: Optional[XGBClassifier] = None
        self.feature_names_: List[str] = []
        self.scale_pos_weight_: float = NEGATIVE_PRICE_DEFAULT_SCALE_POS_WEIGHT
        self.n_negative_train_: int = 0
        self.n_train_: int = 0

    @staticmethod
    def _feature_frame(X: pd.DataFrame) -> pd.DataFrame:
        """
        Build the negative-price feature matrix; derive hour/month from index if needed.

        Args:
            X: Full feature DataFrame (DatetimeIndex preferred).

        Returns:
            DataFrame with NEGATIVE_PRICE_FEATURES columns (missing → 0).
        """
        out = pd.DataFrame(index=X.index)
        for col in NEGATIVE_PRICE_FEATURES:
            if col == "month":
                if "month" in X.columns:
                    out["month"] = X["month"]
                elif isinstance(X.index, pd.DatetimeIndex):
                    out["month"] = X.index.month.astype(float)
                else:
                    out["month"] = 0.0
            elif col == "hour_of_day":
                if "hour_of_day" in X.columns:
                    out["hour_of_day"] = X["hour_of_day"]
                elif isinstance(X.index, pd.DatetimeIndex):
                    out["hour_of_day"] = X.index.hour.astype(float)
                else:
                    out["hour_of_day"] = 0.0
            elif col in X.columns:
                out[col] = X[col]
            else:
                out[col] = 0.0
                logger.debug("Negative-price feature %s missing — zero-filled", col)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "NegativePriceClassifier":
        """
        Fit binary classifier for P(price < 0).

        Args:
            X_train: Feature matrix.
            y_train: DA price target (EUR/MWh).

        Returns:
            self
        """
        mask = y_train.notna()
        yt = y_train.loc[mask]
        Xc = self._feature_frame(X_train.loc[mask])
        self.feature_names_ = list(Xc.columns)

        is_neg = (yt < 0).astype(int)
        self.n_negative_train_ = int(is_neg.sum())
        self.n_train_ = int(len(is_neg))
        n_pos = max(self.n_negative_train_, 1)
        n_neg = max(self.n_train_ - self.n_negative_train_, 1)
        empirical = float(n_neg) / float(n_pos)
        # Always use the real-2024 prior 94.8/5.2 ≈ 18.2. Empirical weights
        # drift with expanding windows and under-emphasise rare negatives.
        self.scale_pos_weight_ = float(NEGATIVE_PRICE_DEFAULT_SCALE_POS_WEIGHT)
        if abs(empirical - self.scale_pos_weight_) > 5.0:
            logger.debug(
                "Neg-price empirical weight=%.2f differs from prior=%.2f — using prior",
                empirical,
                self.scale_pos_weight_,
            )

        params = {
            "n_estimators": 280 if not self.fast else 100,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.9,
            "min_child_weight": 5,
            "reg_lambda": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": RANDOM_SEED,
            "tree_method": "hist",
            "scale_pos_weight": self.scale_pos_weight_,
        }
        self.model = XGBClassifier(**params)
        tw = exponential_time_weights(Xc.index)
        self.model.fit(Xc, is_neg, sample_weight=tw, verbose=False)

        path = self.settings.models_dir / "xgboost_negative_price_classifier.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        write_json(
            self.settings.models_dir / "negative_price_classifier_meta.json",
            {
                "features": self.feature_names_,
                "scale_pos_weight": self.scale_pos_weight_,
                "n_negative_train": self.n_negative_train_,
                "n_train": self.n_train_,
                "negative_share_pct": 100.0 * self.n_negative_train_ / max(self.n_train_, 1),
                "risk_threshold": NEGATIVE_PRICE_RISK_THRESHOLD,
                "feature_name": NEGATIVE_PRICE_FEATURE_NAME,
            },
        )
        logger.info(
            "Negative-price classifier fitted — neg_hours=%s (%.1f%%) scale_pos_weight=%.2f",
            self.n_negative_train_,
            100.0 * self.n_negative_train_ / max(self.n_train_, 1),
            self.scale_pos_weight_,
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict P(price < 0) for each row.

        Args:
            X: Feature frame.

        Returns:
            Series named ``negative_price_probability`` in [0, 1].
        """
        if self.model is None:
            raise RuntimeError("NegativePriceClassifier must be fit before predict")
        Xc = self._feature_frame(X)
        proba = self.model.predict_proba(Xc)[:, 1]
        return pd.Series(proba, index=X.index, name=NEGATIVE_PRICE_FEATURE_NAME)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return a copy of ``X`` with ``negative_price_probability`` appended.

        Args:
            X: Feature frame.

        Returns:
            Augmented DataFrame (original columns preserved).
        """
        out = X.copy()
        out[NEGATIVE_PRICE_FEATURE_NAME] = self.predict_proba(X)
        return out

    def high_risk_mask(self, X: pd.DataFrame) -> pd.Series:
        """
        Boolean mask where P(negative) > ``NEGATIVE_PRICE_RISK_THRESHOLD`` (0.3).

        Args:
            X: Feature frame (or already-augmented frame).

        Returns:
            Boolean Series ``high_negative_price_risk``.
        """
        if NEGATIVE_PRICE_FEATURE_NAME in X.columns:
            probs = X[NEGATIVE_PRICE_FEATURE_NAME]
        else:
            probs = self.predict_proba(X)
        return (probs > NEGATIVE_PRICE_RISK_THRESHOLD).rename("high_negative_price_risk")


class TwoStageExtremeForecaster:
    """
    Two-stage forecaster for heavy-tailed German DA prices.

    Stage 1 — binary XGBoost classifier predicts P(extreme hour), where
    extreme := price > training-window 85th percentile OR price < 0.
    Stage 2 — if P(extreme) > 0.6, use a regressor trained only on extreme
    hours; otherwise use the standard (all-hours) regressor.

    Purpose:
        Reduce tail MAE on Dunkelflaute spikes and negative-price gluts
        without harming normal-hour accuracy (published mixture / regime-switch
        approach for electricity price forecasting).

    Inputs:
        Full feature matrix + target; classifier uses CLASSIFIER_FEATURES.

    Outputs:
        Point predictions; extreme probabilities; routing flags.

    Side Effects:
        Saves classifier / extreme / normal model JSON under outputs/models/.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None, fast: bool = False) -> None:
        """
        Args:
            params: Optional XGBRegressor hyperparameter overrides.
            fast: Use fewer trees for demo walk-forward runs.
        """
        self.settings = get_settings()
        self.fast = fast
        base = self.settings.xgb_params.to_dict()
        if params:
            base.update(params)
        if fast and not params:
            # Only apply demo defaults when caller did not pass tuned params
            base["n_estimators"] = min(int(base.get("n_estimators", 200)), 200)
            base["max_depth"] = min(int(base.get("max_depth", 5)), 5)
        self.early_stopping_rounds = int(base.pop("early_stopping_rounds", 60))
        self.params = base
        self.normal_model: Optional[XGBRegressor] = None
        self.extreme_model: Optional[XGBRegressor] = None
        self.classifier: Optional[XGBClassifier] = None
        self.neg_price_clf: Optional[NegativePriceClassifier] = None
        self.feature_names_: List[str] = []
        self.classifier_features_: List[str] = []
        self.n_extreme_train_: int = 0
        self.extreme_model_fitted_: bool = False
        self.extreme_high_eur_: float = float(EXTREME_PRICE_HIGH_EUR)
        self._use_price_transform: bool = False

    @staticmethod
    def _ensure_classifier_frame(X: pd.DataFrame) -> pd.DataFrame:
        """
        Build classifier feature matrix; derive ``month`` from the index if absent.

        Args:
            X: Full feature DataFrame with DatetimeIndex.

        Returns:
            DataFrame with CLASSIFIER_FEATURES columns (missing → 0).
        """
        out = pd.DataFrame(index=X.index)
        for col in CLASSIFIER_FEATURES:
            if col == "month":
                if "month" in X.columns:
                    out["month"] = X["month"]
                elif isinstance(X.index, pd.DatetimeIndex):
                    out["month"] = X.index.month.astype(float)
                else:
                    out["month"] = 0.0
            elif col in X.columns:
                out[col] = X[col]
            else:
                out[col] = 0.0
                logger.debug("Classifier feature %s missing — zero-filled", col)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "TwoStageExtremeForecaster":
        """
        Fit classifier, normal regressor, and extreme-only regressor.

        Args:
            X_train / y_train: Training panel.
            X_val / y_val: Optional validation for early stopping on normal model.

        Returns:
            self
        """
        # —— Stage 0: negative-price probability → regression feature ——
        self.neg_price_clf = NegativePriceClassifier(fast=self.fast)
        self.neg_price_clf.fit(X_train, y_train)
        X_train_aug = self.neg_price_clf.transform(X_train)
        X_val_aug = self.neg_price_clf.transform(X_val) if X_val is not None else None

        self.feature_names_ = list(X_train_aug.columns)
        # Allow NaN features (e.g. price_lag_8736h warm-up) — XGBoost handles missing.
        # Only require a finite target.
        mask = y_train.notna()
        Xt, yt_raw = X_train_aug.loc[mask], y_train.loc[mask]
        # Regime-local extreme labels (~15% per calendar year + negatives)
        extreme, latest_thr = label_extreme_prices(yt_raw)
        self.extreme_high_eur_ = latest_thr
        self.n_extreme_train_ = int(extreme.sum())
        # Log transform disabled (USE_PRICE_TRANSFORM=False)
        yt = transform_price(yt_raw) if USE_PRICE_TRANSFORM else yt_raw
        self._use_price_transform = USE_PRICE_TRANSFORM

        # Exponential time weights: recent (post-crisis) hours count more
        time_w = exponential_time_weights(Xt.index)
        self._time_weight_ratio_ = float(time_w[-1] / max(time_w[0], 1e-12)) if len(time_w) else 1.0

        # —— Stage 1: extreme-event classifier ——
        Xc = self._ensure_classifier_frame(Xt)
        self.classifier_features_ = list(Xc.columns)
        clf_params = {
            "n_estimators": 300 if not self.fast else 120,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 5,
            "reg_lambda": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": RANDOM_SEED,
            "tree_method": "hist",
        }
        # Scale pos weight for class imbalance
        n_pos = max(int(extreme.sum()), 1)
        n_neg = max(int((~extreme).sum()), 1)
        clf_params["scale_pos_weight"] = n_neg / n_pos
        self.classifier = XGBClassifier(**clf_params)
        self.classifier.fit(Xc, extreme.astype(int), sample_weight=time_w, verbose=False)
        clf_path = self.settings.models_dir / "xgboost_extreme_classifier.json"
        self.classifier.save_model(str(clf_path))

        # —— Stage 2a: normal (all-hours) regressor ——
        model_params = {**self.params}
        eval_metric = model_params.pop("eval_metric", ["rmse", "mae"])
        self.normal_model = XGBRegressor(**model_params, eval_metric=eval_metric)
        if X_val_aug is not None and y_val is not None:
            vm = y_val.notna()
            y_val_fit = transform_price(y_val.loc[vm]) if USE_PRICE_TRANSFORM else y_val.loc[vm]
            try:
                self.normal_model.fit(
                    Xt,
                    yt,
                    sample_weight=time_w,
                    eval_set=[(X_val_aug.loc[vm], y_val_fit)],
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose=False,
                )
            except TypeError:
                self.normal_model = XGBRegressor(
                    **model_params,
                    eval_metric=eval_metric,
                    early_stopping_rounds=self.early_stopping_rounds,
                )
                self.normal_model.fit(
                    Xt,
                    yt,
                    sample_weight=time_w,
                    eval_set=[(X_val_aug.loc[vm], y_val_fit)],
                    verbose=False,
                )
        else:
            self.normal_model.fit(Xt, yt, sample_weight=time_w, verbose=False)
        self.normal_model.save_model(str(self.settings.models_dir / "xgboost_point.json"))

        # —— Stage 2b: extreme-only regressor ——
        self.extreme_model_fitted_ = False
        if self.n_extreme_train_ >= MIN_EXTREME_TRAIN_SAMPLES:
            Xe, ye = Xt.loc[extreme], yt.loc[extreme]
            # Magnitude up-weight × exponential time weight
            mag_w = np.maximum(1.0, yt_raw.loc[extreme].abs().to_numpy(dtype=float) / 100.0)
            sample_w = mag_w * time_w[extreme.to_numpy()]
            ext_params = {
                **{k: v for k, v in self.params.items() if k not in ("eval_metric",)},
                "n_estimators": min(int(self.params.get("n_estimators", 800)), 800 if not self.fast else 200),
                "max_depth": min(int(self.params.get("max_depth", 6)) + 1, 8),
                "learning_rate": min(float(self.params.get("learning_rate", 0.04)), 0.06),
            }
            self.extreme_model = XGBRegressor(**ext_params, eval_metric=["rmse", "mae"])
            self.extreme_model.fit(Xe, ye, sample_weight=sample_w, verbose=False)
            self.extreme_model.save_model(
                str(self.settings.models_dir / "xgboost_extreme_regressor.json")
            )
            self.extreme_model_fitted_ = True
        else:
            logger.warning(
                "Only %s extreme training hours (< %s) — extreme regressor disabled; "
                "routing falls back to normal model",
                self.n_extreme_train_,
                MIN_EXTREME_TRAIN_SAMPLES,
            )
            self.extreme_model = None

        logger.info(
            "Two-stage fit — extreme hours=%s (%.1f%%) | classifier saved | extreme_regressor=%s | "
            "neg_price_feature=%s | extreme_threshold=%.1f EUR (p%.0f) | route_prob>%.2f | "
            "price_transform=%s | time_weight_end/start=%.2f (rate=%.4f/day)",
            self.n_extreme_train_,
            100.0 * self.n_extreme_train_ / max(len(yt), 1),
            self.extreme_model_fitted_,
            NEGATIVE_PRICE_FEATURE_NAME,
            self.extreme_high_eur_,
            EXTREME_PRICE_PERCENTILE,
            EXTREME_PROB_THRESHOLD,
            USE_PRICE_TRANSFORM,
            getattr(self, "_time_weight_ratio_", 1.0),
            TIME_WEIGHT_DECAY,
        )
        write_json(
            self.settings.models_dir / "two_stage_meta.json",
            {
                "n_extreme_train": self.n_extreme_train_,
                "extreme_prob_threshold": EXTREME_PROB_THRESHOLD,
                "extreme_high_eur": self.extreme_high_eur_,
                "extreme_percentile": EXTREME_PRICE_PERCENTILE,
                "extreme_low_eur": EXTREME_PRICE_LOW_EUR,
                "classifier_features": self.classifier_features_,
                "extreme_model_fitted": self.extreme_model_fitted_,
                "negative_price_feature": NEGATIVE_PRICE_FEATURE_NAME,
                "negative_price_risk_threshold": NEGATIVE_PRICE_RISK_THRESHOLD,
                "use_price_transform": USE_PRICE_TRANSFORM,
                "time_weight_decay": TIME_WEIGHT_DECAY,
                "time_weight_end_over_start": getattr(self, "_time_weight_ratio_", None),
            },
        )
        return self

    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Append ``negative_price_probability`` using the fitted classifier.

        Args:
            X: Raw feature frame.

        Returns:
            Augmented frame for regressors / QRA.
        """
        if self.neg_price_clf is None:
            raise RuntimeError("TwoStageExtremeForecaster must be fit before transform_features")
        return self.neg_price_clf.transform(X)

    def predict_extreme_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Stage-1 P(extreme) for each row.

        Args:
            X: Feature frame.

        Returns:
            Series in [0, 1] named ``extreme_prob``.
        """
        if self.classifier is None:
            raise RuntimeError("TwoStageExtremeForecaster must be fit before predict")
        Xc = self._ensure_classifier_frame(X)
        proba = self.classifier.predict_proba(Xc)[:, 1]
        return pd.Series(proba, index=X.index, name="extreme_prob")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Two-stage point forecast.

        Routes to extreme regressor when P(extreme) > 0.6 and that model exists.

        Args:
            X: Feature frame.

        Returns:
            Series ``y_pred``.
        """
        meta = self.predict_with_meta(X)
        return meta["y_pred"]

    def predict_with_meta(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict with routing metadata.

        Args:
            X: Feature frame.

        Returns:
            DataFrame with columns:
            ``y_pred``, ``extreme_prob``, ``used_extreme_model``, ``y_pred_normal``,
            ``negative_price_probability``, ``high_negative_price_risk``.
        """
        if self.normal_model is None or self.classifier is None or self.neg_price_clf is None:
            raise RuntimeError("TwoStageExtremeForecaster must be fit before predict")

        X_aug = self.transform_features(X)
        cols = self.feature_names_
        out = pd.DataFrame(index=X.index)
        out["negative_price_probability"] = X_aug[NEGATIVE_PRICE_FEATURE_NAME]
        out["high_negative_price_risk"] = (
            out["negative_price_probability"] > NEGATIVE_PRICE_RISK_THRESHOLD
        )
        out["extreme_prob"] = self.predict_extreme_proba(X)
        out["y_pred_normal"] = np.nan
        out["y_pred"] = np.nan
        out["used_extreme_model"] = False

        # Predict on all rows — XGBoost accepts NaN (annual-lag warm-up)
        mask = pd.Series(True, index=X_aug.index)
        if not mask.any():
            return out

        normal_hat = self.normal_model.predict(X_aug.loc[mask, cols])
        out.loc[mask, "y_pred_normal"] = normal_hat
        out.loc[mask, "y_pred"] = normal_hat

        route = mask & (out["extreme_prob"] > EXTREME_PROB_THRESHOLD)
        if self.extreme_model_fitted_ and self.extreme_model is not None and route.any():
            ext_hat = self.extreme_model.predict(X_aug.loc[route, cols])
            out.loc[route, "y_pred"] = ext_hat
            out.loc[route, "used_extreme_model"] = True

        # Inverse signed-log transform back to EUR/MWh
        if getattr(self, "_use_price_transform", USE_PRICE_TRANSFORM):
            out.loc[mask, "y_pred"] = inverse_transform_price(out.loc[mask, "y_pred"].values)
            out.loc[mask, "y_pred_normal"] = inverse_transform_price(
                out.loc[mask, "y_pred_normal"].values
            )

        # Price-floor correction: high extreme probability but muted prediction
        floor_mask = (
            mask
            & (out["extreme_prob"] > EXTREME_FLOOR_PROB)
            & (out["y_pred"] < EXTREME_FLOOR_PRED_EUR)
            & out["y_pred"].notna()
        )
        if floor_mask.any():
            out.loc[floor_mask, "y_pred"] = out.loc[floor_mask, "y_pred"] * EXTREME_FLOOR_MULTIPLIER

        n_routed = int(out["used_extreme_model"].sum())
        n_high_neg = int(out["high_negative_price_risk"].sum())
        n_floor = int(floor_mask.sum()) if isinstance(floor_mask, pd.Series) else 0
        logger.info(
            "Two-stage predict — %s/%s hours routed to extreme model (threshold=%.2f) | "
            "HIGH neg-price risk hours=%s (P>%.2f) | floor_boost=%s",
            n_routed,
            int(mask.sum()),
            EXTREME_PROB_THRESHOLD,
            n_high_neg,
            NEGATIVE_PRICE_RISK_THRESHOLD,
            n_floor,
        )
        return out

    def explain(self, X_sample: pd.DataFrame, n_samples: int = 200) -> Dict[str, float]:
        """SHAP on the normal-stage model (primary path)."""
        if self.normal_model is None:
            raise RuntimeError("Model not fitted")
        # Reuse XGBoostPointForecaster.explain logic via a thin shim
        shim = XGBoostPointForecaster(params=self.params)
        shim.model = self.normal_model
        shim.feature_names_ = self.feature_names_
        X_aug = self.transform_features(X_sample) if self.neg_price_clf is not None else X_sample
        return shim.explain(X_aug, n_samples=n_samples)
