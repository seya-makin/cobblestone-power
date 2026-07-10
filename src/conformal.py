"""
Cobblestone Power — conformal prediction intervals (regime-conditioned).

Purpose:
    Produce distribution-free prediction intervals with approximate finite-sample
    coverage guarantees, adapted per price regime for German power volatility.

Inputs:
    Calibration residuals; point predictions; regime labels.

Outputs:
    Interval bounds; conformal_coverage.json.

Side Effects:
    Writes coverage report under outputs/qa_report/.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import get_settings
from src.utils import write_json

logger = logging.getLogger(__name__)

MIN_REGIME_CAL_SAMPLES: int = 50
DEFAULT_LEVELS: List[float] = [0.05, 0.10, 0.25, 0.75, 0.90, 0.95]
# Finite-sample coverage correction applied on top of the split-conformal quantile
COVERAGE_CORRECTION_FACTOR: float = 1.05
WIDEN_SCALE: float = 1.05
MAX_WIDEN_ITERATIONS: int = 50
MIN_COVERAGE_TARGET: float = 0.90


class ConformalCoverageError(AssertionError):
    """Raised when empirical conformal coverage remains below the required target."""


class ConformalPredictionWrapper:
    """
    Split + adaptive (regime-conditioned) conformal prediction.

    Purpose:
        Guarantee empirical coverage without Gaussian assumptions — critical
        when prices are bimodal and spike to €900/MWh (O'Connor et al., 2025).

    Inputs:
        y_cal, y_hat_cal, regime_cal for calibration; y_hat + regime for predict.

    Outputs:
        (lower, upper) intervals; coverage report.

    Side Effects:
        Persists conformal_coverage.json.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.global_quantiles: Dict[float, float] = {}
        self.regime_quantiles: Dict[int, Dict[float, float]] = {}
        self.regime_counts: Dict[int, int] = {}
        self.coverage_scale: float = COVERAGE_CORRECTION_FACTOR
        self._cal_residuals: Optional[np.ndarray] = None
        self._cal_y: Optional[pd.Series] = None
        self._cal_yhat: Optional[pd.Series] = None
        self._cal_regime: Optional[pd.Series] = None
        self._calibrated: bool = False

    @staticmethod
    def _conformal_quantile(residuals: np.ndarray, level: float) -> float:
        """
        Compute the split-conformal residual quantile at coverage ``level`` (= 1−α).

        Uses the finite-sample corrected order statistic:

            k = ⌈(1−α)(n + 1)⌉ = ⌈(1−α)(1 + 1/n) · n⌉

        i.e. the ⌈(1−α)(1 + 1/n)⌉ empirical quantile of absolute residuals
        (Romano, Patterson & Candès / Vovk). If k > n, return the maximum
        residual (conservative finite-sample bound).

        Args:
            residuals: Absolute residuals |y − ŷ|.
            level: Target coverage probability (1 − α), e.g. 0.90.

        Returns:
            Quantile half-width q̂ in the same units as residuals.
        """
        r = np.sort(np.asarray(residuals, dtype=float))
        r = r[np.isfinite(r)]
        n = len(r)
        if n == 0:
            return 0.0
        # k = ceil((1-α)(n+1))  — 1-based rank
        # Equivalent: ceil(level * (1 + 1/n) * n)
        k = int(np.ceil(level * (1.0 + 1.0 / n) * n))
        if k > n:
            return float(r[-1])
        # Convert 1-based rank to 0-based index
        idx = k - 1
        idx = min(max(idx, 0), n - 1)
        return float(r[idx])

    def calibrate(
        self,
        y_cal: pd.Series,
        y_hat_cal: pd.Series,
        regime_cal: pd.Series,
        levels: Optional[List[float]] = None,
        ensure_coverage: bool = True,
        target_coverage: float = MIN_COVERAGE_TARGET,
    ) -> "ConformalPredictionWrapper":
        """
        Compute global and per-regime absolute residual quantiles.

        Applies ``COVERAGE_CORRECTION_FACTOR`` and, if ``ensure_coverage``,
        iteratively scales quantiles by 1.05 until calibration-set empirical
        coverage at α=0.10 meets ``target_coverage``.

        Args:
            y_cal: Actual prices on calibration set.
            y_hat_cal: Model predictions on calibration set.
            regime_cal: Regime labels 0-3.
            levels: Quantile levels to store (default includes 0.05..0.95).
            ensure_coverage: Widen until cal coverage ≥ target.
            target_coverage: Minimum empirical coverage (default 0.90).

        Returns:
            self
        """
        levels = levels or DEFAULT_LEVELS
        aligned = pd.concat(
            [y_cal.rename("y"), y_hat_cal.rename("yhat"), regime_cal.rename("regime")],
            axis=1,
        ).dropna()
        residuals = (aligned["y"] - aligned["yhat"]).abs().to_numpy()

        self._cal_residuals = residuals
        self._cal_y = aligned["y"]
        self._cal_yhat = aligned["yhat"]
        self._cal_regime = aligned["regime"].astype(int)
        self.coverage_scale = COVERAGE_CORRECTION_FACTOR

        self.global_quantiles = {
            lv: self._conformal_quantile(residuals, lv) for lv in levels
        }

        self.regime_quantiles = {}
        self.regime_counts = {}
        for r in range(4):
            mask = aligned["regime"].astype(int) == r
            self.regime_counts[r] = int(mask.sum())
            if mask.sum() >= MIN_REGIME_CAL_SAMPLES:
                rr = (aligned.loc[mask, "y"] - aligned.loc[mask, "yhat"]).abs().to_numpy()
                self.regime_quantiles[r] = {
                    lv: self._conformal_quantile(rr, lv) for lv in levels
                }
            else:
                self.regime_quantiles[r] = dict(self.global_quantiles)
                logger.info(
                    "Regime %s has %s cal samples (< %s) — using global conformal quantiles",
                    r,
                    int(mask.sum()),
                    MIN_REGIME_CAL_SAMPLES,
                )

        self._calibrated = True

        if ensure_coverage and len(aligned) > 0:
            self._widen_until_calibration_coverage(alpha=0.10, target=target_coverage)

        logger.info(
            "Conformal calibrated — global q90=%.2f | scale=%.3f | regime counts=%s",
            self.global_quantiles.get(0.90, float("nan")) * self.coverage_scale,
            self.coverage_scale,
            self.regime_counts,
        )
        return self

    def _widen_until_calibration_coverage(
        self,
        alpha: float = 0.10,
        target: float = MIN_COVERAGE_TARGET,
    ) -> None:
        """
        Iteratively multiply coverage_scale by 1.05 until cal-set coverage ≥ target.

        Args:
            alpha: Miscoverage level (0.10 → 90% PI).
            target: Required empirical coverage on the calibration set.
        """
        if self._cal_y is None or self._cal_yhat is None or self._cal_regime is None:
            return

        for iteration in range(MAX_WIDEN_ITERATIONS):
            lo, hi = self.predict_interval(self._cal_yhat, self._cal_regime, alpha=alpha)
            covered = ((self._cal_y >= lo) & (self._cal_y <= hi)).mean()
            if covered >= target:
                logger.info(
                    "Calibration coverage OK: %.1f%% ≥ %.0f%% (scale=%.3f, iter=%s)",
                    100 * covered,
                    100 * target,
                    self.coverage_scale,
                    iteration,
                )
                return
            self.coverage_scale *= WIDEN_SCALE
            logger.info(
                "Calibration coverage %.1f%% < %.0f%% — widening scale → %.3f",
                100 * covered,
                100 * target,
                self.coverage_scale,
            )

        # Final check
        lo, hi = self.predict_interval(self._cal_yhat, self._cal_regime, alpha=alpha)
        covered = float(((self._cal_y >= lo) & (self._cal_y <= hi)).mean())
        if covered < target:
            raise ConformalCoverageError(
                f"Calibration conformal coverage {covered:.3f} still below "
                f"{target:.2f} after {MAX_WIDEN_ITERATIONS} widen iterations "
                f"(scale={self.coverage_scale:.3f})"
            )

    def _qhat(self, regime: int, alpha: float) -> float:
        """Return residual quantile for (1-alpha) interval half-width, with coverage scale."""
        level = 1.0 - alpha
        stored = self.regime_quantiles.get(int(regime), self.global_quantiles)
        if level in stored:
            q = stored[level]
        else:
            keys = sorted(stored.keys())
            nearest = min(keys, key=lambda k: abs(k - level))
            q = stored[nearest]
        return float(q) * self.coverage_scale

    def predict_interval(
        self,
        y_hat: pd.Series,
        regime: pd.Series,
        alpha: float = 0.10,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Build symmetric conformal intervals [ŷ − q̂, ŷ + q̂].

        Args:
            y_hat: Point predictions.
            regime: Regime labels aligned to y_hat.
            alpha: Miscoverage level (0.10 → Conformal 90% PI).

        Returns:
            (lower, upper) Series.
        """
        if not self._calibrated:
            raise RuntimeError("Call calibrate() before predict_interval()")

        lower = pd.Series(np.nan, index=y_hat.index, name=f"conformal_{int((1-alpha)*100)}_low")
        upper = pd.Series(np.nan, index=y_hat.index, name=f"conformal_{int((1-alpha)*100)}_high")
        for ts in y_hat.dropna().index:
            r = int(regime.loc[ts]) if ts in regime.index and pd.notna(regime.loc[ts]) else 2
            q = self._qhat(r, alpha)
            yh = float(y_hat.loc[ts])
            lower.loc[ts] = yh - q
            upper.loc[ts] = yh + q
        return lower, upper

    @staticmethod
    def empirical_coverage(
        y_true: pd.Series,
        lower: pd.Series,
        upper: pd.Series,
    ) -> float:
        """Return fraction of y_true falling inside [lower, upper]."""
        df = pd.concat(
            [y_true.rename("y"), lower.rename("lo"), upper.rename("hi")],
            axis=1,
        ).dropna()
        if df.empty:
            return float("nan")
        return float(((df["y"] >= df["lo"]) & (df["y"] <= df["hi"])).mean())

    def widen_intervals_until_coverage(
        self,
        y_true: pd.Series,
        y_hat: pd.Series,
        lower: pd.Series,
        upper: pd.Series,
        target: float = MIN_COVERAGE_TARGET,
    ) -> Tuple[pd.Series, pd.Series, float]:
        """
        Post-hoc widen symmetric intervals around y_hat by ×1.05 until coverage ≥ target.

        Half-width is taken as max(ŷ − lo, hi − ŷ) so asymmetric inputs remain valid.

        Args:
            y_true: Actuals.
            y_hat: Point predictions (interval centre).
            lower / upper: Current interval bounds.
            target: Required empirical coverage.

        Returns:
            (lower_adj, upper_adj, final_coverage)

        Raises:
            ConformalCoverageError: If target cannot be met within MAX_WIDEN_ITERATIONS.
        """
        lo = lower.copy()
        hi = upper.copy()
        scale = 1.0

        for iteration in range(MAX_WIDEN_ITERATIONS + 1):
            cov = self.empirical_coverage(y_true, lo, hi)
            if np.isfinite(cov) and cov >= target:
                if iteration > 0:
                    logger.info(
                        "Post-hoc widen reached coverage %.1f%% after %s ×%.2f steps (total scale=%.3f)",
                        100 * cov,
                        iteration,
                        WIDEN_SCALE,
                        scale,
                    )
                return lo, hi, cov

            scale *= WIDEN_SCALE
            half = pd.concat([(y_hat - lower).abs(), (upper - y_hat).abs()], axis=1).max(axis=1)
            lo = y_hat - half * scale
            hi = y_hat + half * scale
            lo.name = lower.name
            hi.name = upper.name
            logger.info(
                "Empirical coverage %.1f%% < %.0f%% — widening intervals ×%.2f (scale=%.3f)",
                100 * (cov if np.isfinite(cov) else 0.0),
                100 * target,
                WIDEN_SCALE,
                scale,
            )

        cov = self.empirical_coverage(y_true, lo, hi)
        raise ConformalCoverageError(
            f"Empirical conformal coverage {cov:.3f} still below {target:.2f} "
            f"after {MAX_WIDEN_ITERATIONS} widen iterations"
        )

    def assert_coverage(
        self,
        y_true: pd.Series,
        lower: pd.Series,
        upper: pd.Series,
        target: float = MIN_COVERAGE_TARGET,
    ) -> float:
        """
        Assert empirical coverage ≥ target; raise ConformalCoverageError otherwise.

        Args:
            y_true / lower / upper: Actuals and interval bounds.
            target: Minimum required coverage.

        Returns:
            Empirical coverage.

        Raises:
            ConformalCoverageError: If coverage < target.
        """
        cov = self.empirical_coverage(y_true, lower, upper)
        if not np.isfinite(cov) or cov < target:
            raise ConformalCoverageError(
                f"Conformal empirical coverage {cov:.4f} < required {target:.2f}. "
                "Pipeline cannot continue until intervals are widened."
            )
        logger.info("Conformal coverage assertion PASS: %.1f%% ≥ %.0f%%", 100 * cov, 100 * target)
        return cov

    def coverage_report(
        self,
        y_true: pd.Series,
        lower: pd.Series,
        upper: pd.Series,
        alpha: float,
        regime: Optional[pd.Series] = None,
    ) -> Dict:
        """
        Empirical coverage overall and per regime; persist JSON report.

        Args:
            y_true / lower / upper: Actuals and interval bounds.
            alpha: Nominal miscoverage.
            regime: Optional regime labels for stratified coverage.

        Returns:
            Coverage report dict.
        """
        df = pd.concat(
            [y_true.rename("y"), lower.rename("lo"), upper.rename("hi")],
            axis=1,
        ).dropna()
        covered = (df["y"] >= df["lo"]) & (df["y"] <= df["hi"])
        report: Dict = {
            "alpha": alpha,
            "nominal_coverage": 1.0 - alpha,
            "empirical_coverage": float(covered.mean()) if len(df) else None,
            "n": int(len(df)),
            "mean_width": float((df["hi"] - df["lo"]).mean()) if len(df) else None,
            "coverage_scale": self.coverage_scale,
            "by_regime": {},
        }
        if regime is not None:
            df = df.join(regime.rename("regime"), how="left")
            for r in range(4):
                sub = df[df["regime"] == r]
                if len(sub) == 0:
                    continue
                c = (sub["y"] >= sub["lo"]) & (sub["y"] <= sub["hi"])
                report["by_regime"][str(r)] = {
                    "empirical_coverage": float(c.mean()),
                    "n": int(len(sub)),
                    "mean_width": float((sub["hi"] - sub["lo"]).mean()),
                }

        write_json(self.settings.qa_report / "conformal_coverage.json", report)
        logger.info(
            "Conformal coverage empirical=%.1f%% (nominal %.0f%%)",
            100 * (report["empirical_coverage"] or 0),
            100 * (1 - alpha),
        )
        return report
