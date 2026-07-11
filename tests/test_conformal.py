import numpy as np
import sys
sys.path.insert(0, '.')

def test_conformal_coverage_logic():
    np.random.seed(42)
    y_true = np.random.normal(100, 20, 1000)
    y_hat = y_true + np.random.normal(0, 5, 1000)
    residuals = np.abs(y_true - y_hat)
    alpha = 0.10
    n = len(residuals)
    q_hat = np.quantile(residuals, np.ceil((1 - alpha) * (n + 1)) / n)
    lower = y_hat - q_hat
    upper = y_hat + q_hat
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    assert coverage >= 0.85

def test_conformal_interval_is_symmetric():
    y_hat = 100.0
    q_hat = 15.0
    lower = y_hat - q_hat
    upper = y_hat + q_hat
    assert upper - y_hat == y_hat - lower

def test_volatility_scaled_quantile_widens_for_higher_std():
    q_global = 50.0
    std_global = 20.0
    std_regime = 80.0
    q_regime = q_global * (std_regime / std_global)
    assert q_regime == 200.0
    assert q_regime > q_global

def test_conformal_finite_sample_quantile_rank():
    residuals = np.arange(1, 101, dtype=float)
    alpha = 0.10
    n = len(residuals)
    k = int(np.ceil((1 - alpha) * (n + 1)))
    assert k == 91
    q_hat = residuals[k - 1]
    assert q_hat == 91.0
