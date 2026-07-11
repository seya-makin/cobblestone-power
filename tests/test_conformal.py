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
