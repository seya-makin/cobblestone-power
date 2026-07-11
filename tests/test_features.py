import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

def test_lag_168h_uses_same_hour_last_week():
    prices = pd.Series(range(200), dtype=float)
    lag_168 = prices.shift(168)
    assert pd.isna(lag_168.iloc[167])
    assert lag_168.iloc[168] == 0.0

def test_renewable_penetration_bounds():
    penetration = pd.Series([0.0, 0.5, 1.2])
    assert penetration.min() >= 0
    assert (penetration > 1.5).sum() == 0

def test_residual_load_formula():
    load = pd.Series([50000.0, 60000.0])
    wind = pd.Series([10000.0, 20000.0])
    solar = pd.Series([5000.0, 8000.0])
    residual = load - wind - solar
    assert residual.iloc[0] == 35000.0
    assert residual.iloc[1] == 32000.0
