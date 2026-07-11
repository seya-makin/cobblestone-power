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

def test_price_lag_24h_alignment():
    prices = pd.Series(range(48), dtype=float)
    lag_24 = prices.shift(24)
    assert pd.isna(lag_24.iloc[23])
    assert lag_24.iloc[24] == 0.0
    assert lag_24.iloc[47] == 23.0

def test_cyclic_hour_encoding_bounds():
    hour = pd.Series([0, 6, 12, 18])
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    assert (hour_sin >= -1).all() and (hour_sin <= 1).all()
    assert (hour_cos >= -1).all() and (hour_cos <= 1).all()
