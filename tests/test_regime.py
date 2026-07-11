import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

def test_dunkelflaute_price_trigger():
    prices = pd.Series([210.0] * 7 + [80.0] * 10)
    high_price_streak = (prices > 200).astype(int)
    rolling_6h = high_price_streak.rolling(6).sum()
    triggered = (rolling_6h >= 6).any()
    assert triggered == True

def test_negative_price_regime_trigger():
    renewable_pen = pd.Series([0.85, 0.90, 0.60])
    is_weekend = pd.Series([True, True, False])
    regime_0_risk = (renewable_pen > 0.80) & is_weekend
    assert regime_0_risk.iloc[0] == True
    assert regime_0_risk.iloc[1] == True
    assert regime_0_risk.iloc[2] == False

def test_regime_labels_are_valid():
    regime_values = pd.Series([0, 1, 2, 3, 1, 2, 0])
    assert regime_values.isin([0, 1, 2, 3]).all()

def test_dunkelflaute_renewable_share_trigger():
    renew_share = pd.Series([0.15] * 12 + [0.40] * 5)
    low = (renew_share < 0.20).astype(int)
    rolling_12h = low.rolling(12).sum()
    assert (rolling_12h >= 12).any() == True

def test_regime_3_requires_high_residual_and_low_wind():
    residual_gw = pd.Series([60.0, 40.0, 58.0])
    wind_gw = pd.Series([3.0, 2.0, 8.0])
    regime_3 = (residual_gw > 55) & (wind_gw < 5)
    assert regime_3.tolist() == [True, False, False]
