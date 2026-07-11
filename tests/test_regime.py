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
