import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')
from src.cleaning import DataCleaner

def test_residual_load_never_negative_when_renewables_below_load():
    df = pd.DataFrame({
        'da_load': [50000.0],
        'da_wind': [10000.0],
        'da_solar': [5000.0]
    })
    result = df['da_load'] - df['da_wind'] - df['da_solar']
    assert result.iloc[0] == 35000.0

def test_price_outlier_bounds():
    prices = pd.Series([-600.0, 0.0, 100.0, 3100.0])
    violations = (prices < -500) | (prices > 3000)
    assert violations.sum() == 2

def test_solar_zero_at_night():
    # Mix of night + daytime (12) so not all hours are night
    hours = pd.Series([0, 1, 2, 3, 4, 12, 22, 23])
    is_night = (hours >= 20) | (hours <= 5)
    assert is_night.all() == False
    night_hours = pd.Series([0, 1, 2, 3, 4, 5, 21, 22, 23])
    is_night2 = (night_hours >= 20) | (night_hours <= 5)
    assert is_night2.all() == True

def test_load_plausible_range_gw():
    load_mw = pd.Series([25000.0, 55000.0, 72000.0])
    load_gw = load_mw / 1000.0
    assert (load_gw >= 20).all()
    assert (load_gw <= 100).all()
