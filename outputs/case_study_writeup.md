# German Power Fair Value — Day-Ahead Forecasting Pipeline
**Seya Makin** | seyamakin04@gmail.com | github.com/seya-makin
**Market:** DE-LU Day-Ahead | **Submission:** July 2026

---

## Why Germany, Why Now

The German power market in 2024 is the most structurally 
interesting in Europe. Two opposing forces define it:

Renewable cannibalization drove 457 negative-price hours 
in 2024 — 5.2% of the year — as solar penetration hit 
17% of generation. On summer weekends, wind and solar 
simultaneously flood the grid and prices collapse to 
-€500/MWh. A model blind to this dynamic is useless.

Dunkelflaute events are the opposite extreme. In November 
2024 (144 hours) and December 2024 (72 hours), prolonged 
wind-and-solar droughts under high-pressure systems drove 
German intraday prices to €820/MWh and €900/MWh — the 
highest in 18 years. Gas peakers earned over 50% of their 
annual wholesale revenues in those two events alone.

Any fair-value system for the German DA market must 
explicitly model both regimes. This pipeline does.

---

## 1. Data Ingestion and Quality

**Source:** SMARD (smard.de) — Bundesnetzagentur electricity 
market platform. Data retrieved from ENTSO-E under EU 
Regulation 543/2013. Equivalent data quality, no API key 
required, fully public.

**Programmatic API endpoints:**

| Series | Filter ID | Endpoint Pattern | Unit |
|---|---|---|---|
| DA Price DE-LU | 4169 | /chart_data/4169/DE-LU/ | EUR/MWh |
| Total Load | 410 | /chart_data/410/DE/ | MW |
| Wind Onshore | 123 | /chart_data/123/DE/ | MW |
| Wind Offshore | 3791 | /chart_data/3791/DE/ | MW |
| Solar PV | 125 | /chart_data/125/DE/ | MW |

Base URL: https://www.smard.de/app/chart_data/
{filter}/{region}/{filter}_{region}_hour_{ts}.json

Response: [[unix_ms, value], ...] arrays converted to 
UTC-indexed hourly Series.

**Dataset:** 26,304 hours, 2022-01-01 to 2024-12-31.

**Timezone:** UTC internal storage. CET/CEST for display. 
Spring-forward gap interpolated. Autumn duplicate averaged. 
Assert: exactly 8,784 rows for 2024 — passes.

**Structural breaks:**
- post_ukraine_war: 2022-02-24 gas price shock
- post_nuclear_phaseout: 2023-04-15 DE nuclear → 0 MW

**LLM Data QA (Gemini 2.0 Flash):**
20 electricity-market-specific validation rules proposed 
programmatically and executed as Python boolean conditions. 
Rules include: solar zero between 20:00-05:00 CET, load 
bounds 20,000-100,000 MW, price bounds -500 to 3,000 
EUR/MWh, DE nuclear = 0 after 2023-04-15, Dunkelflaute 
detection logic. QA score: 97.1/100 — PASS.

**2024 verified facts:**
- Mean DA price: €78.51/MWh
- Negative hours: 457 (5.2%) — Apr-Sep concentrated
- Dunkelflaute detected: Nov 2-7 (144h), Dec 12-14 (72h)

---

## 2. Forecasting and Model Validation

**Target:** Hourly DA prices for delivery day+1. 
Delivery-period aggregates derived from hourly distribution.

**The fundamental insight:**
German prices are driven by one number: residual load = 
load - wind - solar. High residual load → gas marginal → 
prices spike. Low or negative → renewables marginal → 
prices collapse. Every feature is built around capturing 
this at multiple timescales.

**67 features across 8 groups:**

| Group | Count | Key Features |
|---|---|---|
| Fundamental | 11 | residual_load, renewable_penetration |
| Fuel/Carbon | 7 | clean_spark_spread, dark_spread |
| Calendar | 16 | hour_sin/cos, DE holidays, peak_hour |
| Lags | 8 | price_lag_168h (r=0.71), price_lag_8736h |
| Rolling | 10 | price_roll7d_mean, negative_price_freq_7d |
| Interactions | 6 | solar_x_summer_weekend, wind_x_offpeak |
| Regime | 8 | price_regime 0-3, dunkelflaute_severity |
| Structural | 3 | post_ukraine_war, post_nuclear_phaseout |

Leakage prevention: all lags shifted ≥24h. 
Formal assertion: no feature shows correlation > 0.95 
with future target.

**Results:**

| Model | 2024 MAE | Skill vs Naive |
|---|---|---|
| Seasonal Naive | ~32 EUR/MWh | baseline |
| Ridge Regression | ~29 EUR/MWh | +9% |
| XGBoost + Two-Stage | **25.37 EUR/MWh** | **+22.7%** |

**MAE by year:**

| Year | MAE | Context |
|---|---|---|
| 2022 | 80.18 | Ukraine crisis — prices €200-400/MWh |
| 2023 | 52.54 | Transition year |
| 2024 | **25.37** | Primary benchmark year |

Directional accuracy: **82.9%** — correct price direction 
4 out of every 5 hours across all conditions.

**Two-stage extreme model:**
Binary classifier detects extreme hours (dynamic 85th 
percentile threshold, ≈€93/MWh in 2024). Dedicated 
regressor trained on extreme hours with sample weights 
proportional to price level. 17.2% of 2024 hours routed 
to extreme model.

**Negative price model:**
Binary classifier with scale_pos_weight=18.2 matched to 
real 5.2% negative frequency. Summer recall: 91.1%. 
Full-year recall: 88.0%.

**Validation: 52-window expanding walk-forward**
- Minimum train: 365 days
- Step: 7 days (production cadence)
- Test: full 2024, 8,784 hours
- Zero shuffling, zero future leakage

**Conformal Prediction (O'Connor et al., 2025):**
Regime-conditioned split conformal prediction. 
Distribution-free coverage guarantee — no Gaussian 
assumptions. Critical for German power where prices 
are bimodal. Empirical 90% coverage: **91.0%** ✓

---

## 3. Prompt Curve Translation

**Delivery-period aggregates:**

| Horizon | Definition | Instrument |
|---|---|---|
| Prompt Day Baseload | Mean 24h | DE DA base block |
| Prompt Day Peak | Mean hours 08-20 Mon-Fri | DE DA peak block |
| Prompt Week Base | 7-day mean | EEX Week Base Future |
| Prompt Month Base | Calendar month mean | EEX Month Base Future |

All with 80% and 95% conformal prediction intervals.

**Signal construction:**
LONG  if FV > reference + 0.5 × CP80_width
SHORT if FV < reference - 0.5 × CP80_width
NEUTRAL otherwise
HIGH conviction if CP80 width < 80% of 7-day mean

**Regime-conditional trading:**

Regime 3 (Dunkelflaute): residual load > 55GW, wind 
< 5GW → LONG prompt peak. Buy DE prompt peak on EEX 
or gas generation optionality. Nov 2024 event returned 
>€94,000/MW over 144 hours.

Regime 0 (Glut): renewable penetration > 80%, weekend, 
summer → SHORT prompt baseload. Buy battery 
charge/discharge spread.

**8 invalidation conditions:**
1. Wind forecast revised > 20% post-signal
2. TTF gas moves > €3/MWh overnight
3. French nuclear drops > 2,000 MW unexpectedly
4. Actual DA clearing > 2σ from forecast
5. Residual load changes > 15% post-signal
6. Unplanned outage > 1,000 MW announced
7. Regime classification changes pre-delivery
8. Conformal band expands > 50% vs 7-day mean

---

## 4. AI-Accelerated Workflow

Three Gemini 2.0 Flash integrations. All programmatic. 
All logged to JSONL. All auditable.

**Component 1 — LLM Data QA**
Input: schema + 10 sample rows + statistics
Output: 20 physics-grounded validation rules as Python 
boolean conditions executed against 26,304 rows
Productivity gain: replaces 3 hours of manual rule 
writing with 4-second LLM call
Log: outputs/logs/llm_qa_prompts.jsonl

Example rule: "Solar > 0 between 20:00-05:00 CET 
indicates timezone error (ERROR severity)"

**Component 2 — Daily Market Commentary**
Input: 24 pipeline-computed metrics only
Output: 180-word Bloomberg-style market note
Anti-hallucination guard: every number in commentary 
cross-checked against input metrics (±0.5 tolerance)
Log: outputs/logs/llm_commentary_prompts.jsonl

**Component 3 — Ingestion Config Generation**
Input: SMARD/ENTSO-E field documentation
Output: structured JSON config with column names, 
units, expected ranges, aggregation rules
Eliminates manual hardcoding of field mappings
Log: outputs/logs/llm_config_generation.jsonl

| Component | Logged | Hallucination Check | On Failure |
|---|---|---|---|
| QA Rules | ✓ JSONL | N/A | Log + continue |
| Commentary | ✓ JSONL | ✓ ±0.5 tolerance | Flag + log |
| Config Gen | ✓ JSONL | N/A | Fallback defaults |

---

## Appendix

**Repo:** github.com/seya-makin/cobblestone-power
**Pipeline:** python run_pipeline.py --mode full
**Dashboard:** streamlit run dashboard/app.py
**Submission:** submission.csv — 8,784 rows, 2024 
out-of-sample with 7 quantiles, conformal intervals, 
regime labels, and Dunkelflaute risk scores

**References:**
1. O'Connor et al. (2025). Conformal Prediction for 
Electricity Price Forecasting. Energy and AI, 21.
2. Marcjasz et al. (2023). Distributional neural 
networks for electricity price forecasting. 
Energy Economics, 125.
3. Weron (2014). Electricity price forecasting: 
A review. International Journal of Forecasting, 30(6).
4. Wood Mackenzie (2025). Weathering the lulls: 
risks and opportunities of Dunkelflaute.
5. Timera Energy (2025). Impact of German 
Dunkelflaute on flex asset value.
6. FfE (2026). German electricity prices on 
EPEX Spot in 2025.
