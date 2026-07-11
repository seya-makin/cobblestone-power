# German Power Fair Value — Day-Ahead Forecasting Pipeline
**Seya Makin** | seyamakin04@gmail.com | github.com/seya-makin/cobblestone-power  
**Market:** DE-LU Day-Ahead | **Submission:** July 2026

---

This pipeline implements a production-grade European power trading stack: SMARD ingestion (no auth), LLM-powered QA, two-stage probabilistic forecasting with conformal uncertainty, curve translation into tradable signals, a REST API, and Docker deployment. The 2024 German DA market is the test case — 457 negative-price hours (5.2%) and Nov/Dec Dunkelflaute power prices reaching €820/MWh and €900/MWh — the highest in 18 years. This pipeline forecasts DA prices across both regimes and translates outputs into tradable signals.

---

## 1. Data Ingestion and Quality

**Source:** SMARD (smard.de) — Bundesnetzagentur electricity market platform. Data retrieved from ENTSO-E under EU Regulation 543/2013. Equivalent data quality, no API key required, fully public.

**Programmatic API endpoints:**

| Series | Filter ID | Endpoint Pattern | Unit |
|---|---|---|---|
| DA Price DE-LU | 4169 | /chart_data/4169/DE-LU/ | EUR/MWh |
| Total Load | 410 | /chart_data/410/DE/ | MW |
| Wind Onshore | 123 | /chart_data/123/DE/ | MW |
| Wind Offshore | 3791 | /chart_data/3791/DE/ | MW |
| Solar PV | 125 | /chart_data/125/DE/ | MW |

Base URL: `https://www.smard.de/app/chart_data/{filter}/{region}/{filter}_{region}_hour_{ts}.json`  
Response: `[[unix_ms, value], ...]` arrays converted to UTC-indexed hourly Series.

**Dataset:** 26,304 hours, 2022-01-01 to 2024-12-31.  
**Timezone:** UTC internal storage; CET/CEST for display. Spring-forward gap interpolated; autumn duplicate averaged. Assert: exactly 8,784 rows for 2024 — passes.

**Structural breaks:** post_ukraine_war (2022-02-24); post_nuclear_phaseout (2023-04-15, DE nuclear = 0 MW).

**LLM Data QA (Gemini 2.0 Flash):** 20 electricity-market-specific validation rules proposed programmatically and executed as Python boolean conditions (solar zero 20:00–05:00 CET, load 20–100 GW, price −500 to 3,000 EUR/MWh, nuclear = 0 after phase-out, Dunkelflaute logic). QA score: 97.1/100 — PASS. This reduces manual rule authoring from hours to seconds while producing domain-specific rules a generalist engineer would miss.

**2024 verified facts:** Mean DA price €78.51/MWh; negative hours 457 (5.2%); Dunkelflaute detected: Nov 2–7 2024 (111 Dunkelflaute hours within 144h window), Dec 12–14 2024 (45 Dunkelflaute hours within 72h window).


---

## 2. Forecasting and Model Validation

**Target:** Hourly DA prices for delivery day+1. Delivery-period aggregates derived from the hourly distribution.

**Fundamental insight:** German prices are driven by residual load = load − wind − solar. High residual load → gas marginal → prices spike. Low or negative → renewables marginal → prices collapse. Features capture this at multiple timescales.

67 features across 8 groups: fundamental supply/demand (11, led by residual_load), fuel/carbon (7, including clean spark spread), calendar with cyclic encoding (16), lag features shifted minimum 24h (8, price_lag_168h is strongest predictor at r=0.71), rolling statistics (10), interaction terms (6), regime features (8, including dunkelflaute_severity), and structural break dummies (3). Leakage assertion: no feature shows correlation above 0.95 with future target.

**Results:**

| Model | 2024 MAE | Skill vs Naive |
|---|---|---|
| Seasonal Naive | ~32 EUR/MWh | baseline |
| Ridge Regression | 37.91 EUR/MWh | -15.5% |
| XGBoost + Two-Stage | **25.37 EUR/MWh (RMSE: 40.0)** | **+22.7%** |

**MAE by year (walk-forward 2022–2024):**

| Year | MAE | Context |
|---|---|---|
| 2022 | 80.18 | Ukraine crisis — prices €200–400/MWh |
| 2023 | 52.54 | Transition year |
| 2024 | **25.37** | Primary benchmark year |

Negative-price recall: **87.7%** of 457 actual negative-price hours correctly flagged as high-risk — directly relevant to renewable cannibalization trading. Directional accuracy: **82.9%**. Two-stage architecture: binary classifier routes 17.2% of hours to an extreme-event regressor trained with price-proportional sample weights; a dedicated negative-price classifier (scale_pos_weight=18.2, summer recall 91.1%, full-year 88.0%) feeds probability as a feature into the main model.

**Validation:** 52-window expanding walk-forward; min train 365 days; step 7 days; test full 2024 (8,784 hours); zero shuffling, zero future leakage.

**Conformal Prediction (O'Connor et al., 2025):** Regime-conditioned split conformal prediction with distribution-free coverage — no Gaussian assumptions. Empirical 90% coverage: **91.0%**. Full 2024 out-of-sample predictions in submission.csv (8,784 rows): point forecast, 7 quantiles, conformal intervals, regime label, and Dunkelflaute risk score per hour.

---

## 3. Prompt Curve Translation

| Horizon | Definition | Instrument |
|---|---|---|
| Prompt Day Baseload | Mean 24h | DE DA base block |
| Prompt Day Peak | Mean hours 08–20 Mon–Fri | DE DA peak block |
| Prompt Week Base | 7-day mean | EEX Week Base Future |
| Prompt Month Base | Calendar month mean | EEX Month Base Future |

All with 80% and 95% conformal prediction intervals.

**Signal:** LONG if fair-value estimate (FV) > reference + 0.5 × CP80_width; SHORT if FV < reference − 0.5 × CP80_width; NEUTRAL otherwise. HIGH conviction if CP80 width < 80% of 7-day mean.

Regime 3 (Dunkelflaute, residual load above 55GW and wind below 5GW): LONG prompt peak. Nov 2024 event drove DA prices to €145/MWh — gas peakers earned a disproportionate share of annual wholesale revenue over 111 hours. Regime 0 (renewable glut, penetration above 80% on summer weekends): SHORT prompt baseload, long battery charge/discharge spread.

**Invalidation:** wind revision >20%; TTF move >€3/MWh overnight; French nuclear drop >2,000 MW; DA clearing >2σ from forecast; residual load change >15%; unplanned outage >1,000 MW; regime flip pre-delivery; conformal band expands >50% vs 7-day mean.

Simplified 2024 backtest: 326 trades, 82.2% win rate, +10,272 EUR/MW P&L (illustrative — no transaction costs).

---

## 4. AI-Accelerated Workflow

Three Gemini 2.0 Flash integrations — all programmatic, logged to JSONL, auditable. Also exposed via FastAPI REST API (6 endpoints, Swagger at /docs) and Docker.

Component 1 — LLM Data QA: Gemini receives schema plus 10 sample rows plus statistics and proposes 20 electricity-market-specific validation rules executed programmatically as Python boolean conditions. Component 2 — Market Commentary: 150–180 word Bloomberg-style note generated from 24 pipeline metrics only, with anti-hallucination guard cross-checking every number at ±0.5 tolerance. Component 3 — Ingestion Config: field documentation converted to structured JSON ingestion config, eliminating manual field mapping hardcoding. All prompts, responses, and outputs logged to JSONL under outputs/logs/.

| Component | Logged | Hallucination Check | On Failure |
|---|---|---|---|
| QA Rules | Yes — JSONL | N/A | Log + continue |
| Commentary | Yes — JSONL | Yes — ±0.5 tolerance | Flag + log |
| Config Gen | Yes — JSONL | N/A | Fallback defaults |

**References:** O'Connor et al. (2025); Marcjasz et al. (2023); Weron (2014); Wood Mackenzie (2025); Timera Energy (2025); FfE (2026).
