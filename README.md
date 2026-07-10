# German Power Fair Value Forecasting System
**Cobblestone Energy — Graduate Software Engineer Case Study**  
Seya Makin | seyamakin04@gmail.com | github.com/seya-makin

## Market Context

German day-ahead power in 2024–2025 is defined by two opposing forces. On one side, renewable cannibalization: when wind and solar run together on mild weekends, prices collapse toward −€500/MWh as the merit order is flooded and thermal units bid negative to avoid shutdown costs. A model that cannot forecast negative-price hours is useless to a trader who must decide whether to short prompt or buy storage spreads.

On the other side, Dunkelflaute: prolonged wind-and-solar droughts under winter high-pressure systems. In November and December 2024, two such events drove German intraday prices to €820–€900/MWh — the highest in 18 years — and gas peakers earned a disproportionate share of annual wholesale revenue in days. This system is built to detect both regimes early, quantify uncertainty with conformal prediction, and translate hourly fair value into a tradable curve view a desk can open at 06:45 before the DA auction.

## System Architecture

```
ENTSO-E / synthetic ──► Clean + DST ──► LLM QA (gemini-2.0-flash)
                              │
                              ▼
                     Regime Detector (0–3 + Dunkelflaute)
                              │
                              ▼
                     67 Features (leakage-checked)
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
           Naive           Ridge         XGBoost
                                              │
                                              ▼
                                    Conformal PI (regime-aware)
                                              │
                    ┌─────────────────────────┼─────────────────┐
                    ▼                         ▼                 ▼
              Curve / Signal            Walk-forward         Backtest
                    │                         │
                    ▼                         ▼
            LLM Commentary              submission.csv
                    │
                    ▼
            Streamlit Dashboard
```

## What Makes This System Different

1. **Regime-aware modelling** — separate conformal intervals per price regime  
2. **Dunkelflaute detection** with severity scoring and Nov/Dec 2024 verification  
3. **Conformal prediction** for distribution-free uncertainty (O'Connor et al., 2025)  
4. **Three LLM components** via `gemini-2.0-flash`: QA rules, market commentary, config generation  
5. **Simplified trading backtest** translating forecast skill into illustrative P&L  
6. **Solar cannibalization** risk scoring  
7. **Structural break features** for Ukraine war and nuclear phase-out  

## Data Sources

| Source | Dataset | Frequency | ENTSO-E Code | Coverage | Notes |
|--------|---------|-----------|--------------|----------|-------|
| ENTSO-E | Day-ahead prices | Hourly | A44 / 12.1.D | 2022–2024 | DE-LU / DE-AT-LU split |
| ENTSO-E | Load forecast | Hourly | A65 / 6.1.B | 2022–2024 | CTY\|DE |
| ENTSO-E | Wind/solar forecast | Hourly | A69 / 14.1.C | 2022–2024 | B18/B19/B16 |
| ENTSO-E | Unavailability | Hourly | A80 / 15.1.A | 2022–2024 | DE + FR nuclear |
| ENTSO-E | Cross-border flows | Hourly | 12.1.G | 2022–2024 | 8 neighbours |
| ENTSO-E | Actual generation | Hourly | 16.1.B&C | 2022–2024 | Regime diagnostics |
| Synthetic fuels | TTF / EUA / coal | Hourly | — | Demo | Replace with live feeds |

If `ENTSOE_API_KEY` is still `placeholder_add_when_received`, ingestion skips live calls and builds a physically plausible synthetic panel (including Nov/Dec 2024 Dunkelflaute spikes) so the full pipeline runs offline.

## Features (67 total)

| Group | Count | Examples | Leakage-safe? |
|-------|-------|----------|---------------|
| A Fundamental | 11 | residual_load, renewable_penetration | Yes |
| B Fuel/carbon | 7 | ttf_gas_price, clean_spark_spread (lagged) | Yes |
| C Calendar | 16 | hour_sin/cos, is_peak_hour, DE holidays | Yes |
| D Lags | 8 | price_lag_24h…336h | Yes (≥24h) |
| E Rolling | 10 | price_roll7d_* shifted 24h | Yes |
| F Interactions | 6 | solar_x_summer_weekend | Yes |
| G Regime | 8 | price_regime, dunkelflaute_severity | Yes |
| H Structural | 3 | post_ukraine_war, post_nuclear_phaseout | Yes |

See `data/processed/features_manifest.json` after a run for per-feature rationale and stats.

## Models

- **Seasonal Naive** — same hour 7 days ago (skill denominator)  
- **Ridge (α=10)** — linear benchmark on residual load + calendar + lag-168  
- **XGBoost** — main point model; Optuna-tunable; quantile models via `reg:quantileerror`  
- **Conformal prediction** — split conformal on absolute residuals, **conditioned on regime**, so intervals widen in Dunkelflaute and tighten in quiet hours without assuming Gaussian errors  

## Validation Methodology

Walk-forward expanding window (min 365 days train), weekly refit, forecast the next 7 days, through all of 2024 (8,784 hours). Random train/test splits are wrong for prices: they leak future regime structure. Metrics include MAE, skill vs naive/ridge, pinball, Winkler, conformal coverage, and regime-stratified MAE.

## Results

Populate after `python run_pipeline.py --mode full` from `outputs/forecasts/walk_forward_metrics.json`. Headline fields: MAE, skill vs naive, conformal 90% coverage, Dunkelflaute-window errors, negative-price recall.

## Regime Analysis

| Regime | Trigger | Typical price |
|--------|---------|---------------|
| 0 NEGATIVE/GLUT | High renewables + weekend + summer | −€500 to €10 |
| 1 LOW | Residual &lt; 30 GW | €10–60 |
| 2 NORMAL | Residual 30–55 GW | €60–150 |
| 3 HIGH/DUNKELFLAUTE | Residual &gt; 55 GW + low wind / drought | €150–€900+ |

Dunkelflaute module requires (wind+solar)/load &lt; 10% for ≥24 consecutive hours; verified against Nov 2–7 and Dec 12–14 2024.

## Curve Translation

Hourly forecasts aggregate to baseload / peak / off-peak for tomorrow, next week, and next month, each with Conformal 80% and 95% bands. Trading signals compare model fair value to a reference (seasonal naive proxy offline; live would use the prompt curve) with eight explicit invalidation conditions.

## AI Components

1. **LLM QA (`gemini-2.0-flash`)** — proposes 20 physically grounded validation rules; executed with safe `eval`; audited in `outputs/logs/llm_qa_prompts.jsonl`  
2. **LLM Commentary** — Bloomberg-style ≤180-word note from pipeline metrics only; anti-hallucination number check  
3. **LLM Config Generation** — ENTSO-E field docs → structured ingestion JSON  

## Trading Backtest

**Disclaimer:** Simplified research backtest. Does not account for bid-ask spreads, market impact, position limits, counterparty credit, or actual EPEX trading constraints. For illustrative purposes only.

Strategy: Fundamental Divergence — trade next-day peak when \|model − naive\| &gt; €8/MWh; P&amp;L vs realised peak. Regime 3 trades are expected to dominate profitability.

## Setup

```bash
git clone <repo> && cd cobblestone-power
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Optional: set ENTSOE_API_KEY and GEMINI_API_KEY in .env

python run_pipeline.py --mode full          # end-to-end (uses synthetic data if no ENTSO-E key)
streamlit run dashboard/app.py              # trader dashboard
```

Modes: `ingest | clean | qa | regime | features | validate [--resume] | forecast | backtest | commentary | submission | dashboard`.

## Limitations and Future Work

- No live NWP weather; renewable forecasts are ENTSO-E day-ahead only  
- Fuel prices are synthetic in offline mode  
- No intraday continuous trading features  
- Forward curve integration (EEX) would replace the naive reference in signal construction  
- Production would add real-time feeds, position limits, and credit checks  

## References

1. O'Connor et al. (2025). Conformal Prediction for Electricity Price Forecasting. Energy and AI.  
2. Lipiecki et al. (2025). Stealing Accuracy: THieF for Day-ahead Electricity Prices.  
3. Marcjasz et al. (2023). Distributional neural networks for electricity price forecasting.  
4. Weron (2014). Electricity price forecasting: A review of the state-of-the-art.  
5. Timera Energy (2025). Impact of German Dunkelflaute on flex asset value.  
6. Wood Mackenzie (2025). Weathering the lulls: risks and opportunities of Dunkelflaute.  
7. FfE (2026). German electricity prices on the EPEX Spot exchange in 2025.  
