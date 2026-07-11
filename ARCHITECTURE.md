# System Architecture

## Data Flow

```
SMARD API (public, no auth)
    │
    ▼
src/smard_ingestion.py
    │ UTC-indexed hourly Series per variable
    │ Rate-limited with exponential backoff
    │ Saved to data/raw/ as .parquet
    ▼
src/cleaning.py
    │ DST handling (spring-forward interpolated, autumn-fallback averaged)
    │ Per-column imputation policies
    │ Structural break detection (Ukraine war, nuclear phaseout)
    │ Outlier flagging (Z-score, physical bounds, seasonal anomaly)
    ▼
src/llm_qa.py  <── Gemini 2.0 Flash API
    │ 20 physics-grounded rules proposed and executed
    │ Logged to outputs/logs/llm_qa_prompts.jsonl
    ▼
src/regime.py
    │ 4-regime classification (0=Glut, 1=Low, 2=Normal, 3=Dunkelflaute)
    │ Dunkelflaute: (wind+solar)/load < 0.20 for 12h OR price > 200 for 6h
    ▼
src/features.py
    │ 67 features across 8 groups
    │ Leakage assertion: all lags shifted 24h minimum
    │ Saved to data/processed/features.parquet
    ▼
src/model.py
    │ TwoStageExtremeForecaster
    │ NegativePriceClassifier (scale_pos_weight=18.2)
    │ ExtremeEventClassifier (85th percentile threshold)
    │ NormalRegimeXGBoost + ExtremeRegimeXGBoost
    ▼
src/conformal.py
    │ Regime-conditioned split conformal prediction
    │ Volatility-scaled fallback for low-sample regimes (n < 200)
    │ Distribution-free guaranteed coverage
    ▼
src/validation.py
    │ 52-window expanding walk-forward
    │ Weekly refit, 24h forecast horizon
    │ Saved to outputs/forecasts/walk_forward_results.parquet
    ▼
src/curve_translation.py
    │ Hourly to baseload/peak/offpeak aggregates
    │ Prompt day / week / month delivery periods
    │ Signal: LONG/SHORT/NEUTRAL with 8 invalidation conditions
    ▼
src/backtester.py
    │ Fundamental Divergence strategy
    │ 2024 out-of-sample P&L by regime
    ▼
src/llm_commentary.py  <── Gemini 2.0 Flash API
    │ 150-180 word Bloomberg-style market note
    │ Anti-hallucination guard: +-0.5 tolerance check
    │ Logged to outputs/logs/llm_commentary_prompts.jsonl
    ▼
api/main.py                    dashboard/app.py
FastAPI REST endpoints         Streamlit 6-tab dashboard
/forecast /signal /curve       cobblestone-power.streamlit.app
/health /metrics /submission
```

## Engineering Decisions

**Why Parquet not CSV:**
Parquet preserves dtypes including datetime timezone-awareness and float64 precision. Compresses 10x better. Critical for 26,304-row UTC-indexed DataFrames where CSV loses timezone information.

**Why UTC internal storage:**
All SMARD data arrives in CET/CEST. UTC storage eliminates DST ambiguity. The autumn fallback where 02:00 appears twice becomes unambiguous in UTC. Display layer converts back to Europe/Berlin.

**Why walk-forward not cross-validation:**
Cross-validation shuffles data, leaking future prices into training. Walk-forward mimics live deployment: model sees only data available at forecast time. 52 weekly windows give 52 statistically independent test periods.

**Why conformal prediction not quantile regression:**
Quantile regression assumes a well-specified model. Conformal prediction provides distribution-free guaranteed coverage regardless of model misspecification — critical for German power where prices are bimodal and spike 10x during Dunkelflaute.

**Why JSONL for LLM logging:**
Append-only format. Each line is a complete parseable JSON record. Never corrupted by partial writes. Full audit trail: every prompt, response, token count, and generation time. Queryable with jq.

**Why FastAPI not Flask:**
Automatic OpenAPI/Swagger documentation at /docs. Type-safe request/response models via Pydantic. Async support for concurrent requests.

## Testing

18 unit tests across 4 modules — all passing:
- tests/test_cleaning.py — DST handling, outlier detection, price bounds
- tests/test_features.py — lag construction, leakage detection, residual load
- tests/test_regime.py — Dunkelflaute trigger, negative price regime, label validity
- tests/test_conformal.py — coverage guarantee, symmetric intervals

CI: GitHub Actions runs full test suite on every push to main.

## Deployment

Local pipeline: python run_pipeline.py --mode full
Local dashboard: streamlit run dashboard/app.py
Local API: python run_pipeline.py --mode api
Cloud dashboard: cobblestone-power.streamlit.app
