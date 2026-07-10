# Methodology — German Day-Ahead Fair Value Forecasting

**Cobblestone Power Analytics v1.0.0**  
Seya Makin | seyamakin04@gmail.com

---

## 1. Why walk-forward, not cross-validation

Electricity prices are a non-stationary, strongly autocorrelated process with regime shifts (Ukraine war, nuclear phase-out, Dunkelflaute). K-fold cross-validation randomly interleaves future hours into the training fold, which:

- leaks future price levels and volatility into the model,
- overstates skill on spike regimes that a desk would not have seen yet,
- fails to mimic the operational constraint: at 11:00 on day *T* you only know history ≤ *T*.

Walk-forward expanding windows enforce that constraint. We require at least 365 days of history, refit weekly, and score the next 168 hours through calendar 2024 (8,784 hours). Metrics are therefore closer to what a production monitor would report.

## 2. Why XGBoost over neural networks for this problem

For tabular European power fundamentals (residual load, lags, calendar, fuels), gradient-boosted trees remain the practical default:

- **Sample efficiency** — three years of hourly data is large for trees, still modest for deep nets without heavy regularisation.
- **Heterogeneous features** — mixes of continuous MW, cyclic encodings, and binary structural breaks suit split-based learners.
- **Operational transparency** — SHAP via `pred_contribs` gives hour-level attribution a trader can challenge (“why is residual_load pushing +€14?”).
- **Quantile heads** — `reg:quantileerror` yields a full predictive distribution without a separate distributional network.

Neural approaches (DNN, distributional nets à la Marcjasz et al.) can win on pure accuracy with careful tuning and weather NWP inputs; they are a natural extension once live weather and intraday features are wired in. They are not the first instrument a desk trusts at 06:45 without a long calibration history.

## 3. Why conformal prediction over quantile regression alone

Quantile regression (including XGB quantile models) estimates conditional quantiles but **does not guarantee** finite-sample coverage. German DA prices are bimodal (negative glut vs Dunkelflaute spikes); Gaussian or even homoscedastic residual assumptions fail exactly when risk matters most.

Split conformal prediction calibrates on absolute residuals \(r_i = |y_i - \hat{y}_i|\) and forms

\[
[\hat{y} - \hat{q},\; \hat{y} + \hat{q}],\quad
\hat{q} = \text{the }\lceil(1-\alpha)(1+1/n)\rceil\text{-th residual quantile.}
\]

We further **condition on price regime**, so quiet hours get tight Conformal 90% PIs while regime-3 hours widen automatically. If a regime has fewer than 50 calibration points, we fall back to the global residual distribution. Dashboard intervals are labelled explicitly as “Conformal 90% PI” — the terminology a quant reviewer expects (O'Connor et al., Energy and AI, 2025).

Quantile models remain useful for pinball scores and reliability diagrams; conformal wraps the point model for coverage-critical trading bands.

## 4. Regime detector design and Nov/Dec 2024 calibration

Four regimes encode the merit-order story:

| ID | Name | Rule (simplified) |
|----|------|-------------------|
| 0 | NEGATIVE/GLUT | renewable penetration &gt; 80%, weekend, Apr–Sep |
| 1 | LOW | residual load &lt; 30 GW |
| 2 | NORMAL | residual 30–55 GW |
| 3 | HIGH/DUNKELFLAUTE | residual &gt; 55 GW and (wind &lt; 5 GW or drought flag) |

Dunkelflaute is detected when \((\mathrm{wind}+\mathrm{solar})/\mathrm{load} &lt; 0.10\) persists for ≥24 consecutive hours, with severity tiers at 10% / 7% / 5%. Soft probabilities from multinomial logistic regression on residual load and renewables enter the XGBoost feature set so the point model can specialise without hard switches.

Verification: the detector is asserted against the known Nov 2–7 and Dec 12–14 2024 events. In synthetic offline mode those windows are embedded in the data generator; with live ENTSO-E data the same checks apply.

## 5. Anti-hallucination guard for gemini-2.0-flash commentary

Market notes that invent prices are a compliance and trading risk. The commentary module:

1. Feeds **only** pipeline-computed metrics (baseload, conformal bounds, residual load, SHAP names, skill, regime risks).
2. Instructs the model never to invent figures.
3. Parses every numeric token in the response and requires each to match an input metric within ±0.5 (with a small allow-list for regime indices 0–3).
4. Sets `contains_hallucination_flag` and surfaces PASS/FAIL on the dashboard.

If `GEMINI_API_KEY` is unset, a deterministic template commentary is used so the pipeline never blocks on LLM availability.

## 6. Limitations — what this model cannot capture

- **Intraday continuous trading** dynamics and ramping constraints after DA clearing  
- **True NWP weather revisions** between forecast publication and delivery  
- **Plant-level outages** beyond ENTSO-E unavailability aggregates  
- **Cross-border capacity auctions** and flow-based market coupling detail  
- **Forward curve shape** (we proxy the “market” with seasonal naive offline)  
- **Behavioural / political** shocks not in fundamentals  

Negative-price and Dunkelflaute skill will degrade if renewable forecast bias shifts structurally (e.g. new offshore build-out) without retraining.

## 7. Path to production readiness

1. **Live ENTSO-E + fuel feeds** on a schedule before DA gate closure  
2. **Weather NWP** (ECMWF/ICON) for wind/solar nowcasts and revision features  
3. **EEX / broker forward curve** as the signal reference instead of naive  
4. **Intraday update loop** after DA results for residual positions  
5. **Position limits, credit, and execution** adapters — the current backtest is explicitly illustrative only  
6. **Model monitoring** — conformal coverage drift alerts by regime  
7. **Human-in-the-loop** for commentary before external distribution  

Until those exist, treat outputs as a research-grade fair-value engine and decision-support dashboard — not an automated execution system.
