# Project State

**Last Updated:** 2024-05-24

## Completed
- Initialised project skeleton matching the exact folder structure specified in `MASTER_CONTEXT.md`.
- Configured `pyproject.toml` using `uv` with all required dependencies (`pandas`, `pandas-ta`, `ccxt`, `yfinance`, `duckdb`, `scikit-learn`, `xgboost`, `lightgbm`, `vectorbt`, `mlflow`, `optuna`, `pytest`, `ruff`).
- Updated `.gitignore` to exclude `.env`, `data/`, `mlruns/`, `__pycache__/`, and other standard artefacts.
- Created a reusable pytest helper (`tests/test_no_lookahead.py`) to enforce strict temporal boundaries and prevent lookahead bias in feature calculations.
- **Implemented `src/data_engine/crypto_fetcher.py`**: Full OHLCV ingestion pipeline with ccxt pagination, clock-skew alignment, duplicate removal, gap classification (outage/illiquid/data_error), and outlier flagging (magnitude & OHLC integrity). Includes atomic parquet storage to prevent corruption.
- **Implemented `src/data_engine/macro_fetcher.py`**: Macro-economic data ingestion via yfinance (with deterministic scale corrections for yield indices) and CoinGecko for BTC dominance. Handles retries, timezone alignment, and atomic parquet merging.
- **Implemented comprehensive unit tests** (`tests/test_data_engine.py`) covering gap classification logic, outlier detection, OHLC integrity checks, and strict no-lookahead temporal boundary enforcement.
- **Implemented live smoke tests** (`tests/test_binance_real_data.py`) to validate the full pipeline against real Binance network data, checking history depth, index monotonicity, and cleaning/classification sanity checks.

## Current Issues
- Feature engineering, predictive core, backtesting scaffold, risk/sizing, and monitoring modules remain as empty placeholders awaiting implementation.
- Live smoke tests require manual execution (`pytest -v -s`) and are not integrated into CI due to network dependency and rate limits.
- No automated scheduling or cron-job integration is yet in place for the daily macro/CoinGecko data appends.

## Next Steps
1. **Feature Engineering**: Develop no-lookahead feature calculators using `pandas-ta`, validated against the temporal boundary helper.
2. **Predictive Core**: Build the k-NN analog engine, gradient boosting models (XGBoost/LightGBM), and linear ensemble meta-model.
3. **Backtesting Scaffold**: Set up walk-forward validation using `vectorbt`, ensuring train/validation/test folds never overlap.
4. **Macro/MTF/Volatility Logic**: Implement regime classification, multi-timeframe alignment scoring, and volatility state detection per the exact tables in `MASTER_CONTEXT.md`.
5. **Risk & Sizing**: Code quarter-Kelly sizing with hard execution-layer caps (2% single-trade, 6% correlated, 10% aggregate).
6. **Monitoring & Kill-Switches**: Develop live monitoring for calibration drift, feature drift, and unconditional halt triggers requiring manual sign-off.
7. **Testing & Validation**: Expand `pytest` coverage to include regime classification accuracy, risk gate enforcement, and backtesting logic validation.


+## Session Update: Feature Engineering Implementation (2024-05-24)
+
+### Completed Work
+- Successfully architected and implemented the complete Feature Engineering module
(`src/features/feature_calculator.py`) covering Phases 1 through 6.
+- **Phase 1 (Foundation):** Established strict input validation, chronological sorting enforcement, and duplicate
timestamp rejection to guarantee temporal integrity before any calculations proceed.
+- **Phase 2 (Structural Features):** Implemented Higher High/Lower Low flagging (N=20/50), Break-of-Structure (BOS),
Change-of-Character (ChoCh), Swing Failure Patterns (SFP), range-vs-trend classification, distance-to-support/resistance
metrics, and Wyckoff volume-range phase detection.
+- **Phase 3 (Volatility Features):** Integrated ATR(14) percentile ranking, Bollinger Band width normalisation,
20-period rolling volatility, precise state classification (Compressed/Expanding/Extended per `MASTER_CONTEXT.md`
thresholds), and a robust tail-risk Spike detector.
+- **Phase 4 (Momentum Features):** Developed RSI(14) with rate-of-change, vectorised bullish/bearish divergence
proxies, MACD histogram slope, StochRSI, ADX/DI spread, and multi-horizon price returns (5/10/20 periods).
+- **Phase 5 (Volume Features):** Calculated volume z-scores, volume-on-breakout ratios, candle body ratios, and
upper/lower wick magnitude/direction metrics.
+- **Phase 6 (Normalisation & Output Formatting):** Enforced a strict trailing 252-candle percentile ranking pipeline
for all continuous features, ensuring cross-asset scale comparability (BTC vs ETH) while preserving binary flags and
state classifications. Implemented explicit index realignment post-`pandas-ta` calculations to prevent silent temporal
misalignment.
+
+### Problems Encountered & Resolutions
+- **Temporal Leakage Risk:** Initial rolling window logic lacked explicit chronological sorting guarantees, risking
lookahead bias if upstream data was unsorted. *Resolution:* Added strict `is_monotonic_increasing` checks and automatic
index sorting in `_validate_input` before any feature calculation proceeds.
+- **Floating-Point Precision in Pivot Detection:** Exact equality (`==`) for HH/LL flags caused missed structural
pivots due to exchange micro-decimal variations and resampling aggregation. *Resolution:* Replaced with `np.isclose()`
using a deterministic epsilon tolerance (`1e-6`).
+- **Percentile Ranking Exclusion Flaw:** Heuristic prefix matching incorrectly excluded continuous structural metrics
(e.g., distance-to-resistance) from normalisation, violating the "no raw values leave this module" constraint.
*Resolution:* Refactored exclusion logic to target only binary flags, state classifications, and already-ranked columns
via deterministic pattern matching.
+- **BOS/ChoCh Signal Dilution:** Original BOS logic compared closes against immediate prior candles, generating
excessive false positives that would dilute signal quality for the k-NN engine. *Resolution:* Recalibrated to reference
N=20/N=50 structural pivots, aligning with professional SMC/ICT standards and reducing noise.
+- **Pandas-ta Index Misalignment:** `pandas_ta` occasionally returns DataFrames with slightly shifted indices during
internal resampling or NaN-dropping. *Resolution:* Implemented explicit `.reindex(df.index)` safeguards after every
feature category calculation to guarantee strict 1:1 temporal mapping before concatenation.
+- **Warm-up NaN Propagation:** Spike detection and rolling calculations introduced leading NaNs that could break
downstream shape expectations or trigger unnecessary warnings. *Resolution:* Explicitly filled leading warm-up NaNs with
`0.0` (no spike/neutral state) and documented behaviour, ensuring consistent DataFrame shapes without distorting
early-window distributions.
+
+### Remaining Challenges & Open Items
+- **Predictive Core Implementation:** The k-NN analog engine, gradient boosting models (XGBoost/LightGBM), and linear
ensemble meta-model remain unimplemented.
+- **Backtesting Scaffold:** Walk-forward validation configuration via `vectorbt` with strict non-overlapping folds is
pending.
+- **Macro/MTF/Volatility Logic Integration:** Regime classification, multi-timeframe alignment scoring, and volatility
state detection need to be wired into the live trading gates per exact specification tables.
+- **Risk & Sizing Engine:** Quarter-Kelly sizing logic with hard execution-layer caps (2%/6%/10%) requires
+- **Macro/MTF/Volatility Logic Integration:** Regime classification, multi-timeframe alignment scoring, and volatility
state detection need to be wired into the live trading gates per exact specification tables.
+- **Risk & Sizing Engine:** Quarter-Kelly sizing logic with hard execution-layer caps (2%/6%/10%) requires
implementation.