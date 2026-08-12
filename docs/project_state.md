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

## Session Update: Predictive Core Review, Backtesting Engine Build, and Predictive Core Strategy Adapter (2026-08-12)

### Completed Work

- **Reviewed the uncommitted `src/predictive_core/` changes** (`labeling.py`, `splitter.py`, `analog_engine.py`,
`boosted_core.py`, `calibration.py`) against four specific criteria: (1) whether test-fold data could leak into
training or hyperparameter tuning, (2) whether the k-NN independence deduplication actually deduplicates overlapping
windows from the same directional move or only appears to, (3) whether direction-classifier calibration is actually
checked anywhere and reports a real number, (4) whether the minimum-50-sample gate can be bypassed by any code path.
No leakage found in Component B's Optuna hyperparameter search (train/val only, test never in scope of the tuning
closures). Calibration confirmed genuinely implemented (isotonic regression fit on validation predictions only, real
ECE/Brier numbers reported per class on the test fold). The minimum-sample gate confirmed unbypassable (`predict()` is
the only method able to produce an `AnalogForecast`, and probability fields are structurally `None` unless the gate
passes).
- **Built the complete walk-forward backtesting engine in `src/backtesting/`** on top of `vectorbt`, satisfying Build
Plan step 1D (Layer 14, basic backtesting scope):
  - `windows.py` — calendar-based walk-forward fold generator (2-year train / 6-month validation / 6-month test,
    rolling forward 3 months, minimum 4 complete folds enforced by raising `ValueError` rather than silently returning
    fewer). Deliberately distinct from `predictive_core.splitter.WalkForwardSplitter`, which is a row-count splitter
    for ML cross-validation, not a calendar-time splitter for strategy backtesting.
  - `costs.py` — `TransactionCostModel`, fit exclusively on each fold's train partition: 75th-percentile historical
    spread (proxied from candle high-low range, since no L2/quote data is ingested by this system), taker fee,
    slippage scaling with both order size and candle range, and weekend/overnight gap-frequency cost applied per
    trade based on how many gap-prone bars the holding period spans.
  - `metrics.py` — Sharpe (bar-level and trade-level variants), profit factor, max drawdown, EV per trade, all
    designed to only ever consume already cost-adjusted inputs.
  - `kill_criteria.py` — all five Layer 14 research-phase kill criteria (OOS Sharpe < 0.5, OOS profit factor < 1.2,
    OOS max drawdown > 25%, Risk-On regime performance worse than aggregate, ensemble weight instability across > 2
    consecutive folds) reported as independent `pass`/`fail`/`not_evaluated` results, never collapsed into one
    aggregate boolean, and explicitly labeled throughout as distinct from the separate, stricter Tier 1 live-trading
    gate. The IS/OOS 60%-Sharpe-degradation overfitting check is implemented separately and clearly labeled as a
    diagnostic, not a sixth Layer 14 criterion.
  - `harness.py` — `WalkForwardBacktester`, the orchestrator wiring windows + cost model + `vectorbt` order simulation
    + metrics + kill criteria into a per-fold and aggregate `BacktestReport`. Structurally never passes a fold's test
    partition to a strategy's `fit()` method.
  - Added a `backtesting:` section to `config.yaml` documenting every default (windows, cost model, kill-criteria
    thresholds), consistent with the code's actual hardcoded defaults, though not yet wired to a config loader.
  - Added `tests/test_backtesting.py`, verified against a full end-to-end run at the literal MASTER_CONTEXT window
    sizes on ~4.5 years of synthetic hourly data.
- **Reviewed the backtesting engine** against five specific criteria: (1) any leakage between fold boundaries, (2)
whether transaction costs are genuinely applied before Sharpe/EV calculation or accidentally after, (3) whether the
Layer 14 kill criteria actually halt further processing or only log a warning, (4) whether the 60% IS/OOS degradation
check uses the correct pair of numbers, (5) whether the five criteria are clearly labeled as Layer 14 (research-phase)
and not conflated with the Tier 1 (live-trading) gate. Confirmed correct on points 1, 4, and 5. Point 2 was confirmed
correct and additionally hand-verified: built a 4-bar, 2-trade synthetic segment, independently derived every number
(execution prices, fees, slippage, net PnL, EV, profit factor, Sharpe) in isolation from the code, and cross-checked
against the actual code path — every number matched to floating-point precision.
- **Explained Build Plan step 1E (Go/No-Go review) to the user** without writing code, after extracting the actual
text of `architecture/Build Plan.docx` to ground the explanation precisely rather than guessing: 1E is an explicit
human judgment call (not an AI task) to check the five Layer 14 criteria from 1D's backtest output and decide
whether to proceed to Phase 2 or revise the hypothesis (different asset, timeframe, or setup definition) and repeat
1C–1D. Flagged that passing 1E means "worth building on," not "ready for capital" — that bar is the separate, stricter
Tier 1 gate checked later at Phase 5.
- **Built `src/backtesting/predictive_core_strategy.py`** — `PredictiveCoreStrategy`, an adapter wiring the real
Component A (`HistoricalAnalogEngine`) and Component B (`train_direction_classifier`, `train_quantile_regressor`)
into `WalkForwardBacktester`'s `SignalStrategy` protocol (`fit`/`generate_positions`/`ensemble_weights`). Built
against the actual current interfaces (traced from the real code, not guessed), handling: Component B's training
functions requiring a mandatory `test_df` argument that must never be the harness's real test partition (val_df is
passed as a placeholder, and every test-specific field on the returned result is discarded as meaningless); the
direction classifier's raw `.predict()` output being uncalibrated softmax (calibrated probabilities require manually
reapplying the per-class isotonic calibrators attached to the model as `.calibrators_` and renormalizing); avoiding
`WalkForwardTrainer` (boosted_core's own orchestrator, which uses an incompatible row-count fold splitter) in favor of
calling `train_direction_classifier`/`train_quantile_regressor` directly on the harness-provided train/val partitions;
and purging the trailing `label_horizon` rows at both the train/val and val/[next] boundaries to stop triple-barrier
labels (which look up to `max_candles` bars forward) from leaking across fold lines.

### Problems Encountered & Resolutions

- **Cross-asset and under-separation bug in `HistoricalAnalogEngine._deduplicate`.** The independence-enforcement
logic used raw row position in the reference frame as a proxy for temporal separation. Empirically confirmed (via a
dedicated verification script, not just code reading) that when fit on a pooled multi-asset frame — the only kind of
reference frame `predictive_core.labeling.pool_assets` produces — this caused two simultaneous failures: 298 of 629
candidates in one test run were wrongly rejected because they landed on a nearby row occupied by a *different asset*
(BTC and ETH instances were being treated as "the same directional move" purely from row adjacency), and 21 same-asset
pairs were both counted as independent despite being as little as 12 real hours apart against a configured 24-hour
separation (genuinely overlapping windows from the same move double-counted, a direct violation of MASTER_CONTEXT
constraint 9). *Resolution:* `fit()` now stores real timestamps and asset identity alongside the existing row-position
array; `_deduplicate` was rewritten to fold a candidate into an accepted one only if both hold: same asset, and real
elapsed time below `min_temporal_separation * bar_interval`. Re-verified empirically post-fix: zero same-asset
violations, and cross-asset pairs within 1 hour of each other correctly both survive as independent.
- **`predictive_core` had zero pytest-discoverable tests.** `tests/test_predictive_core.py` only contained a `main()`
script runnable via `python -m tests.test_predictive_core`; `pytest` silently collected nothing from it, violating this
project's own Definition of Done ("passes pytest"). *Resolution:* added four real `test_*` functions covering
splitter fold isolation, the minimum-sample gate, the deduplication regression (fails on the pre-fix code, passes
after), and end-to-end walk-forward training with calibration, while keeping the original narrative script for manual
runs.
- **Layer 14 kill criteria did not actually halt anything.** `WalkForwardBacktester.run()` built the `Layer14KillCriteriaReport`
and simply returned it inside `BacktestReport` — nothing anywhere checked `any_failed` to stop or raise. Purely
informational, contrary to MASTER_CONTEXT's imperative language ("STOP, revise hypothesis" / "STOP, revise risk
engine"). *Resolution:* added a `Layer14KillCriteriaFailed` exception (carries the full report via `.report`) and a
`raise_on_kill_criteria_failure: bool = True` constructor flag on `WalkForwardBacktester` — the default is now to
halt; callers doing deliberate exploratory/notebook inspection of a failing run must opt out explicitly.
- **Slippage priced every exit-to-flat order at zero.** `TransactionCostModel.per_bar_fees_and_slippage` scaled
slippage by the raw *target position level* at each bar rather than the *size of the order actually being traded*.
Since an exit-to-flat order's post-order target is 0, the formula charged zero slippage on it regardless of how large
the position being closed was — discovered while doing the hand-verification exercise for review point 2 above, not
by reading the code. *Resolution:* rewrote the formula to scale slippage by `|Δposition|` (the traded delta, treating
the first bar's implicit prior position as flat) instead of the raw post-order target level. Rewrote the affected test
with hand-derived expected values to lock in the corrected behavior.

### Remaining Challenges & Open Items

- **Critical, unresolved performance bug in `HistoricalAnalogEngine.predict()` (specifically `_deduplicate`).**
Discovered while verifying `PredictiveCoreStrategy` end-to-end: `predict()` costs approximately 100ms per call at a
realistic reference-set size (~5,000 rows), which is why the first end-to-end verification attempt timed out after
two minutes on only 6,000 synthetic bars. Profiled with `cProfile`: `_deduplicate`'s O(k²) nested-loop independence
check (k = raw candidates within the distance threshold for a given query) accounts for essentially all of it —
293,000 Python-level `abs()` calls traced across just 20 queries in one profiling run. At real 2-year-backtest scale
(tens of thousands of reference rows, `generate_positions()` called twice per fold across a minimum of 4 folds), this
would make a real walk-forward run take on the order of tens of minutes to hours, and is a hard blocker to actually
completing Build Plan step 1D/1E on real data. **Not fixed.** The fix is straightforward (vectorize the nested Python
loop into a numpy broadcast comparison) but was intentionally left for a separate, explicitly-scoped follow-up rather
than folded into this session's work, since it touches previously-reviewed and committed code from an earlier task.
- **`PredictiveCoreStrategy` has only been mechanically verified, not behaviorally verified.** At the small synthetic
scale used for verification (1,500 hourly bars), the adapter never produced a single nonzero position. Investigated
rather than assumed benign: MASTER_CONTEXT's N≥50-independent-instances gate is close to mathematically unsatisfiable
with that little data (1,500 bars ÷ 48-bar independence separation ≈ 26 possible independent slots, below 50 no
matter what) — correct, conservative gate behavior, not a bug. But it means the full combination chain (Component A
gate → Component A/B direction agreement → confidence threshold → magnitude-based sizing) has only been verified
piece by piece, never observed firing together end-to-end on data with enough history for the gate to actually pass.
This is blocked on both more historical data and the performance fix above (larger data makes the O(k²) cost worse).
- **No real BTC/ETH historical data has been run through either the backtesting harness or the predictive core
strategy adapter.** Every verification this session used synthetic (random-walk or seeded-RNG) data. Build Plan step
1D/1E cannot actually be completed — i.e., the user cannot yet make the Go/No-Go judgment call — until real historical
OHLCV, features, and triple-barrier labels are prepared and run through the full pipeline.
- **Open interpretive question on what "clean pass" means at Phase 1E.** Two of the five Layer 14 criteria (Risk-On
regime performance vs. aggregate, ensemble weight stability) will report `not_evaluated` on every current run, since
their required inputs — the macro regime classifier (Layer 3) and the linear ensemble meta-model (Layer 7C) — are
Phase 2 work that doesn't exist yet. Whether 1E's "passes cleanly" bar should apply to only the 3 currently-computable
criteria (Sharpe, profit factor, max drawdown) or requires all five genuinely evaluated (which would make 1E
impossible to complete before Phase 2 exists) was surfaced to the user as a judgment call, not resolved.
- **`PredictiveCoreStrategy`'s position-sizing and gating logic is an explicit placeholder, not final entry logic.**
Only 1 of MASTER_CONTEXT's six trade-entry gates (constraint 8) is enforced by the adapter — the analog engine's
N≥50 sample gate, which comes for free from `AnalogForecast.is_valid`. MTF alignment, macro regime, volatility regime,
the EV lower-CI-bound threshold, and the risk score gate all require modules that don't exist yet. The position-sizing
formula (magnitude-scaled, confidence-gated) is a deliberately simple stand-in to give the Layer 14 backtest *some*
signed, bounded position series to evaluate — it is not quarter-Kelly sizing (Layer 10, not built).
- **`PredictiveCoreStrategy.label_horizon` must be manually kept in sync with whatever `max_candles` value was used
when `apply_triple_barrier` was run upstream on the input data.** The adapter has no way to verify this from inside
`fit()`; a mismatch would silently under- or over-purge at fold boundaries.

### Next Steps

1. **Fix the `HistoricalAnalogEngine._deduplicate` O(k²) performance bug** (vectorize the independence check with a
numpy broadcast comparison instead of the current nested Python loop) — this is the primary blocker to running
anything at real scale.
2. **Re-verify `PredictiveCoreStrategy` end-to-end on a larger dataset** once the performance fix lands, to confirm
the full signal-generation chain (both components' gates, agreement check, and sizing) actually fires and produces
trades, not just that each piece works in isolation.
3. **Source/prepare real BTC and ETH historical OHLCV, features, and triple-barrier labels** covering enough calendar
history for the 2-year/6-month/6-month/3-month-step/minimum-4-fold walk-forward spec (at least ~3.75 years of
continuous history required).
4. **Actually run `WalkForwardBacktester` with `PredictiveCoreStrategy` against real data** and read the resulting
five Layer 14 criteria — this is what allows the user to perform the Build Plan step 1E Go/No-Go review for real,
which could not happen this session since only synthetic data was used.
5. **Resolve the open interpretive question** about whether Phase 1E's "clean pass" bar applies to the 3
currently-computable Layer 14 criteria or requires all five (which would gate 1E behind Phase 2 module completion).
6. **If 1E passes and Phase 2 is greenlit:** begin building the Macro Regime Engine (Layer 3), MTF Alignment (Layer 4),
Liquidity Trap heuristic (Layer 5), Volatility Classifier (Layer 6), Ensemble Meta-Model (Layer 7C), Explainability
Engine (Layer 8), and Risk Engine (Layer 9), per the exact tables in `MASTER_CONTEXT.md`.

Resume this session with:
claude --resume 372037fd-6202-4770-811f-2e88ee7cc4ae