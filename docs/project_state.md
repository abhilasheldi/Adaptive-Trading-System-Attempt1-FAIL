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

## Session Update: `_deduplicate` Performance Fix (2026-08-13)

### Completed Work

- **Fixed the `HistoricalAnalogEngine._deduplicate` O(k²) performance bug** flagged as the primary blocker in the
prior session. The nested-Python-loop independence check (each candidate compared via `zip`/`abs()` against every
previously accepted candidate) was replaced with a per-asset, timestamp-sorted bucket + `bisect` approach: each
candidate now only compares against its two time-neighbors in its own asset's bucket (`bisect_left`/`insort`,
O(log m) in C) instead of scanning the full accepted set in the Python interpreter (O(m) per candidate, all in
Python). `fit()` now precomputes `_ref_timestamps_ns` (int64 nanoseconds) once, since `bisect` needs plain-int
comparisons rather than per-element `np.datetime64`/`Timedelta` objects.
- **Verified equivalence, not just speed.** The old O(k²) implementation was kept (only inside the test file, not
production code) as `_naive_deduplicate_reference` and cross-checked against the new implementation across 40+
randomized trials (varying separation thresholds, random acceptance orders, with and without an asset column) —
zero mismatches. This was a pure performance change: identical accepted-index output for identical input.
- **Benchmarked:** ~0.63ms/call on a 2,283-candidate pool (vs. the ~100ms/call documented in the prior session at
comparable scale) — roughly 150x faster.
- **Added `tests/test_predictive_core.py::test_deduplicate_matches_naive_reference_and_is_fast`**, a permanent
regression test asserting both equivalence (vs. the naive reference, across randomized trials) and a runtime bound
(200 calls over a 500+-candidate pool in under 5s) so an accidental reintroduction of O(k²) behavior fails CI
instead of only surfacing at real-data scale.
- Full suite: 44/44 tests pass (43 pre-existing + 1 new).

### Remaining Challenges & Open Items (carried forward, now unblocked by the fix above)

- **`PredictiveCoreStrategy` re-verification at scale** — still needs to be run end-to-end on a larger dataset now
that the performance blocker is removed, to confirm the full signal-generation chain actually fires and produces
trades (not just that each piece works in isolation on tiny synthetic data).
- **No real BTC/ETH historical data has been run through either the backtesting harness or the predictive core
strategy adapter** — still entirely synthetic-data-verified. This, not performance, is now the primary blocker to
completing Build Plan step 1D/1E.
- **Open interpretive question on Phase 1E "clean pass"** — still unresolved (3 currently-computable Layer 14
criteria vs. all 5, which would gate 1E behind Phase 2 module completion).
- **`PredictiveCoreStrategy`'s position-sizing/gating logic remains an explicit placeholder** — only the N≥50
sample gate is wired in; MTF alignment, macro regime, volatility regime, EV lower-CI threshold, and risk score gate
all require Phase 2 modules that don't exist yet.
- **`PredictiveCoreStrategy.label_horizon` still requires manual sync with upstream `apply_triple_barrier`'s
`max_candles`** — no internal verification exists for this.

### Next Steps

1. **Re-verify `PredictiveCoreStrategy` end-to-end on a larger synthetic dataset** to confirm the full gate →
agreement-check → sizing chain fires and produces trades now that the performance fix removes the timeout blocker.
2. **Source/prepare real BTC and ETH historical OHLCV, features, and triple-barrier labels** (~3.75+ years of
continuous history for the 2yr/6mo/6mo/3mo-step/min-4-fold walk-forward spec).
3. **Run `WalkForwardBacktester` with `PredictiveCoreStrategy` against real data** and read the resulting five
Layer 14 criteria — this is what allows the actual Build Plan step 1E Go/No-Go review.
4. **Resolve the open interpretive question** about Phase 1E's "clean pass" bar.
5. **If 1E passes and Phase 2 is greenlit:** begin building the Macro Regime Engine (Layer 3), MTF Alignment
(Layer 4), Liquidity Trap heuristic (Layer 5), Volatility Classifier (Layer 6), Ensemble Meta-Model (Layer 7C),
Explainability Engine (Layer 8), and Risk Engine (Layer 9).

## Session Update: Real-Data Validation of the Data Engine, and 4 Bugs Found/Fixed (2026-08-13)

### Completed Work

- **Verified `data_engine/crypto_fetcher.py` against live Binance data** (not synthetic): fetched ~9 years of real
BTC/USDT 1h OHLCV (78,664 candles, 2017-08-17 to 2026-08-13) and real ETH/USDT 1h and 4h data. Pipeline (fetch →
`remove_duplicates_and_skew` → `classify_gaps` → `flag_outliers`) ran cleanly: 99.96% of candles classified
"normal", 0.09% flagged as outliers (consistent with real flash-crash wicks, not a bug). Both existing live smoke
tests (`tests/test_binance_real_data.py`) pass.
- **Built `src/predictive_core/dataset.py`** (`build_asset_modeling_frame`, `feature_columns`) — the missing bridge
between `features.feature_calculator` and `predictive_core.labeling`/the backtesting harness. Root problem it
solves: `calculate_features()` deliberately emits ONLY percentile-ranked feature columns (no raw OHLCV, no raw ATR
— by design, per that module's "no raw values leave this module" rule from an earlier session). But
`apply_triple_barrier()` needs raw-scale `close`/`high`/`low` and a raw-scale ATR to set real price-level barriers,
and `PredictiveCoreStrategy`/`WalkForwardBacktester` both require raw `close`. Piping `calculate_features()`'s
output directly into either would KeyError (no `close` at all) or silently corrupt every barrier level (percentile
ATR in [0,1] used as a dollar offset). This had never been exercised because every existing test builds its own
raw-scale `vol_atr_14` directly in synthetic fixtures, bypassing `calculate_features()` entirely. Verified against
real BTC and ETH data: output has 39 feature columns (all in [0,1]), raw OHLCV intact (BTC close $18.7k–$126k over
the period), `magnitude` in real USD terms ($28–$12,266), `duration` correctly bounded by `max_candles`, zero NaNs.
Added `tests/test_predictive_core_dataset.py` (synthetic-data, no network) locking in the raw-vs-percentile
distinction so this can't silently regress.
- **Ran the full chain end-to-end on real data for the first time**: `HistoricalAnalogEngine.fit()`/`predict()` on
20,000 real BTC rows (0.008s fit, ~11ms/predict, 99/100 real queries passed the N≥50 gate — a meaningfully
non-trivial pass rate, neither impossibly strict nor trivially always-true) and a full `PredictiveCoreStrategy.fit()`
+ `generate_positions()` run on real 2022 BTC data (4000 train / 800 val / 400 test rows), which produced 358/400
nonzero positions. This directly resolves the prior session's open item ("never observed producing an actual trade
end-to-end") — the earlier synthetic-data run failed to produce any nonzero position only because 1,500 bars was
mathematically too little data for the N≥50 gate, not a strategy bug.

### Bugs Found and Fixed

1. **`config.yaml`'s macro ticker key `btc_dom` didn't match what `macro_fetcher.ingest_macro` actually looks
for (`"btc_dominance"`)**, so `btc_dom_requested` was always `False`. Effect, confirmed live: `"BTC.D"` (the
config's old value, not a real yfinance symbol) was sent to yfinance and returned six entirely-`NaN` columns
(`btc_dom_Close`, `btc_dom_Open`, etc.), while real BTC dominance was never fetched from CoinGecko at all — a
direct hit to Layer 3's "BTC dominance expanding" Risk-On signal. Fixed by renaming the config key to
`btc_dominance`. Verified live post-fix: `btc_dominance_pct` correctly populated (56.25%) via CoinGecko, no more
garbage columns. Added `tests/test_macro_fetcher.py` (mocked, no network) covering both the routing-present and
routing-absent cases.
2. **`macro_fetcher._SCALE_CORRECTIONS` multiplied `^TNX`/`^TYX`/`^FVX` (Treasury yield tickers) by 0.1**, on the
belief that Yahoo reports "yield × 10". Checked against live data and found false: `^TNX` closed at 0.917 on
2021-01-04 (matches the real, well-documented ~0.92% 10Y yield that day) and 4.670 on 2026-08-06 (matches the real
~4.67% yield then) — both already correctly-scaled percentages, no ×10 involved anywhere. The 0.1 factor was
silently dividing real yield data by 10 on every fetch. Removed the correction entirely; values now pass through
unmodified. Added a mocked regression test.
3. **`HistoricalAnalogEngine.predict(..., query_timestamp=...)` crashed on tz-aware timestamps** —
`TypeError: can't compare offset-naive and offset-aware datetimes`. Root cause: `self._ref_df[ts_col].to_numpy()`
on a tz-aware column returns an object array of tz-aware `Timestamp`s, which can't be compared against
`np.datetime64(query_timestamp)` (numpy has no timezone representation). Real crypto data from `crypto_fetcher` is
always UTC tz-aware (`pd.to_datetime(..., utc=True)`); every existing test used tz-naive synthetic timestamps, so
this was never caught — until real data was run through `PredictiveCoreStrategy.generate_positions()`, which calls
`predict(..., query_timestamp=ts)` on every row, meaning the full strategy crashed on 100% of real-data calls. This
was the most severe of the four bugs found this session. Fixed by comparing through the already-tz-normalized
`_ref_timestamps_ns` (added a `_to_ns` helper that normalizes both naive and tz-aware inputs to the same UTC-based
int64-ns scale `fit()` already computes). Also cleaned up a related `UserWarning` ("no explicit representation of
timezones available for np.datetime64") at `fit()` time by converting tz-aware timestamps explicitly
(`tz_convert("UTC").tz_localize(None)`) instead of relying on numpy's implicit (and numerically-correct-but-noisy)
coercion. Added `test_predict_accepts_tz_aware_query_timestamp` (regression test, both tz-aware and tz-naive query
paths).
4. Confirmed **no bug** in a suspected fourth spot: pandas `resample("1h")`/`resample("4h")` (used by
`remove_duplicates_and_skew`) both work correctly as pandas offset aliases — checked directly since ccxt-style
lowercase timeframe strings are a plausible pandas-version-compatibility trap, but this one turned out fine.

### Remaining Challenges & Open Items

- **`is_outlier`/`gap_class` rows are flagged by `crypto_fetcher` but nothing downstream filters or otherwise acts
on them** — `calculate_features()` and `build_asset_modeling_frame()` process them like any other candle as long as
OHLC integrity holds. This is a data-quality judgment call (drop outlier candles? down-weight? leave as-is and let
percentile-ranking absorb them?) that hasn't been made, not a code defect — flagging it as an open decision rather
than resolving it unilaterally.
- **`config.yaml` still has no `normalized_data_dir` entry** — MASTER_CONTEXT constraint 3 requires raw and
normalized data to be stored separately, but only `data_dir` (raw) exists in config; `build_asset_modeling_frame`'s
output isn't wired to any save path yet. Next real-data run should decide and add a `data/normalized/` convention.
- **The observed real-BTC direction-label imbalance (from `apply_triple_barrier` with MASTER_CONTEXT's
`tp_mult=2.0`/`sl_mult=1.0`)** is heavily skewed toward "down" (in one 2023–2026 BTC sample: 20,423 down / 10,894 up
/ 68 neutral) — a mechanical consequence of the stop-loss barrier (1×ATR) being structurally closer than the
take-profit barrier (2×ATR), so short-term noise hits it more often regardless of trend. This is spec-driven
(MASTER_CONTEXT's own multipliers), not a bug, but worth the user's awareness before interpreting Component B's
direction-classifier calibration numbers on real data.
- **`_deduplicate`'s performance fix and this session's bug fixes are still only validated on ~2–4 year real-data
slices**, not the full ~9-year history or the full 2yr/6mo/6mo/3mo-step walk-forward spec (which real BTC data now
easily covers in calendar span). The next step from the prior session — actually running `WalkForwardBacktester`
with `PredictiveCoreStrategy` against full real data and reading the five Layer 14 criteria — is now realistically
unblocked (data works, performance works, the tz crash is fixed) but still not done.

### Next Steps

1. **Run the full `WalkForwardBacktester` + `PredictiveCoreStrategy` against real BTC and ETH data** at the actual
MASTER_CONTEXT window sizes (2yr/6mo/6mo/3mo-step, min 4 folds) — real data now covers this easily (~9 years for
BTC, ~6+ years for ETH on Binance). This is what allows the actual Build Plan step 1E Go/No-Go review.
2. **Decide and implement a `data/normalized/` storage convention** for `build_asset_modeling_frame` output,
consistent with MASTER_CONTEXT constraint 3.
3. **Decide how (or whether) to act on `is_outlier`/`gap_class` flags** upstream of feature engineering.
4. **Resolve the open interpretive question** about Phase 1E's "clean pass" bar (carried over, unchanged).
5. **If 1E passes and Phase 2 is greenlit:** begin building the Macro Regime Engine (Layer 3), MTF Alignment
(Layer 4), Liquidity Trap heuristic (Layer 5), Volatility Classifier (Layer 6), Ensemble Meta-Model (Layer 7C),
Explainability Engine (Layer 8), and Risk Engine (Layer 9).

## Session Update: Real-Data Walk-Forward Investigation — Root Cause Found (2026-08-13)

**This is the most important entry in this file to date.** It documents a multi-stage investigation that started
as "run the real backtest" and ended with empirical proof of *why* the predictive core fails on real data — not a
code bug, but a mismatch between what MASTER_CONTEXT's hypothesis actually claims and what the current pipeline
tests. Read this in full before making the Phase 1E Go/No-Go call or writing any more predictive-core code.

### What was built this session (context for the investigation below)

Before the investigation itself, four real code changes were made and verified, in this order:

1. **Data engine bug fixes**, verified live against Binance/Yahoo/CoinGecko: a macro-ticker config key mismatch
that silently broke BTC dominance ingestion; a wrong ×0.1 "yield correction" that was corrupting real Treasury
yield data by 10x (disproven against live data spanning 2021 and 2026); a timezone-handling crash in
`HistoricalAnalogEngine.predict()` that made the real (tz-aware) data path crash on 100% of calls.
2. **`src/predictive_core/dataset.py`** (`build_asset_modeling_frame`, `feature_columns`) — the missing bridge
between `features.feature_calculator` (which deliberately strips raw OHLCV and emits only percentile-ranked
features) and `predictive_core.labeling`/the backtesting harness (which need raw `close`/`high`/`low` and a
real-price-scale ATR for triple-barrier levels). Without this, real data could not flow through the system at all.
3. **Two new diagnostic layers** for Component B (the direction classifier), added specifically because
calibration checks (ECE/Brier) cannot detect a classifier that has simply learned to always predict the majority
class — a constant predictor outputting the true base rate is *calibrated by definition*: (a)
`skill_diagnostics.py` — per-class precision/recall/F1, and a logloss "skill score" against a train-base-rate
baseline; (b) plumbing (`PredictiveCoreStrategy.diagnostics()`, `SignalStrategy.diagnostics()` protocol,
`FoldBacktestResult.diagnostics`) so these numbers actually reach the backtest report instead of being computed
and immediately discarded, which is what the code did before this session.
4. **Per-sample class weighting** added to `train_direction_classifier` (`weight_i = n_samples / (n_classes *
class_count[y_i])`, computed from `y_train` only, per fold — the standard "balanced" formula), plus an interim
safety cap (`PredictiveCoreStrategy.max_position` default lowered from 1.0 to 0.02, explicitly NOT real Layer 10
sizing, just preventing a Phase-1 research run from misrepresenting classifier skill as catastrophic risk-taking),
plus a `label_horizon`/`apply_triple_barrier` cross-check (`df.attrs['label_horizon']` set at labeling time,
checked in `PredictiveCoreStrategy.fit()`, raises loudly on mismatch, skips gracefully if the metadata didn't
survive some upstream operation).

All of the above was necessary, all of it works, and none of it was the actual problem. The investigation below is
what actually explains the results.

### The investigation, in the order it happened

**Stage 1 — first real-data walk-forward run (2 Optuna trials, deliberately cheap sanity check).** Real BTC/USDT
1h data, 2022-08 to 2026-08 (~4 years, 5 walk-forward folds at the spec 2yr/6mo/6mo/3mo-step sizing). Result:
catastrophic. Aggregate OOS Sharpe **-20.28**, profit factor **0.21**. The new diagnostics immediately explained
why: every fold showed "down" recall = 1.00 (the classifier calls "down" on essentially every row) and "up" recall
≈ 0.00–0.02 (it essentially never calls "up"), with a logloss skill score of 0.004–0.010 — i.e., **no better than
a model that just always guesses the training set's base rates.** Textbook majority-class-parrot behavior.

**Stage 2 — hypothesis: "it just needs a real tuning budget."** Re-ran identically with the full spec Optuna
budget (15 trials, 50 early-stopping rounds vs. the crippled run's 2/10) — 47.7 minutes instead of 37 minutes for
the same 5 folds. Result: **statistically indistinguishable** from Stage 1. Sharpe -19.81, profit factor 0.211,
skill score 0.008–0.015, "down" recall still 1.00 every fold, "up" recall still ≈0. **This ruled out "insufficient
tuning" as the explanation** — 7.5x more search budget produced no meaningful change.

**Stage 3 — hypothesis: "it's a class-imbalance problem in the loss function."** The real-data label split is
mechanically skewed (~65% down / 34% up / <1% neutral on BTC 2023-2026) — a geometric artifact of MASTER_CONTEXT's
own `tp_mult=2.0`/`sl_mult=1.0` (the stop-loss barrier sits half the distance from entry as the take-profit
barrier, so ordinary noise hits it far more often, regardless of true market direction). `train_direction_classifier`
had zero class rebalancing, so "mostly predict the majority class" is a very reachable local optimum for a raw
multiclass-logloss minimizer under this much imbalance. Added the balanced class-weighting formula (see above) and
re-ran identically (again 15 trials, 50 early-stopping rounds, same 5 folds). Result: **partially different, but
still fundamentally the same failure.** Sharpe improved to -12.90, profit factor to 0.231, max drawdown dropped
from 12.3% to 5.6%, and OOS trade count fell substantially (fold 0: 2,004 → 1,302; fold 3: 2,275 → 211) — the
weighted model became more conservative (cleared its own confidence gate less often), which mechanically reduces
realized losses. But the actual discrimination diagnostics **did not move**: skill score 0.005–0.016 (same range
as before), "down" recall still 1.00 every fold, "up" recall still ≈0.00–0.01. **This ruled out "class imbalance in
the loss function" as the (full) explanation too** — rebalancing changed how often the model fires, not what it
actually knows.

**Stage 4 — the feature-signal check that found the real answer.** Two clean diagnostics, run on the full ~9-year
real BTC/USDT dataset (78,378 modeling rows after warm-up, 78,226 after excluding the negligible neutral class),
completely independent of Optuna/weighting/calibration/agreement-gates/any of the system's other machinery:

- **Univariate**: for each of the 39 `feature_calculator` features, treat its raw percentile-ranked value as a
  score and compute its ROC-AUC against the binary up/down label directly (`sklearn.roc_auc_score`). AUC=0.5 means
  a feature carries zero rank-discriminative information on its own. Result, across ALL 39 features: **the single
  best feature (`mom_di_spread`, the ADX directional-index spread) scored AUC=0.5172** — 1.7 percentage points
  above chance. The median feature scored |AUC-0.5|=0.0024 — indistinguishable from noise. (The full per-feature
  ranked CSV this produced was a workspace-cleanup casualty on 2026-08-13 — every number that mattered is preserved
  above; re-running the check, described in full in this entry, regenerates it in under a minute.)
- **Multivariate ceiling**: a single, generously-capacitated LightGBM binary classifier (63 leaves, unbounded
  depth, 2000 max boosting rounds with real early stopping, no Optuna limits, no class weighting, no calibration,
  none of the system's other constraints — deliberately the *most* favorable conditions the exact same 39 features
  could be given) on a simple chronological 70/15/15 split of the full dataset. Result: **test-set AUC = 0.5289**,
  test-set logloss = 0.6401 vs. a constant-base-rate baseline's logloss of 0.6409 — a **0.12% improvement over
  guessing the base rate for every single row, using full nonlinear feature interactions and unlimited model
  capacity.** Feature importances were spread fairly evenly across generic volatility/momentum/structural features
  (`vol_std_20`, `vol_atr_14`, `vol_bb_width`, `struct_dist_to_resist`, `mom_adx_14`, ...) — no small subset of
  features dominates, which is itself evidence of "the model is fitting noise/marginal structure," not "there's a
  real signal buried under a few dominant features the tuned pipeline hasn't found yet."

**Conclusion of the investigation**: the near-zero skill is not a training bug, not a tuning-budget problem, and
not (primarily) a class-imbalance problem. It is the correct, honest output of the classifier given what it was
actually asked to do — because **the current 39-feature percentile-ranked technical feature set contains
essentially no separable signal for predicting the current triple-barrier label, even under maximally favorable
modeling conditions.** Nothing found in Stages 1-3 could have fixed this, because there was never enough signal in
the inputs for any amount of tuning or rebalancing to find.

### Why: the actual structural problem, not just the symptom

MASTER_CONTEXT's hypothesis (see its own opening section) is narrow and **conditional**: *"post-compression
liquidity-sweep reversals in BTC and ETH on the 1H-4H timeframe, during risk-on macro regimes, with multi-timeframe
structural alignment."* Four distinct conditions have to hold simultaneously for this hypothesis to even apply to a
given candle: (1) a compression state, (2) a liquidity sweep having just occurred, (3) a risk-on macro regime, (4)
multi-timeframe structural alignment.

What has actually been tested this session — necessarily, because it's all that's been built — is a completely
**unconditional** question: *given literally any candle in ~9 years of BTC history, does price move 2×ATR up or
1×ATR down first within 48 hours?* Every one of the four conditions above requires a module that is explicitly
Phase 2, not-yet-built work:

- Condition 1 (compression) is *partially* available — `vol_state_compressed` is one of the 39 features (Layer 6,
  already built) — but it is only ever used as one undifferentiated input feature among 39, never as a *filter*
  that restricts which candles the classifier is even asked to predict.
- Condition 2 (liquidity sweep) requires Layer 5 (`macro_mtf_liquidity_vol`'s Liquidity Trap heuristic) — this
  module is currently empty (`__init__.py` only). It does not exist.
- Condition 3 (risk-on regime) requires Layer 3 (Macro Regime Engine) — also does not exist yet.
- Condition 4 (MTF alignment) requires Layer 4 (MTF Alignment scorer) — also does not exist yet.

With none of these four filters in place, the predictive core was trained on the *entire* unconditioned dataset —
tens of thousands of ordinary candles that have nothing to do with the hypothesized setup, alongside whatever small
number of genuine liquidity-sweep-reversal instances may exist within that history. If real edge exists
specifically within that narrow subset, training (and evaluating AUC) across the *entire* dataset would dilute it
to statistical invisibility — which is exactly the AUC≈0.53 ceiling actually observed. A 0.53 AUC ceiling averaged
across ~78,000 unconditioned candles is fully consistent with a much stronger, real, but rare and currently-
unidentified edge existing within a small conditioned subset of those candles; the univariate/multivariate checks
run this session cannot distinguish between "no edge exists anywhere" and "edge exists but is diluted below
detectability by testing unconditionally" — they only rule out "the modeling technique or tuning is the problem."

**Component A (the k-NN analog engine) is subject to the same structural gap.** Its neighbors are drawn from the
same globally-imbalanced, unconditioned reference set, with no setup-specific conditioning either. For a typical
query, its empirical neighbor vote share has no particular reason to differ from the global ~65/34 base rate,
because nothing currently tells it "only compare this query against other post-compression-sweep instances."

### Is this "the hypothesis failed"? — an honest, precise answer

Not exactly, and the distinction matters for the Phase 1E judgment call. MASTER_CONTEXT's Layer 14 kill-criteria
language ("STOP, revise hypothesis") is written for the situation where the hypothesis, tested as specified, comes
up empty. What actually happened here is closer to: **the hypothesis has not yet been tested as specified**,
because three of its four defining conditions have no implementation to test them with. The unconditional test
that *was* run is a reasonable, necessary Phase 1 step (proving the data/feature/predictive-core/backtesting
pipeline works end-to-end on real data, which it now does — see prior session entry), but it is a weaker,
different claim than "the liquidity-sweep-reversal hypothesis has no edge."

**Stage 5 — the cheap conditioned-subset experiment (executed same session, immediately after Stage 4).** Built
`scripts/compression_sweep_filter.py`: a minimal, explicitly-labeled **diagnostic-only proxy** for Layer 5 (NOT a
real implementation of it — `macro_mtf_liquidity_vol/` is still empty), constructed entirely from features
`feature_calculator.py` already computes, requiring no new indicator logic. **Removed during a 2026-08-13
workspace cleanup** (self-labeled as diagnostic-only, not production code, and the investigation it served is
paused — see the end of this session's entries) — logic reproduced exactly below so it can be rebuilt in under five
minutes if this line of investigation resumes:

- "post-compression" = `vol_state_compressed` was true at any point in the trailing 5 candles (inclusive of the
  current row) — a blunt proxy; a real Layer 5 would define this far more rigorously (contraction slope, duration,
  etc.). The lookback (5) was chosen empirically: checked 3/5/10/20-candle lookbacks first and picked the smallest
  that stayed meaningfully selective rather than capturing most of the dataset (compression alone is true 35.5% of
  all candles and is autocorrelated/persists in runs, so a naive longer lookback like 20 candles would have made
  "recent compression" true for 56% of ALL candles — nowhere near selective enough to call a "setup").
- "liquidity sweep" = `struct_sfp_bullish` or `struct_sfp_bearish` fires on the current candle — already-computed
  Swing Failure Pattern features (a wick beyond a recent swing high/low, immediately rejected back inside range),
  which is structurally exactly a stop-hunt/liquidity-sweep pattern, requiring no new logic to reuse.
- Implementation was exactly: `mask = (df['vol_state_compressed'].rolling(5, min_periods=1).max() > 0) &
  ((df['struct_sfp_bullish'] > 0) | (df['struct_sfp_bearish'] > 0))` — a single boolean Series aligned to the
  modeling frame's index, both inputs already point-in-time-safe, trailing-only rolling window (no lookahead).

Applied to the same full ~9-year real BTC dataset used in Stage 4: **1,600 of 78,378 rows (2.04%) pass the
filter** — a genuinely selective subset, not "almost everything," which was the real risk with this kind of blunt
proxy filter. Re-ran both Stage 4 diagnostics on this filtered subset:

- **Class balance within the filtered subset**: 1,047 down / 553 up / 0 neutral (65.4% / 34.6% / 0%) — statistically
  indistinguishable from the unconditioned dataset's 65.8% / 33.6% / 0.2% split. This confirms directly that the
  mechanical label-imbalance driver (the `tp_mult`/`sl_mult` barrier-geometry asymmetry) applies uniformly
  regardless of setup type, exactly as expected — it is a property of the labeling rule, not of "normal" vs.
  "post-compression-sweep" candles specifically. (The complete absence of any neutral-labeled row in this subset is
  itself intuitive: sweep candles are by definition decisive/volatile moves, not the small, indecisive moves that
  earn a neutral label.)
- **Univariate AUC** (up vs. down, 1,600 rows): max |AUC-0.5| = 0.0206 (`struct_dist_to_resist`), median = 0.0068 —
  the same order of magnitude as the unconditioned check (max 0.0172, median 0.0024), if anything nominally higher
  on both ends, but on a much smaller sample where AUC estimates are inherently noisier. No feature shows anything
  resembling real univariate separation. (Ranked CSV cleaned up 2026-08-13 along with the other diagnostic
  artifacts; every number that mattered is preserved here.)
- **Multivariate ceiling** (chronological 70/15/15 split → train=1,120 / val=240 / test=240): the model's own early
  stopping halted after **1 boosting round** — i.e., it found nothing worth building a second tree on. **Test AUC =
  0.4361** (below 0.5 — worse than a coin flip on this sample), test logloss (0.6494) very slightly *worse* than
  the constant-base-rate baseline (0.6477). A 2,000-resample bootstrap on the 240-row test set gives a **95% CI of
  [0.36, 0.51]** for test AUC — the entire interval sits at or below chance, and its width (0.15) reflects how much
  genuine statistical uncertainty a 240-row test set carries; this is a small, underpowered check, not a precise
  measurement.

**Honest interpretation — this is a genuinely negative-leaning result, but a limited one, not a conclusive one.**
Filtering to this specific cheap proxy for "post-compression liquidity sweep" did *not* reveal hidden signal the
unconditioned check was diluting — if anything, the point estimates on this subset were worse than unconditioned,
though within a confidence interval wide enough that "no different from the unconditioned 0.53 ceiling" cannot be
ruled out either. Two real, honest caveats limit how much weight this result can bear: (1) the filter itself is a
crude proxy, not a rigorous Layer 5 implementation — a real liquidity-trap heuristic (order-flow-aware, multi-
timeframe, threshold-tuned) might select a meaningfully different and more genuinely edge-concentrated subset than
"any SFP within 5 candles of any compression reading"; (2) 1,600 rows (240 in the held-out test fold) is a small
sample for detecting a real-but-modest edge with any statistical confidence — the wide bootstrap CI is the honest
signature of that, not evidence of "no edge," just evidence of "not enough data here to tell." This result should
be read as "the cheapest possible version of the conditioning experiment did not find evidence of edge, and does
not by itself justify the full Phase 2 build-out on its own" — not as "the liquidity-sweep-reversal hypothesis is
disproven," which is a claim this crude, small-sample check cannot support either way.

**Stage 6 — pooling BTC+ETH under a bounded (non-lifetime) window, executed same session immediately after
Stage 5.** Explicit user instruction: pool both assets, but do NOT use each asset's full lifetime history (BTC's
full history runs back to 2017-08-17; using it unbounded while ETH only had a shorter window fetched would also
make the pool lopsided). Used the same bounded window as Stages 1-3's walk-forward checks (2022-08-01 onward,
~4 years) for both assets, for direct comparability. Mechanics, to avoid any cross-asset leakage: raw OHLCV
fetched/windowed per asset first, `build_asset_modeling_frame` (which internally calls `apply_triple_barrier`) run
per asset separately so no asset's forward-looking label window can ever reference the other asset's candles, the
Stage 5 `compression_sweep_mask` filter applied to each asset's modeling frame separately, and only THEN
concatenated and stable-sorted by timestamp (`kind="mergesort"`, the same convention `predictive_core.labeling.
pool_assets` already uses) — i.e., pooling happens only after every per-asset, leakage-sensitive step is complete.

**Important caveat surfaced immediately by this run, before even looking at the signal results**: pooling under
the bounded window did NOT net a bigger sample than Stage 5's BTC-lifetime-only check. BTC's bounded-window filtered
subset is only 663 rows (vs. 1,600 using BTC's full 9-year lifetime in Stage 5) — the ~4-year bounded window is
simply much shorter than BTC's full history, so restricting to it costs more sample than pooling in ETH gains back.
Pooled BTC (663) + ETH (560) = **1,223 rows total**, smaller than Stage 5's single-asset 1,600. This is a direct,
mechanical consequence of the explicit "no lifetime data" instruction, reported transparently rather than
glossed over — it means this experiment answers "does pooling assets under a fair, comparable window help,"
not "does pooling assets with maximum available history help" (a different, not-yet-run experiment).

Results on the pooled, bounded-window, filtered subset (1,223 rows; up-vs-down subset after excluding neutral:
1,223 rows, 0 neutral again, same pattern as Stage 5):

- **Univariate AUC**: max |AUC-0.5| = 0.0274 (`mom_adx_14`), median = 0.0037 — same order of magnitude as both
  Stage 4 (unconditioned) and Stage 5 (single-asset filtered) checks. No feature shows real separation. (Ranked CSV
  cleaned up 2026-08-13 along with the other diagnostic artifacts; every number that mattered is preserved here.)
- **Multivariate ceiling** (chronological 70/15/15 → train=856 / val=183 / test=184, test fold composition 106
  BTC / 78 ETH, confirming the pool is genuinely interleaved, not asset-siloed by the split): early stopping ran
  16 rounds this time (vs. Stage 5's single round) before halting, but the result is, if anything, **slightly worse
  than Stage 5**: VAL AUC = 0.4784, **TEST AUC = 0.4746** (both below chance), TEST logloss (0.6681) worse than the
  constant-base-rate baseline (0.6595). Bootstrap 95% CI on test AUC: **[0.3854, 0.5627]** (n_test=184) — wider than
  Stage 5's CI (0.177 vs. 0.147), consistent with the smaller test-fold sample, and still centered at/below chance.

**Honest interpretation, unchanged in substance from Stage 5**: pooling BTC+ETH under a fair bounded window did not
reveal hidden signal, and — because the "no lifetime data" constraint meant this pooling didn't actually increase
sample size relative to Stage 5 — this result carries essentially the same statistical weight as Stage 5, not more.
The confidence interval, if anything, is slightly wider (less certain) than Stage 5's. **Three consecutive
independent checks now (Stage 4 unconditioned, Stage 5 BTC-filtered, Stage 6 BTC+ETH-filtered-bounded) have found
no evidence of directional edge in this feature set for this labeling target**, all while remaining honestly
underpowered on the conditioned checks specifically. Path (a) from the previous Next-Steps entry ("pool BTC+ETH to
roughly double the sample") has now been tried under the "no lifetime data" constraint and did NOT achieve a
meaningful sample-size increase — a genuinely larger, well-powered conditioned sample would require either using
more historical data (which was explicitly excluded here), waiting for more real-time data to accumulate, or
designing a deliberately less-selective (higher base-rate, more inclusive) filter, trading specificity for sample
size.

### Next Steps (supersedes item 1 in the previous entry, which is now done; items 2 and 3(a)'s pooling half are
also now done — see Stages 5 and 6 above)

1. **Do not rebuild the predictive core's modeling technique.** k-NN analog matching + calibrated gradient
boosting + isotonic calibration are reasonable, standard, correctly-implemented choices for this kind of problem
— Stages 1-3 above prove the *architecture* was never the bottleneck. Rebuilding it without addressing the scope
problem in Stages 4-6 would very likely reproduce the same near-zero-signal result at higher engineering cost.
2. ~~Build a minimal compression/sweep filter and re-test on the conditioned subset~~ — **done, see Stage 5.**
~~Pool BTC+ETH to increase the filtered sample size~~ — **done, see Stage 6, but did NOT increase sample size**
under the explicit "no lifetime data" constraint (1,223 pooled rows vs. Stage 5's 1,600 BTC-only-lifetime rows).
3. **Decide between two paths, now with three consistent negative-leaning-but-underpowered data points instead
of one:**
   (a) **Get a genuinely larger conditioned sample** — the only way to actually narrow these ~0.15-0.18-wide
   confidence intervals — which now explicitly requires either revisiting the "no lifetime data" constraint (e.g.,
   using BTC's full history after all, accepting the asymmetry, or applying the same lifetime treatment to both
   assets), waiting for more real-time data to accumulate, or designing a deliberately less-selective filter
   (larger compression lookback, single-condition instead of both, etc.) that trades setup-specificity for volume;
   or
   (b) **Commit to the full Phase 2 build-out** (Layer 3 Macro Regime, Layer 4 MTF Alignment, Layer 5 Liquidity
   Trap) directly, treating Stages 5-6 as consistent-but-still-inconclusive preliminary evidence, and let the real
   walk-forward backtest with all four hypothesis conditions actually enforced be the real test.
   This is the user's call — three consecutive checks landing in the same "no signal found, but underpowered"
   place is more informative than one, but still isn't a clean answer either way.
4. **If Phase 2's conditioning layers do get built, re-run the full walk-forward backtest with all four conditions
actually enforced** — this remains the first point at which MASTER_CONTEXT's hypothesis, as actually specified,
will have been tested at all.
5. **Consider re-running the Stage 4/5/6 signal checks with a symmetric triple-barrier (tp_mult=sl_mult=1.0)**
purely as a diagnostic, separate from the real 2:1 asymmetric barrier used for the trading system. This would
remove the mechanical 65/34 label skew and give a cleaner read on whether ANY directional signal exists,
independent of the barrier-geometry confound — informational only, not a proposal to change the real system's
barrier configuration.
6. **The Build Plan step 1E Go/No-Go call is still the user's to make**, now with three independent, consistent
data points instead of one: this is not "the system is broken" (every component was independently verified against
real data this session) and not conclusively "the hypothesis has no edge" (every attempt to test the real,
conditioned hypothesis so far has been underpowered) — it is "three checks in a row found nothing, and the next
honest step (3a or 3b above) requires a real resourcing decision, not more free diagnostics under the current
data constraints."
7. **Resolve the open interpretive question** about Phase 1E's "clean pass" bar (carried over from prior entries,
still unresolved).
8. **Decide and implement a `data/normalized/` storage convention** and **decide how to act on `is_outlier`/
`gap_class` flags** (both carried over from the prior entry, unchanged, lower priority than the above).

### Workspace cleanup (2026-08-13, end of session)

Per explicit request, paused this investigation (steps 3a/3b above are unresolved and require a real decision, not
more free diagnostics) and cleaned up everything that was specific to it, keeping everything that is genuinely
reusable production work per the Prompt Playbook / Build Plan:

- **Removed**: `scripts/compression_sweep_filter.py` (self-labeled diagnostic-only proxy, not production code —
  exact logic preserved in the Stage 5 entry above so it can be rebuilt in minutes if this resumes), the three
  investigation CSVs in `reports/` (`feature_signal_univariate_auc*.csv` — every number they contained is already
  in the prose above, nothing was lost), and stale build caches (`__pycache__`, `.pytest_cache`, `.ruff_cache` —
  auto-regenerate, zero value to keep).
- **Kept** (all genuinely useful production work, not investigation-specific): `scripts/run_walkforward_backtest.py`
  (the actual Build Plan step 1D tool), `src/predictive_core/dataset.py`, `src/predictive_core/skill_diagnostics.py`,
  the `diagnostics()` plumbing in `harness.py`/`predictive_core_strategy.py`, the class-weighting and tz/dedup-
  performance fixes in `boosted_core.py`/`analog_engine.py`, the macro-fetcher bug fixes, the `label_horizon`
  cross-check and provenance docstring in `labeling.py`, and every test file added this session. All 60 tests still
  pass after cleanup. `data/raw/*.parquet` (cached real BTC/ETH OHLCV) also kept — legitimate, gitignored, exactly
  what the data engine is supposed to produce, not investigation clutter.

## Session Update: Macro Regime Engine Built (Layer 3) (2026-08-13)

Built `src/macro_mtf_liquidity_vol/macro_regime.py` — the first Phase 2 module (previously `macro_mtf_liquidity_vol/`
was empty). Implements MASTER_CONTEXT's exact four-state table (`classify_macro_regime`,
`MacroSignalSnapshot`/`MacroRegimeResult`). Key decisions, checked directly against source docs rather than
assumed:

- **US10Y is accepted as an input** (the Prompt Playbook's Layer 3 instruction explicitly lists it) **but used in
  neither classification nor confidence** — verified by re-checking MASTER_CONTEXT's own regime table (both the
  `.md` and the source `.docx`), which never uses US10Y in any of the four conditions. Echoed on the result for
  downstream visibility only, with this gap documented in the module docstring rather than silently inventing an
  unspecified rule for it.
- **Exact spec numbers hardcoded, not configurable**: `VIX_RISK_ON_MAX=20.0`, `VIX_RISK_OFF_MIN=25.0`,
  `EVENT_WINDOW_HOURS=24.0` — consistent with how this project already treats other exact spec numbers (e.g.
  `HistoricalAnalogEngine`'s N≥50 gate). Genuinely unspecified judgment calls ("VIX spiking" has no numeric
  definition anywhere in the source docs — the Layer 6 asset-volatility Spike definition is a different variable —
  and "correlations compressing toward 1.0" has no stated threshold either) are explicit, documented, tunable
  parameters, clearly labeled as non-spec.
- **Macro-Event Window takes precedence** over Risk-On/Risk-Off/Transitional (checked first, short-circuits) — a
  documented design decision, since MASTER_CONTEXT's table doesn't state an explicit precedence order but its "no
  new entries" language for an imminent event reads as unconditional caution, not something an otherwise-calm
  signal read should override.
- **`confidence` and `macro_weight` kept as two distinct output numbers**, not one, matching MASTER_CONTEXT's own
  separate note about correlation-driven relevance: `confidence` is a graded margin score (how deep past its
  classified state's thresholds the signals sit — deliberately not a trivial 1.0, since every classified state's
  defining conditions are by construction already satisfied); `macro_weight` is driven purely by `|BTC/QQQ
  correlation|`, independent of state and of confidence, per MASTER_CONTEXT's explicit "reduce macro weight... in
  low-correlation periods" instruction. Verified `|+0.9|` and `|-0.9|` correlation produce identical `macro_weight`
  (sign-independent — relevance is about co-movement existing at all, in either direction, not its sign), and that
  a Risk-On read and a Risk-Off read with the same `|correlation|` get the same `macro_weight`.
- **tz-safety applied proactively** in the event-window check (`_hours_until`), reusing the exact lesson from the
  `HistoricalAnalogEngine.predict()` tz-crash found and fixed earlier this session — normalizes both the snapshot
  timestamp and the caller-supplied event timestamp to UTC before comparing, tested with a tz-naive event against a
  tz-aware snapshot.

**One real bug caught before it shipped**: the first draft's `risk_off_score` accidentally summed the correlation
margin twice and divided by 4, when Risk-Off has only three defining conditions (DXY, VIX, correlation — no
dominance leg, unlike Risk-On's four) and the correlation margin wasn't normalized relative to
`risk_off_correlation_threshold` (0.7) the way the DXY/VIX margins were normalized relative to their own
thresholds. Found by manual review immediately after writing the code, before any test was written against the
buggy version, and fixed to average over 3 threshold-relative components.

Added `tests/test_macro_regime.py` (21 tests): all four states reachable from synthetic input (the literal "Done
when" bar), Macro-Event Window precedence over otherwise-Risk-On signals, exact 24h boundary behavior (inclusive)
and tz-safety on the event check, `macro_weight` sign/state-independence, confidence gradation (deep vs. marginal
Risk-On, genuinely-mixed Transitional), US10Y non-effect-but-echoed, and input validation. Full suite: 81/81 pass.

### Next Steps

- Layer 4 (MTF Alignment) and Layer 5 (Liquidity Trap heuristic) remain unbuilt — `macro_mtf_liquidity_vol/` now
  has one of three planned modules.
- This module is not yet wired into `WalkForwardBacktester` (via `regime_col`) or `PredictiveCoreStrategy` — doing
  so is what would make the Layer 14 `risk_on_regime_vs_aggregate` kill criterion (currently always
  `not_evaluated`) actually computable, and is a natural next step once Layer 4/5 exist alongside it.
- Everything else carried over from prior entries (the compression/sweep signal investigation is paused, not
  resolved; the `data/normalized/` convention and `is_outlier`/`gap_class` handling are still undecided) is
  unchanged.

## Session Update: MTF Alignment Engine Built (Layer 4) (2026-08-13)

Built `src/macro_mtf_liquidity_vol/mtf_alignment.py` (`score_mtf_alignment`, `MTFAlignmentResult`) — second of three
planned `macro_mtf_liquidity_vol/` modules (Layer 5 Liquidity Trap still unbuilt).

**A real spec gap found and resolved before writing any code**: MASTER_CONTEXT's scoring table is inherently a
4-point scale, but neither MASTER_CONTEXT.md nor the Prompt Playbook's build instruction says *which* four
timeframes. Checked the Adaptive Probabilistic Trade Management System doc's own Layer 4 section directly, which
states it explicitly: **1H, 4H, 1D, 1W**. That same document also states "MVP scope (Phase 1 only): BTC and ETH
OHLCV at 1H and 4H. Everything else is added in Phase 2" — confirmed against `config.yaml` (`crypto.timeframes:
["1h", "4h"]`) that 1D/1W data ingestion genuinely doesn't exist anywhere in this codebase yet. The module is built
completely and correctly against the real 4-timeframe spec regardless (this is Phase 2 work, same as Layer 3), and
`score_mtf_alignment` validates that exactly 4 readings are supplied, raising rather than silently accepting fewer
— but a real invocation with genuine 1D/1W data isn't possible until that ingestion work happens. Documented
prominently in the module docstring, the same pattern used for Layer 3's economic-calendar and US10Y gaps.

**Scope boundary, mirroring Layer 3's "classifier, not forecaster" split**: the Adaptive doc's own Layer 4
description says the Layer 2 feature vector "is evaluated at each timeframe... independently" and alignment is
then "scored" — i.e. this module scores agreement across four *already independently derived* per-timeframe
directional reads (a signed float in [-1, 1] per timeframe: sign = direction, magnitude = conviction). It
deliberately does NOT derive a directional call from a raw Layer 2 feature vector itself (e.g. from
`struct_bos_bullish`/`struct_trend_state`/momentum features) — that translation is a separate, genuinely
underspecified design question left for whoever wires a real caller up to this function, not invented here.

**Design decisions**, both documented as non-spec judgment calls: what counts as "neutral" per timeframe
(`DEFAULT_NEUTRAL_BAND=0.15`, since neither MASTER_CONTEXT nor the Adaptive doc quantifies this), and how ties are
handled (`dominant_direction="mixed"` when bullish/bearish counts are equal — verified this naturally produces the
*correct* score for the canonical case: a clean 2-2 split scores exactly 2/4, which is a precise semantic match to
the table's own "mixed structure, no clear directional case" wording, not a coincidence patched over). One
deliberate, flagged tradeoff: a 1-1 tie with 2 neutral readings also lands at score=1, getting the same
conservative "no entry, review existing positions" action as a genuine 4-way structural conflict, even though
"mostly inconclusive" and "severe conflict" are qualitatively different pictures — kept because MASTER_CONTEXT's
own capital-preservation-first philosophy favors caution over a more elaborate disambiguation that isn't specified
anywhere.

`MTFAlignmentResult.__post_init__` enforces score/action/trade_permitted internal consistency structurally (raises
if e.g. score=4 is paired with the wrong action string, or `trade_permitted` disagrees with `score >= 3`) — the
same "can't silently drift out of sync" discipline used elsewhere in this codebase (e.g. `CriterionResult`'s status
validation in `kill_criteria.py`).

Added `tests/test_mtf_alignment.py` (24 tests): all four score bands map to their exact table actions (the literal
"Done when" bar), `trade_permitted` gating (true only for 3/4 and 4/4, per the Adaptive doc's explicit "a trade is
only considered when alignment is 3/4 or 4/4"), exactly-4-timeframes validation, tie-handling including the 2-2 and
1-1-with-neutrals cases, confidence gradation, and the configurable neutral band. Full suite: 105/105 pass.

**Found while researching this task, not yet acted on — flagging for a separate decision**: the Adaptive doc's
Layer 3 section (macro regime) states an exact correlation-based macro-weight rule that wasn't visible when
`macro_regime.py` was built last session: *"When correlation is high (> 0.6), the macro signal is applied at full
weight. When correlation is low (< 0.3), the macro signal weight is reduced proportionally."* The already-built
`_macro_weight` function instead uses a generic linear scale (`|correlation| / correlation_relevance_scale`,
default scale 0.5) that wasn't checked against this more specific text at the time (the search that session found
MASTER_CONTEXT's general note but not this more precise Adaptive-doc passage). Not fixed here — this session's task
was Layer 4, not revisiting Layer 3 — but it's a real, concrete discrepancy between an already-shipped, tested
module and a source doc passage that should have grounded it. Worth a deliberate follow-up.

### Next Steps

- Layer 5 (Liquidity Trap heuristic) remains the last unbuilt `macro_mtf_liquidity_vol/` module.
- Decide whether/how to fix `macro_regime.py`'s `_macro_weight` against the Adaptive doc's exact 0.6/0.3
  correlation thresholds (see finding above) — this touches previously-tested, shipped code, so treat it as its
  own scoped task with its own review, not a quick inline patch.
- Neither Layer 3 nor Layer 4 is wired into `WalkForwardBacktester`/`PredictiveCoreStrategy` yet — both are
  standalone, tested, spec-correct modules waiting for an integration point.
- Everything else carried over from prior entries is unchanged.

## Session Update: Critical Review of `macro_regime.py` — Correlation Thresholds Fixed (2026-08-13)

Requested explicitly as a self-audit ("review it as a senior AI engineer... check your own work... critical...
aligns with everything mentioned in the documents"), scoped to `macro_regime.py` primarily, with `mtf_alignment.py`
and `macro_fetcher.py` also spot-checked. Re-extracted and re-read ALL of MASTER CONTEXT.docx, the Adaptive
Probabilistic Trade Management System doc's full Layer 3 section (not just the grep snippets used when the module
was first built), and the Prompt Playbook, specifically hunting for anything macro-related the first build might
have missed.

### Fixed: the flagged correlation-weight discrepancy from the prior entry

Found the exact source: the Adaptive doc's Layer 3 section states plainly — "When correlation is high (> 0.6), the
macro signal is applied at full weight. When correlation is low (< 0.3), the macro signal weight is reduced
proportionally... This is a dynamic weight, not a binary on/off." This was independently corroborated by a worked
example elsewhere in the same document ("BTC/QQQ corr: 0.64 → macro weight: FULL"), giving two independent
confirmations of the same 0.6 breakpoint.

- Added `CORRELATION_FULL_WEIGHT_THRESHOLD = 0.6` and `CORRELATION_REDUCED_WEIGHT_THRESHOLD = 0.3` as exact,
  hardcoded constants (removed `correlation_relevance_scale` as a tunable parameter — it was standing in for a
  number that turned out to actually be specified).
- `_macro_weight` now implements a single linear ramp from (corr=0, weight=0) to (corr=0.6, weight=1.0), clipped at
  1.0 above 0.6 — reproduces "> 0.6 → full weight" exactly, and yields weight=0.5 at exactly corr=0.3 (documented as
  "consistent with, not uniquely pinned down by" the "< 0.3 → reduced" text, since the source gives only one
  precise breakpoint, not two — a three-piece function would need an invented second parameter the docs don't
  supply, so wasn't built).
- Direct regression test added reproducing the worked example verbatim: `btc_qqq_corr_20d=0.64` → `macro_weight ≈
  1.0`.

### Two additional gaps found during the same re-read, fixed in the same pass

The Adaptive doc's Layer 3 "Output:" line explicitly lists four required outputs: "macro regime label, macro
confidence score (0–1), macro-event window flag, current BTC/QQQ correlation coefficient and derived macro
weight." The first build satisfied two of these (label as `state`, confidence as `confidence`, weight as
`macro_weight`) but was missing two fields entirely:

- **`MacroRegimeResult.btc_qqq_corr_20d`** — the raw correlation coefficient, now echoed on every result (previously
  only the *derived* `macro_weight` was exposed, not the coefficient itself).
- **`MacroRegimeResult.is_macro_event_window`** — an explicit boolean, `== (state == "Macro-Event Window")`,
  structurally validated in `__post_init__` so it can never silently drift out of sync with `state`. MASTER_CONTEXT's
  own table presents Macro-Event Window as one of four mutually-exclusive states (matching this module's design),
  while the Adaptive doc's output line describes it as a separate flag — reconciled by keeping `state` as the single
  source of truth and adding this as a structurally-guaranteed-consistent derived convenience field, satisfying
  both documents' descriptions rather than picking one and dropping the other's explicit requirement.

### Checked and confirmed correct, not changed

- Every regime condition (DXY, VIX absolute thresholds 20/25, dominance, correlation) re-verified word-for-word
  against both MASTER_CONTEXT.docx and the Adaptive doc's full Layer 3 text — identical in both, matches the
  implementation exactly.
- The Macro-Event Window action string ("...existing positions get tighter stops") is byte-exact against
  MASTER_CONTEXT.docx. Noted (not fixed): the Adaptive doc's own phrasing is slightly fuller ("...tighter stops or
  standing exit instructions") — not contradictory, just more expansive. Kept MASTER_CONTEXT.md's shorter wording
  since that file is this project's explicitly-labeled "exact table" reference and has been authoritative all
  session.
- `macro_fetcher.py`'s ticker set (DXY proxy, US10Y, VIX, BTC dominance, SPX, QQQ) checked directly against the
  Adaptive doc's explicit list ("DXY daily, US10Y daily, VIX daily, BTC dominance daily, SPX/QQQ daily") — complete,
  no gaps.
- `mtf_alignment.py` spot-checked against the same Layer 4 text re-read in full this pass — no new discrepancies
  found beyond what was already documented when it was built.

### Testing

Added 6 new tests: exact 0.6/0.3 constants, the worked-example regression (corr=0.64 → weight≈1.0), weight exactly
1.0 at the threshold, weight exactly 0.5 at the 0.3 reference point, weight never exceeds 1.0 above threshold,
correlation coefficient echoed correctly, and `is_macro_event_window` consistency. All pre-existing tests passed
unmodified (none had hardcoded the old 0.5-scale exact `macro_weight` values — all were relative/comparison-based).
Full suite: 112/112 pass.

### Next Steps

- The `macro_fetcher`/`macro_regime`/`mtf_alignment` trio is now internally consistent and re-verified against
  every available source document. Remaining work is the same as before: Layer 5 (Liquidity Trap) unbuilt, neither
  Layer 3 nor Layer 4 wired into the backtester yet.
- Consider whether `mtf_alignment.py` deserves the same depth of re-audit `macro_regime.py` just got (this session
  only spot-checked it) — not done in this pass since the explicit request was scoped to the macro side.

## Session Update: Liquidity Trap Heuristic Built (Layer 5) (2026-08-13)

Built `src/macro_mtf_liquidity_vol/liquidity_trap.py` (`estimate_liquidity_trap`, `LiquidityCluster`,
`LiquidityTrapResult`) — third and last of the `macro_mtf_liquidity_vol/` trio planned this session. Grounded
against source docs before writing anything: `MASTER_CONTEXT.md`/`.docx` has NO table or numbers for this layer at
all (unlike Layers 3/4/6/17) — the full spec lives only in the Adaptive doc's Layer 5 section and the Prompt
Playbook's build instruction. Found the exact "trailing 50 candles" number and the precise sweep definition ("stop-
runs above equal highs or below equal lows, followed by a confirmed reversal") in the Adaptive doc's hypothesis
section, not its Layer 5 section — required reading both, not just the section with the layer's name on it.

**Scope kept disciplined**, same as `mtf_alignment.py`: the Prompt Playbook's actual instruction says "based on
equal highs/lows and wick behavior" — narrower than the fuller Adaptive-doc Layer 5 description, which additionally
lists Fair Value Gap detection, "liquidity void" detection, and previous-session-high/low tracking. None of those
are built here; documented explicitly as out of scope for this task rather than half-implemented.

**Also found, not acted on (relevant to the paused compression/sweep investigation, not this module)**: the
hypothesis section defines "post-compression" with a specific number -- "ATR at or below its 20th percentile" --
distinct from Layer 6's general Compressed state (<30th). Noted for whenever that paused investigation resumes.

### Four real bugs found via testing, not code review — this module's development is the clearest example yet
this session of why "looks right on read-through" isn't sufficient

Random-walk synthetic data was used for initial manual verification specifically because it's more adversarial
than hand-crafted data — and it immediately surfaced problems a clean read of the code did not:

1. **Chained clustering with unbounded span.** Proximity grouping compared each sorted value only to its immediate
predecessor, letting a cluster's total span drift without bound (100.0, 100.2, 100.4, ... each within tolerance of
its neighbor, first and last arbitrarily far apart). Silently merged nearly an entire 50-candle random walk into
one fake cluster. Fixed by anchoring every cluster to its lowest unclaimed member and bounding all members' distance
to that anchor, not to each other.
2. **"Didn't close through it" checked against the wrong reference.** The first draft checked a candle's close
against its OWN high/low (`close < high`) — trivially true for nearly every ordinary candle, so it filtered out
almost nothing. "Rejected at the level" only means something once a shared candidate level exists to be rejected
AT; fixed with a two-stage structure (group by proximity first, then filter members against the GROUP's shared
level).
3. **Resistance/support clusters not filtered by position relative to price.** A cluster from old price action
that price has since moved away from (e.g. a former high cluster now sitting below current price) was being
treated as active resistance — incoherent, since nothing is "above" for price to run into. Fixed by requiring
resistance clusters to sit above the reference price and support clusters below it.
4. **`min`/`max` reversed when identifying which level a sweep actually reached.** When a candle's high exceeds
several clusters at once, the code needs the FARTHEST (highest) one to test "confirmed reversal" against — not the
nearest one, which the wick trivially also exceeded along the way. Had `min()` where `max()` was needed (and vice
versa for support), so an engineered, deliberate sweep test kept failing to confirm even though the sweep was real
— caught via direct debugging of the actual cluster/sweep values, not by re-reading the formula.

All four were found and fixed in sequence by running the function against synthetic data and checking the output
made sense, not by reviewing the code a second time — each fix was verified by re-running the same scenario before
moving to the next bug. The final formal test suite uses fully deterministic, hand-crafted fixtures (a strictly-
increasing baseline with a wide per-step price gap) rather than random walks, specifically to avoid a fifth
near-miss: an early manual "control" check showed an unexpected sweep because injected fixture candles had an
unintended side effect on the OTHER side's clustering, which turned out to be a legitimate consequence of the
(now-correct) logic rather than a bug, but was confusing enough to debug that the formal suite avoids the ambiguity
entirely by isolating each test's fixture.

### Testing

Added `tests/test_liquidity_trap.py` (17 tests): the Done-when bar directly (trap_probability always a clean [0,1]
float, never a claimed exact price), cluster detection correctness (touch counts, close-through-level exclusion,
position-relative-to-price filtering), sweep confirmation in both directions plus the "wick breaches but doesn't
close back" non-confirmation case, no-lookahead (future candles can't affect the current evaluation), and input
validation. Full suite: 129/129 pass.

### Next Steps

- All three `macro_mtf_liquidity_vol/` modules (macro regime, MTF alignment, liquidity trap) are now built and
  individually tested. None are wired into `WalkForwardBacktester`/`PredictiveCoreStrategy` yet.
- `mtf_alignment.py` still hasn't had the same depth of critical re-audit `macro_regime.py` and (via extensive
  testing during development) `liquidity_trap.py` have now had.
- Everything else carried over from prior entries (paused compression/sweep investigation, `data/normalized/`
  convention, `is_outlier`/`gap_class` handling) is unchanged.

## Session Update: Volatility Regime Classifier Built (Layer 6) (2026-08-13)

Built `src/macro_mtf_liquidity_vol/volatility_classifier.py` (`classify_volatility_state`,
`VolatilityStateResult`). Unlike Layer 5, this layer's exact table lives in `MASTER_CONTEXT.md` (checked directly,
matches word-for-word) — the fuller Adaptive doc read still surfaced two things worth recording:

- Layer 2's own feature-list prose elsewhere in the Adaptive doc describes volatility state loosely ("expanding if
  ATR percentile > 60th, compressing if < 30th") — inconsistent with Layer 6's own exact table (Compressed <30th /
  Expanding 30th–70th / Extended >70th, a proper three-way partition). Checked `feature_calculator.py`'s already-
  built `vol_state_expanding` flag: it correctly implements the Layer 6 banded definition, not the looser Layer 2
  prose — confirming the codebase's existing convention (trust the exact table) rather than surfacing a real
  conflict to resolve.
- Adaptive doc's Compressed action text adds "...then scale up" beyond MASTER_CONTEXT.md's shorter wording — same
  treatment as every other minor cross-document wording gap found this session: noted, MASTER_CONTEXT.md's exact
  text used verbatim.

**Design**: deliberately asymmetric input handling. The percentile band (Compressed/Expanding/Extended) accepts an
ALREADY-COMPUTED `atr_percentile` scalar (Layer 2's `vol_atr_pct_rank` — this is a thin classifier, not a
re-derivation of a trailing-252-candle rolling rank). Spike is different: computed HERE, directly, from a short raw
ATR sequence (current + 5 candles back), using the exact formula `feature_calculator.py`'s own `vol_spike` flag
already uses. This split exists because the Done-when bar specifically requires "synthetic ATR sequences...
including the Spike condition specifically (doubling within 5 candles, not just a high percentile)" — which needs
this module's own tests to exercise the doubling arithmetic directly, not trust an external pre-computed flag.

**Precedence** (undocumented in the source tables, a judgment call): Spike is checked first and overrides whatever
the percentile band would otherwise say — mirroring the same "most severe state wins" precedence pattern used for
Macro-Event Window in `macro_regime.py`, and directly verified by tests where a LOW percentile (would read
Compressed) and a HIGH percentile (would read Extended) both correctly yield Spike when the raw sequence genuinely
doubles, and conversely a high percentile WITHOUT doubling correctly stays Extended, not Spike.

This module's development was clean — no bugs found via testing this time (unlike `liquidity_trap.py`'s four),
likely because the logic is genuinely simpler (a single threshold comparison plus one ratio check, vs. clustering
and multi-candle window logic) and because `feature_calculator.py`'s already-tested, already-shipped formulas were
reused directly rather than reimplemented from scratch.

### Testing

Added `tests/test_volatility_classifier.py` (21 tests): all four states directly (the Done-when bar), Spike
overriding both a low AND a high percentile reading, high-percentile-without-doubling staying Extended (proving
Spike and Extended are genuinely independent, not proxies), exact threshold boundaries (30th/70th/ratio=2.0, all
tested on both sides), result self-consistency validation, and input validation. Full suite: 150/150 pass.

### Next Steps

- All four `macro_mtf_liquidity_vol/` modules from this session's build list (macro regime, MTF alignment, liquidity
  trap, volatility classifier) are now built and individually tested. None are wired into
  `WalkForwardBacktester`/`PredictiveCoreStrategy` yet — that integration remains the natural next step whenever
  picked up.
- `mtf_alignment.py` is now the one module in this group that hasn't had a dedicated critical re-audit pass.
- Everything else carried over from prior entries is unchanged.

## Session Update: MTF Alignment Critical Re-Audit + Cross-Module Consistency Review (2026-08-13)

Gave `mtf_alignment.py` the same critical-re-audit treatment `macro_regime.py` got earlier this session, then
extended the review across all four `macro_mtf_liquidity_vol/` modules together, specifically hunting for
cross-module inconsistencies — both requested explicitly by name.

### Real bug found: asymmetric tie confidence

`score_mtf_alignment`'s "mixed" branch (an exact tie between bullish and bearish timeframe counts, e.g. 2-2 or
1-1) computed `confidence` from `bullish_tfs` only:

```python
elif n_bullish == n_bearish:
    dominant_direction = "mixed"
    score = n_bullish
    aligned_tfs = bullish_tfs          # BUG: drops the bearish side entirely
```

Since `confidence = mean(|value| for tf in aligned_tfs)`, this made confidence depend on which arbitrary side
happened to be labeled "bullish" rather than on how strongly conflicted the picture actually is. Concretely:
`{1H:+0.9, 4H:+0.9, 1D:-0.16, 1W:-0.16}` (a strong pair tied against a barely-non-neutral pair) reported
`confidence=0.9`, while the mirrored, structurally identical scenario `{1H:+0.16, 4H:+0.16, 1D:-0.9, 1W:-0.9}`
(same conflict, labels swapped) reported `confidence=0.16`. Two readings that are the same tie from opposite sides
must not get different confidence.

Fix: `aligned_tfs = bullish_tfs + bearish_tfs` — both scenarios now report the same value
(`mean(0.9, 0.9, 0.16, 0.16) = 0.53`). This is the exact same bug *class* the earlier `_deduplicate` work this
session kept surfacing — an operation that's supposed to be symmetric across two sides silently wasn't. Found by
directly probing the tie branch with mirrored synthetic inputs, not by code-reading alone — same lesson as
`liquidity_trap.py`'s four bugs earlier this session: testing found what review missed.

Verified no analogous bug exists in the other three modules by directly re-reading their combination logic:
- `macro_regime.py`'s Transitional confidence: `1.0 - max(risk_on_score, risk_off_score)` — `max()` is inherently
  commutative, not an arbitrary-side pick.
- `liquidity_trap.py`'s `trap_probability = max(resistance_risk, support_risk)` and
  `liquidity_attraction_score = (resistance_risk + support_risk) / 2.0` — same reasoning, both symmetric by
  construction.
- `volatility_classifier.py` has no tie/combination logic at all (single threshold comparison, single ratio check)
  — no analogous bug is possible.

### Cross-module consistency findings (3 fixed)

1. **Missing named constant for the exact spec threshold.** `mtf_alignment.py`'s `score >= 3` gate
   ("trade only considered when alignment is 3/4 or 4/4") was two bare literal `3`s (dataclass validation + return
   statement), while every other exact spec number in the sibling modules (`macro_regime.py`'s `VIX_RISK_ON_MAX`
   etc., `volatility_classifier.py`'s `COMPRESSED_MAX_PERCENTILE` etc., `liquidity_trap.py`'s
   `EQUAL_LEVEL_LOOKBACK`) is a named, top-level, directly-testable constant. Fixed: added
   `TRADE_PERMITTED_MIN_SCORE = 3`, replaced both literals, exported it, added
   `test_exact_spec_trade_permitted_threshold_is_3`.
2. **Unexplained default on a required-everywhere-else field.** `MTFAlignmentResult.reasons` had
   `= field(default_factory=list)` while the analogous `reasons` field on `MacroRegimeResult`,
   `LiquiditySweepResult` (liquidity_trap.py), and `VolatilityStateResult` are all required, no-default fields.
   `score_mtf_alignment` always supplies it explicitly anyway, so the default served no purpose but to let direct
   `MTFAlignmentResult(...)` test construction skip it. Fixed: removed the default (and the now-unused `field`
   import), updated all direct-construction call sites in `test_mtf_alignment.py` to pass `reasons=[]` explicitly,
   added `test_reasons_field_has_no_default_matching_sibling_modules`.
3. **Judgment-call defaults as bare literals instead of named constants.** `macro_regime.py`'s
   `classify_macro_regime` signature had five judgment-call thresholds
   (`vix_spike_5d_roc_pct`, `risk_off_correlation_threshold`, `confidence_scale_dxy_pct`, `confidence_scale_vix_roc_pct`,
   `confidence_scale_dominance_pct`) as bare inline literal defaults, unlike `mtf_alignment.py`'s
   `DEFAULT_NEUTRAL_BAND` / `liquidity_trap.py`'s `DEFAULT_LEVEL_TOLERANCE_ATR` etc. pattern of exposing
   judgment-call numbers as named, importable, top-level `DEFAULT_*` constants. Fixed: added
   `DEFAULT_VIX_SPIKE_5D_ROC_PCT`, `DEFAULT_RISK_OFF_CORRELATION_THRESHOLD`, `DEFAULT_CONFIDENCE_SCALE_DXY_PCT`,
   `DEFAULT_CONFIDENCE_SCALE_VIX_ROC_PCT`, `DEFAULT_CONFIDENCE_SCALE_DOMINANCE_PCT`, updated the function signature
   to reference them. Purely a naming/discoverability fix — values unchanged, confirmed by `test_macro_regime.py`'s
   28 tests passing unmodified.

### Checked and confirmed NOT inconsistencies (deliberately left as-is)

- `_TRADING_IMPLICATIONS` (macro_regime.py, volatility_classifier.py) vs `_ACTIONS` (mtf_alignment.py), and the
  corresponding `trading_implication` vs `action` result fields: different names because MASTER_CONTEXT's own
  tables use different column headers per layer ("Trading Implication" for Layers 3/6, "Action" for Layer 4) — this
  is each module correctly mirroring its own spec's vocabulary, not drift.
- All four `@dataclass(frozen=True)` result types, all four modules' `reasons: List[str]` field typing, and all
  `max()`/`min()` usage in `liquidity_trap.py` (all "farthest extent reached" sweep-detection semantics, never
  arbitrary side-picking) — checked directly via grep and source re-read, no issues.

### Testing

`test_mtf_alignment.py` grew from 24 to 27 tests: `test_tie_confidence_is_symmetric_regardless_of_which_side_is_stronger`
(regression test for the bug above, asserts both mirrored scenarios yield identical confidence equal to the
4-way mean, not either 2-way sub-mean), `test_exact_spec_trade_permitted_threshold_is_3`,
`test_reasons_field_has_no_default_matching_sibling_modules`. Full suite: **153/153 pass**
(`python -m pytest tests/ -q --ignore=tests/test_binance_real_data.py`).

### Next Steps

- All four `macro_mtf_liquidity_vol/` modules have now each had a dedicated critical re-audit
  (`macro_regime.py`, `mtf_alignment.py`) or extensive bug-finding-via-testing during initial development
  (`liquidity_trap.py`'s four bugs, `volatility_classifier.py` clean). Cross-module consistency has been explicitly
  checked and is confirmed good as of this entry.
- None of the four modules are wired into `WalkForwardBacktester`/`PredictiveCoreStrategy` yet — that integration
  is the natural next step whenever picked up.
- Everything else carried over from prior entries (paused compression/sweep investigation, `data/normalized/`
  convention, `is_outlier`/`gap_class` handling, the noted-not-acted-on wording discrepancies between
  MASTER_CONTEXT.md and the Adaptive doc) is unchanged.

## Session Update: Ensemble Meta-Model Built (Layer 7, Component C) (2026-08-13)

Built `src/predictive_core/ensemble.py` (`EnsembleMetaModel`, `EnsembleSignalSet`, `ComponentBSignal`,
`EnsembleForecast`, plus the pure functions `blend_signals`, `fit_ensemble_weights`, `check_weight_stability`,
`equal_weights`). Grounded against the Adaptive doc's Layer 7 Component C section and Problem 4's solution text
(read directly, not from memory) and the Prompt Playbook's 2E build entry, which supplied the exact Done-when bar
and Cursor-review criteria quoted in the build task.

**Architecture**: strictly `sum(w_i * p_i)` over the five approved inputs (Component A, Component B, macro regime,
MTF alignment, liquidity trap), weights on a simplex (`scipy.optimize.minimize`, SLSQP, bounds [0,1] + sum-to-1
constraint), fit via mean multiclass log-loss minimization on the validation fold only. This is the ONLY learned
parameter set in the module — every per-signal "turn this result object into a [down, neutral, up] distribution"
conversion is a fixed, deterministic, documented formula, never itself fit, which is what keeps "no non-linear
stacking, no secondary model" true rather than just asserted.

**Key design decisions, each a documented judgment call (source docs establish these five belong in the linear
combination but give no conversion formula)**:
- Component A (`AnalogForecast`) and Component B (new `ComponentBSignal` — no per-row Component B output type
  existed upstream; `boosted_core.py`'s result dataclasses are fold-level training summaries only) already produce
  genuine 3-class distributions, used directly.
- Macro regime: Risk-On/Risk-Off → bullish/bearish lean scaled by `confidence * macro_weight` (both Layer 3 outputs
  the Adaptive doc's Input line names explicitly — "macro regime score AND dynamic weight"); Transitional AND
  Macro-Event Window → full neutral (Macro-Event Window's `confidence=1.0` is certainty about the *state*, not a
  directional call — deliberately not read as directional conviction, a distinction worth flagging since it's easy
  to miss).
- MTF alignment: bullish/bearish → lean scaled by `confidence`; "mixed" (a genuine tie) and "neutral" → full
  neutral.
- Liquidity trap: directional ONLY when `sweep_just_confirmed` (using `sweep_direction`, scaled by
  `trap_probability`); otherwise full neutral regardless of how high `trap_probability` reads, since the Adaptive
  doc frames this signal as adjusting "entry timing risk," never as directional confirmation the way MTF alignment
  is — a genuinely different role from the other three directional inputs, worth keeping distinct rather than
  force-fitting a directional read that isn't there.
- Component A is the one signal that can be fully EXCLUDED (not blended toward neutral) for a given row —
  `is_valid=False` means "no forecast is issued from this component" (analog_engine's own wording), so there is no
  estimate to blend; its row-level weight is renormalized across the remaining four signals rather than silently
  vanishing. When valid, "forecasts with N close to the minimum threshold are weighted lower... automatically" (no
  formula given) is implemented as a linear confidence ramp from 0 at `N == 50` to 1 at `N == 100` (2x the gate,
  `DEFAULT_ANALOG_CONFIDENCE_RAMP_MULTIPLE`), also fixed, not learned.
- Magnitude/duration quantiles pass through from Component B completely unensembled — no other one of the five
  inputs produces magnitude or duration, so there is nothing to blend against; ensembling a single source against
  nothing would be a no-op dressed up as a computation.
- The Adaptive doc's "Final output" list mentions a 95% CI on the direction probability; deliberately NOT computed
  here — no source doc gives a methodology for deriving one from a blend of five heterogeneous point estimates, and
  Layer 8 (Explainability Engine), which is what actually renders that CI in the spec's own example output, is
  explicitly out of scope for Phase 1. Fabricating a CI method here would be exactly the kind of unstated,
  invented rule this project's other modules have consistently avoided (US10Y in macro_regime.py, precise price
  levels in liquidity_trap.py).

**Weight stability**: exact Adaptive-doc number, `WEIGHT_INSTABILITY_THRESHOLD = 0.15` ("any single weight changing
by more than 0.15 between consecutive folds") — noted explicitly in the module docstring that two illustrative
example lines elsewhere in the Adaptive doc ("STABLE (all fold deltas < 0.10)", "max fold delta: 0.09") are flavor
text for a sample output, not a second competing threshold, to preempt exactly the kind of confusion a careful
review would otherwise flag. `EnsembleMetaModel.fit()` checks stability after every fold and switches to
`equal_weights()` the moment instability triggers; per the spec's "until diagnosed" wording, the fallback is
persistent — `reset_after_diagnosis()` is a separate, explicit call, so a later fold's own individually-stable-
looking weights do NOT auto-clear the fallback (tested directly: `test_fallback_persists_even_if_a_later_fold_looks_individually_stable`).

**Done-when bar** ("feeding a raw Layer 2 feature into the ensemble constructor raises an error rather than
silently being accepted"): satisfied structurally, not by convention. `EnsembleSignalSet` is a frozen dataclass
with exactly five fields, each `isinstance`-checked in `__post_init__` against its specific upstream result type
(`AnalogForecast`, `ComponentBSignal`, `MacroRegimeResult`, `MTFAlignmentResult`, `LiquidityTrapResult`) — a raw
float, dict, or pandas Series fails the type check; an extra keyword argument (e.g. a smuggled-in raw feature)
fails Python's own dataclass `__init__` (no `**kwargs`, no catch-all field). Tested directly for all five fields
individually (`test_wrong_type_for_every_field_individually_raises`), not just Component A.

Added `"scipy"` as an explicit `pyproject.toml` dependency (used directly for the constrained weight optimization;
previously only present transitively via scikit-learn).

### Testing

Added `tests/test_ensemble.py` (44 tests): the Done-when bar and all three Cursor-review criteria quoted in the
build task (raw-feature rejection across all five fields; `fit_ensemble_weights` demonstrated to assign higher
weight to a genuinely-informative signal over four deliberately-uninformative ones, on a 200-row synthetic
validation set; the exact 0.15 threshold triggering instability and fallback correctly, including the "persists
until diagnosed" behavior) — plus per-signal conversion-formula tests (Macro-Event Window's confidence=1.0 NOT
read as directional; MTF "mixed" full-neutral despite nonzero confidence; liquidity trap full-neutral without a
confirmed sweep regardless of probability), Component A exclusion/renormalization and N-ramp monotonicity, exact
linear-blend arithmetic against a hand-computed expected value, and `EnsembleForecast`/`ComponentBSignal`
self-consistency validation. Full suite: **197/197 pass**
(`python -m pytest tests/ -q --ignore=tests/test_binance_real_data.py`).

### Next Steps

- The Ensemble Meta-Model is not yet wired into `PredictiveCoreStrategy` — that adapter's own
  `ensemble_weights()` method currently returns `None` with a docstring noting exactly this ("No formal linear
  ensemble meta-model exists yet (Layer 7C, Phase 2)"); `generate_positions()` still combines Component A/B via a
  hard agreement rule, not `EnsembleMetaModel`. Wiring this in is real integration work (deciding how macro
  regime/MTF/liquidity trap snapshots get computed per-row inside a walk-forward backtest, none of which
  `PredictiveCoreStrategy.fit()`/`generate_positions()` currently constructs) — a natural next step, not done here.
- `min_independent_instances` on `EnsembleMetaModel` must be kept in sync with whatever `HistoricalAnalogEngine`
  was actually configured with, same cross-module contract as `PredictiveCoreStrategy.label_horizon` vs.
  `apply_triple_barrier`'s `max_candles` — not cross-checked at runtime (⁠`AnalogForecast` doesn't carry the gate
  value it was evaluated against), documented instead.
- Everything else carried over from prior entries is unchanged.

## Session Update: Ensemble Meta-Model Cursor Review (2026-08-13)

Reviewed `ensemble.py` against the Prompt Playbook's exact 2E review criteria (quoted verbatim in the review task).
Found and fixed one real, confirmed-by-reproduction bug; the other two criteria checked clean against the existing
design, documented honestly rather than glossed over.

**(1) Raw Layer 2 feature sneaking in — one real gap found and fixed, rest confirmed clean.** Traced every field
actually read by `_row_distributions`/`blend_signals` back to its source: only `AnalogForecast`'s
probability/N fields, `ComponentBSignal`'s probability/quantile fields, and a small, explicitly-named subset of
each Layer 3/4/5 result's own fields (`state`/`confidence`/`macro_weight`; `dominant_direction`/`confidence`;
`sweep_just_confirmed`/`sweep_direction`/`trap_probability`) are ever touched — never `btc_qqq_corr_20d`,
`per_timeframe_direction`, cluster levels, or any other passenger field these result objects happen to carry. The
one real gap: `blend_signals` (exported at module level, called directly throughout this module's own tests) did
NOT itself validate its `signals` argument was an `EnsembleSignalSet` — a raw dict would have failed with a
confusing `AttributeError` deep inside `_row_distributions` rather than a clear `TypeError` naming the problem.
`EnsembleMetaModel.predict()` and `fit_ensemble_weights()` both already isinstance-check before ever reaching
`blend_signals`, so this was never reachable through the model's own public surface, but it was reachable by any
caller using `blend_signals` directly. Fixed: added the same isinstance guard directly to `blend_signals`.

**(2) Weights provably never touched by training/test fold data — confirmed to the same standard as existing
precedent, documented honestly rather than overclaimed.** `fit_ensemble_weights`/`EnsembleMetaModel.fit()` have no
way to independently verify which walk-forward partition they were handed — same limitation
`HistoricalAnalogEngine.fit()` already documents for its own training-only requirement (that module can't verify
its caller passed the training partition either). Re-traced `_mean_log_loss_for_weights`'s closure and the
`scipy.optimize.minimize` argument order to confirm no accidental extra data source or misthreaded argument exists
inside the objective function itself — clean. The honest answer to "provably": enforced by interface naming
(`validation_signals`/`validation_direction`) and caller discipline, not a runtime check, matching the accepted
standard used everywhere else train/val/test isolation matters in this codebase (the actual leakage prevention is
architectural — `WalkForwardBacktester` only ever has the correct slice in scope at the point it calls `fit()`).
Not treated as a bug to fix, since it isn't a regression from any existing standard — but worth stating plainly
rather than claiming a stronger guarantee than actually exists.

**(3) The 0.15 threshold and equal-weight fallback — one real, serious bug found by direct reproduction (not just
code reading) and fixed.** `EnsembleMetaModel.fit()` re-derived the fallback decision by calling
`check_weight_stability()` on the ENTIRE cumulative `_fold_weight_history` on every call. Since
`check_weight_stability` reports UNSTABLE if ANY pair anywhere in the list it's given exceeds 0.15, this meant one
old, already-flagged instability (say, fold 0 → fold 1) would permanently re-win on every subsequent `fit()` call,
regardless of how stable the actual new transition was — making `reset_after_diagnosis()` silently undone by the
very next fold. Reproduced directly before fixing: after `reset_after_diagnosis()`, fitting a fold whose own
transition had `delta=0.0` still flipped `is_using_fallback` back to `True`, with `triggering_fold_pair` still
pinned at the original `(0, 1)`. Fixed: `fit()` now checks only the NEWEST fold-to-fold transition
(`self._fold_weight_history[-2:]`) to decide whether to (keep) trigger(ing) the fallback — `is_using_fallback`
is now a genuine sticky flag driven only by new evidence or an explicit `reset_after_diagnosis()` call, never
re-derived from the full history. The full-history view is still available, unharmed, via a new
`full_history_stability_report()` method — for a human actually doing the "diagnosis" the spec's fallback-exit
condition refers to, not for `fit()`'s own automatic decision.

### Testing

Replaced the previous fallback tests (which manually forced internal flags rather than exercising real sequential
`fit()` calls, and had a conditional `if status == "UNSTABLE"` assertion that could silently pass without ever
checking the fallback path) with deterministic, seed-seeded end-to-end sequences using two mirror-image synthetic
folds (`_informative_fold`, `_swing_fold`) that reliably trip instability. Added
`test_reset_after_diagnosis_survives_the_next_stable_fit_call` as a direct regression test for the bug above.
`tests/test_ensemble.py`: 45 tests (was 44 — net +1: two new/rewritten tests, one old weak test removed). Full
suite: **198/198 pass** (`python -m pytest tests/ -q --ignore=tests/test_binance_real_data.py`).

### Next Steps

- Same as the prior entry (Ensemble Meta-Model not yet wired into `PredictiveCoreStrategy`).
- Everything else carried over from prior entries is unchanged.

## Session Update: Explainability Engine Built (Layer 8) (2026-08-13)

Built `src/predictive_core/explainability.py` (`ExplainabilityEngine`, `SignalContribution`, `TradeRiskSummary`,
`AnthropicProseGenerator`, plus the pure functions `contributions_from_ensemble`, `direction_probability_ci`).
Grounded against the Adaptive doc's Layer 8 section (read directly, including its full example output template)
and the Prompt Playbook's 2F build entry. Also made a small, honest refactor to `ensemble.py`: added
`blend_signals_detailed()` (returns the final blend PLUS each present signal's `(normalized_row_weight,
distribution)` pair), with `blend_signals()` now a thin wrapper around it — needed so Layer 8's per-signal
contribution breakdown reuses the ensemble's exact blend arithmetic instead of re-deriving the same non-trivial
per-signal logic (macro state→lean, MTF mixed→neutral, liquidity trap sweep-gating, Component A's N-ramp/exclusion)
a second time, which is exactly the kind of duplication this project's cross-module reviews have caught drifting
out of sync elsewhere.

**This module computes nothing new about the trade itself — it's a renderer.** Every number in its output is
either read directly from `EnsembleForecast`/`WeightStabilityReport`, or derived by an exact, provable formula from
one:

- **Signal contributions** (provable, not invented): `impact_pct_i = 100 * w_i * (p_i[called] - p_i[opposite])`
  per signal, using each signal's actual row-level (post-renormalization) weight and its own `[down, neutral, up]`
  distribution — both taken directly from `blend_signals_detailed`. Summing all five equals exactly `100 *
  (p_final[called] - p_final[opposite])`, verified directly in the test suite
  (`test_contributions_sum_to_the_called_vs_opposite_net_lean`), not merely asserted. A signal that disagrees with
  the ensemble's called direction, even while outweighted, reads as genuinely negative under this single uniform
  formula — this is how the format's +/- mix arises, without needing two different formulas for the positive and
  negative sides.
- **Deliberately NOT reconstructed**: the Adaptive doc's own example shows contribution lines that aren't among the
  ensemble's five formal inputs at all (e.g. "Market structure (BOS + CHOCH confirmed, 4H)", "RSI bearish
  divergence active (4H)") — raw Layer 2 features the ensemble deliberately never touches. No source doc gives a
  formula for turning a raw feature into a percentage-point contribution (`feature_importance` is a fold-level
  TRAINING statistic, not a per-prediction attribution — using it as one would be a real methodological error, not
  just an approximation). `contributions_from_ensemble` only computes the five formal inputs; a caller with
  additional diagnostic signals (e.g. `VolatilityStateResult`) can construct extra `SignalContribution` entries
  directly and pass them into `explain()` — this module renders whatever it's given, from any source.
- **95% CI**: a Wilson score interval (standard, not invented) over Component A's own `analog_n_independent`,
  explicitly labeled as an approximation (N genuinely belongs to only one of the five blended signals) — returns
  "unavailable" rather than fabricating a CI when Component A didn't contribute a valid forecast. This was the
  exact question `ensemble.py`'s own docstring deferred to "whenever Layer 8 is built" — resolved here.
- **EV gate**: exact MASTER_CONTEXT number, `EV_LOWER_CI_GATE_PCT = 0.15` (constraint 7 / constraint 8's six-gate
  requirement — confirmed by direct grep, not assumed), strict `>` not `>=`.
- **EV estimate, EV lower CI bound, position size**: explicitly NOT computed here — Layer 9 (Risk Engine) / Layer
  10 (Position Sizing) outputs, neither built yet in this codebase. Accepted as an already-computed
  `TradeRiskSummary` the caller supplies, mirroring how `mtf_alignment.py` accepts already-derived directional
  reads instead of computing them from raw features.

**Live prose generation, numeric-safety boundary.** `AnthropicProseGenerator` calls the Anthropic API (added as a
new `pyproject.toml` dependency; installed and confirmed working — `anthropic==0.121.0`) with a cheap/fast model
(`claude-haiku-4-5-20251001` default, matching the Prompt Playbook's "Haiku-tier API call for live text"
instruction) to phrase each contribution's ALREADY-COMPUTED label/detail into a short natural-language line. The
model is never shown, and never asked to produce, any actual number (percentage, N, EV, position size) — it only
returns JSON mapping each contribution's id to a prose phrase; every number in the final report always comes from
this module's own arithmetic regardless of which prose path produced the surrounding text. Verified directly (not
just asserted): an injected generator that returns a phrase containing a fabricated `+999%` still produces a
report where the REAL trailing percentage for that line is the correct computed `impact_pct`
(`test_live_api_cannot_smuggle_wrong_numbers_into_the_report`). The client is constructed lazily (only on an actual
API call attempt, never at `__init__`), so building an `ExplainabilityEngine(enable_live_api=True)` never requires
an API key by itself.

**Done-when bar** ("output format matches the structure above and degrades gracefully if the API call fails"):
verified two ways. Format: `report_text` reproduces the example's structure line-for-line (asset/setup header,
probability+CI line, "Signal contributions:" section with aligned +/- percentages, EV lines with gate check,
position size, stability line) — visually confirmed against a rendered sample, matching closely. Graceful
degradation: ANY failure of the live path (network error, auth error, timeout, malformed/incomplete JSON, or any
exception a caller-injected generator raises) is caught uniformly and falls back to a deterministic template for
EVERY contribution (all-or-nothing, never a partial mix), with `used_live_api=False` reported so the fallback is
visible rather than silent. `enable_live_api=False` (the "config flag to swap in a template-based fallback") skips
the live path entirely — verified the generator is never even invoked when the flag is off, via an injected
generator that raises `AssertionError` if called at all.

### Testing

Added `tests/test_explainability.py` (29 tests). No test makes a real network call — the live-API path is always
exercised through dependency-injected fake generators (success, exception-raising, incomplete/malformed response,
empty-string phrases), never the real `AnthropicProseGenerator`. Covers: the provable sum identity, contribution
sorting/filtering, the exact +0.15% gate boundary (strict `>`), Wilson CI correctness and its
`analog_is_valid=False` → `None` case, the full Done-when bar (format + all four degradation paths), and the
numeric-safety boundary test above. One fixture correction made along the way: an early version of the test
fixture fit a real `EnsembleMetaModel` on a synthetic fold where `macro_regime` was held constant — log-loss
minimization correctly learned to assign it ~0 weight (correct ensemble behavior), which broke two tests that
needed every one of the five signals to carry a real contribution. Fixed by building test forecasts directly via
`blend_signals_detailed` with an explicit, known weight vector (equal weights by default) instead of relying on
optimizer output for these format-focused tests — optimizer behavior itself is already covered by
`test_ensemble.py`. Full suite: **227/227 pass** (`python -m pytest tests/ -q --ignore=tests/test_binance_real_data.py`).

### Next Steps

- The Explainability Engine is not wired into `PredictiveCoreStrategy`/`WalkForwardBacktester` — no caller in this
  codebase currently constructs `TradeRiskSummary` (Layer 9/10 don't exist) or calls `explain()` end-to-end from a
  real backtest row.
- `AnthropicProseGenerator` has not been exercised against the real Anthropic API in this session (no
  `ANTHROPIC_API_KEY` is set in this environment) — only the fallback/injection paths have been verified. The live
  path's actual prompt/response behavior should be sanity-checked against a real API call before relying on it in
  production.
- Everything else carried over from prior entries is unchanged.

## Session Update: Risk Engine Built (Layer 9) (2026-08-13)

Built `src/risk_sizing_selection/risk_engine.py` (`assess_risk`, `RiskEngineInputs`, `RiskAssessment`, plus the
pure functions `_loss_probability`, `_gap_risk_probability`, `_liquidity_trap_contribution`, `_correlation_flag`,
`_compute_risk_score`). Grounded against the Adaptive doc's Layer 9 section (read directly) and the Prompt
Playbook's 2G build entry, whose prereqs (1C = Component A+B, 2D = Volatility Classifier) were both already built.
First module in `src/risk_sizing_selection/` — the package `__init__.py` already existed as an empty scaffold from
earlier project setup.

**P(loss > 1%/2%/5%/10%) — a documented judgment call, not a spec formula.** The Adaptive doc says these are
"derived from the 10th-percentile magnitude estimate from Component B... combined with the stop-loss placement
from Layer 11" but gives no closed-form equation. Traced `labeling.py`'s `apply_triple_barrier` to confirm exactly
what Component B's "magnitude" actually measures: MFE (Maximum Favorable Excursion) toward the REALIZED direction
— there is no adverse/loss-magnitude quantile anywhere in this codebase, so the 10th-percentile magnitude is used
as a general move-size *scale*, not literally a loss estimate. Built a two-regime exponential-decay exceedance
model split at the stop-loss boundary: below the stop, ordinary path risk decays smoothly; beyond it, further loss
can only happen via a gap-through event, so it's gated by `gap_risk_probability` — this is exactly why gap risk is
a separately-named "tail risk component" in the source doc rather than one continuous curve. A monotonicity guard
(`min(gap_risk_probability, prob_at_stop)`) prevents a real bug class: without it, a large stop distance relative
to the magnitude scale could make P(loss > T) jump *up* just past the stop boundary, which isn't a valid
probability curve — caught by reasoning through the math before writing code, then verified by a dedicated
regression test (`test_loss_probability_monotonicity_guard_when_gap_risk_exceeds_at_stop_probability`) with inputs
specifically chosen to trigger the failure mode if the guard were removed.

**Gap risk** — session-classified (UTC weekend / overnight 00:00–06:00 / weekday-intraday), NOT literally historical
gap frequency (the spec's "estimated from historical gap frequency in the asset at the relevant session times"
would require a dedicated historical-gap-analysis pipeline this codebase doesn't have — out of scope for this
module, documented explicitly rather than silently faked). Crypto trades 24/7, so "weekend"/"overnight" here means
thinned liquidity per the spec's own framing, not a closed exchange.

**Liquidity trap risk** — direct reuse of Layer 5's own `LiquidityTrapResult.trap_probability`, scaled by position
size per the spec's explicit instruction ("more relevant at larger position sizes; less relevant for BTC/ETH majors
at retail sizing").

**Portfolio correlation risk** — flags at `DEFAULT_CORRELATION_APPROACH_FRACTION` (80%, a judgment call) of the
exact MASTER_CONTEXT 6% correlated-direction cap (confirmed via direct grep: constraint 6 and Layer 10 text both
give this number), matching the spec's own wording ("flags when combined directional exposure APPROACHES the cap",
not merely exceeds it). Found and fixed a real (if minor) floating-point boundary bug while writing the boundary
test: `6.0 * 0.8` evaluates to `4.800000000000001` in IEEE 754, not the literal `4.8`, so a caller whose own
exposure sum landed on exactly the conceptual 80% threshold could fall on either side of the comparison purely from
float representation. Fixed with a small epsilon tolerance on the comparison — the kind of boundary-precision issue
this project's testing discipline (writing exact-boundary tests, not just interior-value tests) has caught before.

**Fat-tail multiplier — the Done-when bar.** Exact spec range (1.5x–2.0x); the state-to-endpoint mapping is the
judgment call: Spike (Layer 6's own most severe "tail-risk event" state) gets 2.0x, Extended gets 1.5x,
Compressed/Expanding get 1.0x (no adjustment). Applied to all four P(loss > T) probabilities before the risk score
is derived from them — one source of truth, not two separate multiplier applications. Verified directly, not just
asserted: Compressed and Expanding produce byte-for-byte IDENTICAL output; Extended and Spike probabilities are
each exactly `min(1.0, baseline * multiplier)`; a live sanity check (printed, not just asserted) showed
`P(loss>1%)` correctly clipping at 1.000 once `0.717 * 1.5 = 1.076` would otherwise exceed the valid probability
range.

**Risk score (0–100)** — another documented judgment call: an explicit, auditable weighted sum (P(loss>2%) up to 60
points, liquidity trap up to 15, gap risk up to 15, correlation exposure's fraction of the 6% cap up to 10, summing
to a 100-point maximum). `RISK_SCORE_GATE_MAX = 65.0` (exact MASTER_CONTEXT constraint 8 number) is echoed as
`passes_risk_gate` for context — this module computes the score, it does not itself enforce the six-gate check
(Layer 11's job, not built).

### Testing

Added `tests/test_risk_engine.py` (44 tests). Direct Done-when-bar verification (`test_compressed_and_expanding_produce_identical_probabilities`,
`test_extended_multiplies_probabilities_by_exactly_1_5x`, `test_spike_multiplies_probabilities_by_exactly_2_0x`,
`test_spike_probabilities_strictly_exceed_extended_which_strictly_exceed_baseline`), the monotonicity-guard
regression test above, gap-risk session-boundary tests (exact overnight-hour boundaries, weekend detection,
tz-naive-treated-as-UTC), per-component risk-score weight isolation (each of the four terms tested independently at
its max contribution), the exact 6% cap and its float-precision boundary, and `RiskAssessment`'s own internal
consistency validation (non-monotonic probabilities rejected, tail-flag/multiplier mismatch rejected). Full suite:
**271/271 pass** (`python -m pytest tests/ -q --ignore=tests/test_binance_real_data.py`).

### Next Steps

- The Risk Engine is not wired into `PredictiveCoreStrategy`/`WalkForwardBacktester` or `explainability.py` — no
  caller in this codebase currently constructs `RiskEngineInputs` from a real backtest row or feeds a
  `RiskAssessment` into `TradeRiskSummary`/Layer 11's (not yet built) EV filter.
  `RiskEngineInputs.stop_distance_pct`/`.position_size_pct`/`.existing_correlated_direction_exposure_pct` all
  depend on Layer 10/11 outputs that don't exist yet, same situation as `explainability.py`'s `TradeRiskSummary`.
- The gap-risk session-rate defaults (`DEFAULT_GAP_RISK_BASE` etc.) are explicitly NOT calibrated against real
  historical crypto gap-frequency data — a real historical-gap-analysis pipeline would be a genuine future addition
  if this module is relied on beyond a structural placeholder.
- Everything else carried over from prior entries is unchanged.