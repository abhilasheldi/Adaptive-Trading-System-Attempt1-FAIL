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
  above chance. The median feature scored |AUC-0.5|=0.0024 — indistinguishable from noise. Full ranked table saved
  to `reports/feature_signal_univariate_auc.csv`.
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
`feature_calculator.py` already computes, requiring no new indicator logic:

- "post-compression" = `vol_state_compressed` was true at any point in the trailing 5 candles (inclusive of the
  current row) — a blunt proxy; a real Layer 5 would define this far more rigorously (contraction slope, duration,
  etc.). The lookback (5) was chosen empirically: checked 3/5/10/20-candle lookbacks first (see
  `reports/` investigation output) and picked the smallest that stayed meaningfully selective rather than capturing
  most of the dataset (compression alone is true 35.5% of all candles and is autocorrelated/persists in runs, so a
  naive longer lookback like 20 candles would have made "recent compression" true for 56% of ALL candles — nowhere
  near selective enough to call a "setup").
- "liquidity sweep" = `struct_sfp_bullish` or `struct_sfp_bearish` fires on the current candle — already-computed
  Swing Failure Pattern features (a wick beyond a recent swing high/low, immediately rejected back inside range),
  which is structurally exactly a stop-hunt/liquidity-sweep pattern, requiring no new logic to reuse.

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
  resembling real univariate separation. Full ranked table: `reports/feature_signal_univariate_auc_FILTERED.csv`.
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
  Stage 4 (unconditioned) and Stage 5 (single-asset filtered) checks. No feature shows real separation. Full table:
  `reports/feature_signal_univariate_auc_POOLED_FILTERED.csv`.
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