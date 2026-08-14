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

## Session Update: Critical Review of `macro_mtf_liquidity_vol/` — 15 Findings, 13 Fixed (2026-08-14)

Requested as a deliberately adversarial senior-engineer review of all four Phase 2 modules in
`macro_mtf_liquidity_vol/`, hunting for minor bugs, code inconsistencies, and logic flaws. **Every finding below was
empirically reproduced before being reported** (not inferred from code reading), and every fix has a dedicated
regression test. Baseline: 93 module tests / 271 suite tests passing, **none of which caught any of this**. After:
**312 suite tests pass** (+41 new). Zero new `ruff` errors (27 pre-existing import-ordering issues before and after,
left alone as out of scope).

### The two patterns behind almost every finding

Worth recording, because they predict where to look next rather than just listing symptoms:

1. **Every module validated input TYPE and RANGE but not VALIDITY, and every ambiguous case resolved toward
permitting trading.** Findings 1, 2, 5 all failed OPEN — missing or nonsensical data produced a confident,
tradeable classification instead of a refusal. That directly inverts MASTER_CONTEXT's stated priority order
(capital preservation first).
2. **Three separate findings were cases where a documented, tested bugfix addressed one instance of a bug class and
left its mirror image in place.** `liquidity_trap.py`'s `min()` to `max()` fix swapped which false positive you get
rather than eliminating it; `mtf_alignment.py`'s tie-confidence fix was applied to 1 of 3 branches. A fix that
moves a bug rather than removing it is the recurring failure mode in this package.

### Critical

1. **`volatility_classifier.py` — Spike detection missed in-window doubling (kill-switch false negative).**
`spike_ratio` was `recent_atr[-1] / recent_atr[-6]`: the function required 6 values but read only the 2 endpoints.
MASTER_CONTEXT says "ATR doubles **within** 5 candles." Demonstrated: `[10, 10, 40, 20, 12, 11]` — ATR
**quadrupled** mid-window — returned `spike_ratio=1.10` and classified **"Expanding"** ("standard position
sizing"). Spike is simultaneously a Layer 17 kill-switch trigger, one of constraint 8's six entry gates, and the
Risk Engine's 2.0x fat-tail multiplier, so one false negative disables all three at once. *Fixed:* `spike_ratio` is
now the maximum **run-up** ratio anywhere in the window (`max over j of ATR[j] / min(ATR[i] for i <= j)`).
Deliberately order-aware, not a naive `max/min`: a window that starts high and falls has `max/min = 4.0` but
contains no doubling at all — a pure decline is the opposite of a spike, and is tested for explicitly.
**NOT changed, flagged for a separate scoped task:** `feature_calculator.py:224`'s `vol_spike` uses the same old
endpoint-only formula. That flag is a *model input feature*, not the gating path, so the safety-critical path is the
one fixed; changing `vol_spike` silently alters trained-model inputs and needs its own change with its own review.

2. **NaN inputs silently produced tradeable classifications in two modules.** Root cause: `_clip01`'s
`max(0.0, min(1.0, x))` maps NaN to **1.0 — the maximum** — because every NaN comparison is False.
`MacroSignalSnapshot` caught NaN only on `btc_qqq_corr_20d` (incidentally, via its range check); a NaN `vix_level`
passed `nan <= 0`, all hard gates then failed, and the classifier returned **"Transitional"** — "halve position
sizes" — i.e. *the system kept trading on entirely missing macro data*, with both internal margin scores corrupted
to ~max. Same class in `volatility_classifier.py`: all-NaN ATR gave **"Expanding"**, since `nan >= 2.0` is False and
the result dataclass's `spike_ratio < 0` check is False for NaN too. **Not theoretical** — this file already
records yfinance returning six entirely-NaN columns from this exact pipeline. *Fixed:* finite-value validation at
every boundary (all 7 `MacroSignalSnapshot` numeric fields, the full ATR window, `VolatilityStateResult`'s
`spike_ratio`, and liquidity_trap's current-candle + window OHLC), plus `_clip01` now raises on non-finite input as
defence in depth. Note `liquidity_trap.py` already did this correctly (`np.isfinite`) — this was a cross-module
inconsistency, not an unknown pattern.

### High

3. **`liquidity_trap.py` — an ordinary outside bar fired BOTH sweep directions; the code's stated justification was
false.** The comment claimed both directions "can't be confirmed on the same candle in practice (would require
closing both above AND below distinct levels)." Inverted reasoning: it requires closing *below a resistance level
and above a support level*, which any wide-range candle closing mid-range does. Reproduced with resistance ~105 /
support ~95 and a `high=106, low=94, close=100` bar: both branches fired and `sweep_direction` silently became
`"bullish"` purely because that branch runs second. `ensemble.py:370` consumes `sweep_direction` as a directional
signal scaled by `trap_probability`, so a perfectly symmetric, directionally meaningless bar was injecting a hard
bullish lean into the meta-model. *Fixed:* a candle sweeping both sides now reports **no confirmed sweep** (the
ensemble treats that as neutral) with the conflict recorded in `reasons` — the capital-preservation-consistent
resolution.

4. **`liquidity_trap.py` — genuine breakouts labelled as confirmed reversal sweeps.** Confirmation was tested
against the **farthest** swept level (`max`), the most permissive possible rule. With clusters at 102 and 110, a
`high=111, close=108` candle was reported as a confirmed **bearish** sweep despite closing decisively *above* the
102 cluster — a bullish breakout labelled as its opposite. *Fixed:* extent (farthest level) is still what gets
**reported**, but confirmation is now tested against the **nearest** swept level — price must be rejected back past
*all* the liquidity it ran. The earlier `min()` bug this `max()` was introduced to fix is now properly handled by
the `level > reference_price` position filters added later, so the nearest-level test is safe. A dedicated test
guards against over-correcting into never detecting a sweep.

5. **`mtf_alignment.py` — `neutral_band` was entirely unvalidated.** It governs how every reading is interpreted.
With `neutral_band=-0.5`, four readings of **exactly 0.0** — no directional information whatsoever — classified as
a unanimous **4/4 bullish** read with `trade_permitted=True` ("Full position size permitted"), clearing constraint
8's MTF gate, because `0.0 > -0.5`. The mirror failure was silent too: `neutral_band >= 1.0` collapses every
reading to neutral forever. *Fixed:* validated to a finite `[0, 1)`; 0.0 kept legal as a legitimate aggressive
choice.

### Medium

6. **`macro_regime.py` — the Risk-Off VIX-ROC leg contradicted the normalization rule its own siblings follow.**
The inline comment states the principle explicitly (normalize relative to the threshold, "since that threshold —
not zero — is where Risk-Off's condition actually starts being satisfied"). The correlation leg and the VIX-*level*
leg both correctly read 0 at their thresholds; the VIX-*ROC* leg read **1.0 — the maximum** — at its threshold, in
the same expression. A snapshot at the bare minimum of all three Risk-Off conditions reported `confidence=0.168`
driven by a leg that should contribute nothing. *Fixed:* now `(roc - threshold) / threshold`, mirroring the VIX-level
leg's shape. Bare-minimum trigger now reports **0.001**.

7. **`macro_regime.py` — an OR condition was scored by averaging.** `vix_elevated_or_spiking` is an OR, but its two
legs were averaged, halving a decisively-triggered condition: VIX at 40 with flat ROC scored `(0.6+0)/2 = 0.3`.
*Fixed:* `max()`, matching OR semantics — the satisfied disjunct governs. (Risk-On's `vix_calm` genuinely is an AND
and correctly still averages.) That case now reports 0.811.

8. **`mtf_alignment.py` — the 2026-08-13 tie-confidence fix was applied to only 1 of 3 branches.** The
bullish/bearish branches still averaged over the winning side alone, discarding dissent magnitude entirely:
`{+0.2,+0.2,+0.2,-1.0}` and `{+0.2,+0.2,+0.2,-0.16}` both reported **score=3 AND confidence=0.20** — a violent
opposing 1W read indistinguishable from near-consensus, both clearing the MTF gate at "Standard position size
permitted." *Fixed:* confidence for a real dominant direction is now **net conviction toward it**, averaged over
every non-neutral timeframe, clipped at 0. Unanimous readings are unchanged (no dissent to subtract), neutral
timeframes stay excluded from the denominator (absence of evidence, not evidence of weakness), and the tie branch
keeps its own symmetric rule (a tie has no dominant direction to be net toward).

9. **`liquidity_trap.py` — docstring and code disagreed about swept clusters.** The comment promises "risk that an
**UN-SWEPT** nearby cluster gets hunted next," but `nearest_*` was computed over *all* clusters, so a pool that had
just been consumed was still scored as pending forward risk (0.750 in the reproduction). *Fixed:* swept clusters
are excluded; `nearest_*_cluster` now means "nearest cluster still standing," consistent with what
`trap_probability` claims to measure. (Swept clusters remain fully visible via `reasons` and the `sweep_*` fields.)

### Low / consistency

10. **`liquidity_trap.py` cluster `level` was the mean of ALL proximity members**, including wicks the `respected`
filter had just disqualified as non-touches — highs `[100.00, 100.05, 100.10]` with only the first two genuine
reported `level=100.050` instead of `100.025`. That level also feeds `distance_atr` and the sweep-breach
comparison, so the error propagated. *Fixed:* the all-member mean still *defines* the shared level for the respect
test (it must — that test is what identifies the touches), but the reported level is now the mean of genuine
touches only.
11. **`LiquidityTrapResult` validated its sweep invariant in one direction only** — `sweep_just_confirmed=True` with
`sweep_direction=None` ("a sweep confirmed, but in no direction") constructed silently. *Fixed:* both halves
enforced, matching `MacroRegimeResult`'s `is_macro_event_window` treatment.
12. **`LiquidityCluster` was the only result dataclass across all four modules with no `__post_init__`** —
`side="banana", touch_count=-5, distance_atr=-1.0` all constructed silently. *Fixed:* full validation, including
`touch_count >= 2` (a "cluster" is by definition "two or more wicks").
13. **Four unvalidated tunables in `estimate_liquidity_trap`** — `lookback=0` gave an opaque `IndexError`;
`proximity_scale_atr`/`density_saturation_touches` are divisors and raised `ZeroDivisionError` deep inside the risk
calculation. *Fixed:* validated with clear messages.
14. **Signature inconsistency** — the other three modules use keyword-only tunables (`*`); `estimate_liquidity_trap`
took five positional-or-keyword params. *Fixed:* now keyword-only, matching its siblings. Safe: no production
callers exist (only tests).
15. **NOT fixed, deliberately:** all four modules expose `List`/`Dict` fields on `frozen=True` dataclasses, so
`result.reasons.append(...)` mutates a "frozen" object. Converting to tuples ripples through all four modules and
every test constructing these types — a real consistency nit, but it is not a live bug and belongs in its own
scoped change.

### Next Steps

1. **The primary blocker is cleared for the planned backtest run.** Findings 2 and 5 in particular were blocking:
feeding real macro data (which this pipeline has already been observed to return as all-NaN) through
`classify_macro_regime` would have produced plausible-looking `Transitional` reads from garbage, and every one of
the five Layer 14 criteria evaluated in that run would have inherited it.
2. **`feature_calculator.py`'s `vol_spike` still needs the finding-1 fix** in its own scoped change (see above for
why it was deliberately not bundled here).
3. Wiring the seven Phase 2 modules into `WalkForwardBacktester`/`PredictiveCoreStrategy` remains the actual next
build step — unchanged by this review, but now on a materially sounder base. Per the user's 2026-08-14 decision,
**Phase 1E's "clean pass" bar now requires all five Layer 14 criteria to genuinely evaluate** (no longer just the 3
that were computable before Layer 3/7C existed).
4. Everything else carried over from prior entries is unchanged.


## Session Update: `vol_spike` Corrected in `feature_calculator.py` (2026-08-14)

Follow-up to the same day's `macro_mtf_liquidity_vol/` review, which fixed the Layer 6 classifier's spike detection
but deliberately deferred the identical bug in `features.feature_calculator`'s `vol_spike` feature. Done here as its
own scoped change, because `vol_spike` is a **model input feature** (one of the 39 columns fed to the k-NN analog
engine's distance metric and to the LightGBM/XGBoost trainers), so changing it changes model inputs.

### What it actually influences — checked before changing anything

- `vol_spike` is **not consumed by name anywhere else in `src/`**. No gate, kill switch, or risk calculation reads
  it. Verified by direct grep. The gating/kill-switch path uses `macro_mtf_liquidity_vol.volatility_classifier`,
  which computes its own spike independently and was already fixed earlier the same day.
- It is registered via `_FeatureSet.add_excluded`, so it is emitted as a **binary column, not percentile-ranked** —
  it does reach the models, it just isn't re-scaled.
- Its only downstream effect is therefore as one of 39 model input features.

### The change

`atr / atr.shift(5)` (endpoint-only) replaced with `_max_atr_runup_ratio`: the largest factor by which ATR rose
from any earlier point in the trailing 6-candle window to any later one, `max over j of (atr[j] / min(atr[i] for
i <= j))`. Implemented as vectorised shifted-series arithmetic with a running minimum — 66x faster than the
equivalent `rolling(6).apply()`, and verified equal to it to 3e-10 with zero flag mismatches on the full 9-year
real BTC history.

### Two real bugs found while doing it, neither the one being fixed

1. **`+ EPS` in the denominator perturbed every legitimate ratio downward**, so an exact 2.0x doubling evaluated to
1.999999998 and failed the `>= 2.0` test. That also made `vol_spike` disagree with the Layer 6 classifier (which
divides exactly, guarding with `if running_min > 0`) — i.e. the two implementations of one MASTER_CONTEXT rule
disagreed at the exact boundary. *Found by the cross-module equivalence test, not by reading.* Fixed by
SUBSTITUTING EPS for non-positive divisors rather than ADDING it to all of them.
2. **`.where(running_min > 0, EPS)` silently converted warm-up NaNs into EPS** (because `NaN > 0` is also False),
emitting a spurious ~2e9 ratio and flagging a **false spike on every series' early rows**. This one was introduced
by fix 1 and caught immediately by the warm-up-NaN test. Fixed with `.mask(running_min <= 0, EPS)`, whose condition
is correctly False for NaN.

Also corrected an inaccurate docstring on `volatility_classifier._max_runup_ratio` written earlier the same day: it
claimed the ratio returns "< 1.0 for a monotonically falling" window, which the code never did (it floors at 1.0).
Same doc/code mismatch class as review finding #9 — found by auditing my own prior work.

### Measured effect on real data — deliberately quantified rather than assumed

Regenerated the full 39-column feature frame before and after, on both cached real datasets, and diffed
column-by-column:

- **BTC (78,381 rows): exactly one column changed — `vol_spike`, on 17 rows.** The other 38 features are
  bit-identical.
- **ETH (35,076 rows): exactly one column changed — `vol_spike`, on 4 rows.**
- Every change is 0 to 1 — a **strict superset**, no detection was lost in either dataset.
- Zero spikes in the warm-up region (confirming bug 2 above is genuinely gone).

**All 21 newly-flagged candles audited individually, and all 21 are genuine**: true in-window run-up 2.00x–2.10x
against endpoint ratios of 1.87x–1.99x, i.e. precisely the boundary cases the endpoint formula could not see. Zero
false positives. They are real volatility events, not numerical artifacts — median hourly range 1.38% vs. 0.69%
across all candles — and include recognisable events (the 2021-12-04 BTC flash crash, 2022-03-27, 2025-03-02).

### Honest assessment of whether this is a "positive effect"

**Correct and safe, but its effect on model performance will be undetectable — and that is the expected outcome,
not a disappointment.** `vol_spike` fires on ~0.2% of candles either way (137 to 154 on BTC). A binary flag that
rare is near-constant for a gradient booster and contributes almost nothing to a k-NN distance metric; it was never
going to be a meaningful predictor before OR after. It correctly never appeared in the Stage 4 feature-importance
work.

One honest caveat on the severity framing from the review entry above: the synthetic demonstration case
(`[10, 10, 40, 20, 12, 11]`, ATR quadrupling mid-window) is **more extreme than real ATR behaves**. ATR is itself a
14-period average, so it does not whipsaw that fast — in nine years of real BTC data every missed case was marginal
rather than dramatic. The bug was real and the fix is right, but the real-world manifestation is milder than the
synthetic example implies.

The genuine value of this change is therefore: (1) one MASTER_CONTEXT rule now has **one** definition instead of
two divergent ones — the exact drift class that caused this bug; (2) two additional latent bugs found and fixed;
(3) `tests/test_features.py` now exists at all.

### `tests/test_features.py` created — a real gap, not just a home for these tests

`features.feature_calculator` — the 39-feature module everything downstream depends on — **had no dedicated test
file**. It was exercised only indirectly through `test_no_lookahead.py`'s shared helpers. The new file covers the
in-window-vs-endpoint regression, order-awareness (a pure ATR decline has max/min == 4.0 but must never flag), the
1.0 floor, window-width/aging, warm-up NaN handling, the binary-not-percentile-ranked contract, a dedicated
no-lookahead test (mutate a future bar, assert every earlier row is bit-identical), and — most importantly — a
**cross-module equivalence test** asserting `feature_calculator._max_atr_runup_ratio` and
`volatility_classifier._max_runup_ratio` agree row-for-row. That test is what makes the two implementations unable
to silently drift apart again, and it is what caught bug 1 above.

Full suite: **322 passing** (was 312 after the module review, 271 before the day's work). Zero new `ruff` errors
(8 pre-existing in `feature_calculator.py` before and after; the new test file is clean).

### Open observation, not acted on

At ~0.2% positive rate, `vol_spike` is arguably not earning its place as a model input feature at all — it adds a
near-constant dimension to the k-NN distance metric. Whether to keep it as a feature (as opposed to keeping the
Layer 6 categorical state, which is what actually gates trades) is a **design** question worth a deliberate
decision, not a bug. Flagged rather than resolved unilaterally.


## Session Update: Explainability Engine Rebuilt as Pure Template/Rule-Based — LLM Path Removed (2026-08-14)

`src/predictive_core/explainability.py` (Layer 8) rewritten to contain **no LLM call of any kind, in any tier, on
any path** — not in normal operation, not in a fallback, not behind a feature flag. No network I/O, no API key, no
billing dependency. Replaces the prior version, which called the Anthropic API (Haiku-tier) to phrase each
contribution line with a template path as fallback.

### What was removed, not merely disabled

`AnthropicProseGenerator`, `_build_prose_prompt`, `_extract_json_object`, `ProseGenerationError`, the
`enable_live_api` and `prose_generator` constructor parameters, the `used_live_api` output field, the `json` import,
and `DEFAULT_PROSE_MODEL`/`DEFAULT_PROSE_MAX_TOKENS`/`DEFAULT_PROSE_TIMEOUT_SECONDS`. Also removed `anthropic` from
`pyproject.toml` — it was added solely for this module (see this file's Layer 8 build entry) and had zero remaining
consumers, so leaving it declared would have kept a billing dependency in the project for nothing.

Why this is better here, not just cheaper:
- **Determinism.** A trading system's audit log must be reproducible: the same attribution input must render
  identically forever, so a post-mortem on a losing trade reads exactly what was shown at decision time. A sampled
  language model cannot guarantee that.
- **No numeric risk surface.** The old design kept numbers away from the model as defence in depth. Removing the
  model removes the boundary's reason to exist entirely.
- **No runtime dependency.** Explanation generation cannot fail, time out, rate-limit, or cost money at the moment
  a trade is being evaluated.

### What replaces it

Prose is produced by mapping structured attribution facts onto pre-written sentence structures. Numbers are always
inserted directly from the attribution data via format specifiers — never generated, paraphrased, or re-derived.

- **Contribution lines**: `Display label (concrete detail)` — the exact shape the Adaptive doc's example uses
  ("Macro regime (Risk-On, confidence 0.81)").
- **A new rule-selected `Assessment:` block**, appended after the spec sections. Each sentence is chosen by an
  explicit branch: which factor dominates (and whether it's a tie, or negligible); whether the top contributor is
  positive or negative; which macro regime is active (all four states mapped, per Layer 3's exact table); EV
  pass/fail; weight-stability state; data-quality degradation.
- `spec_report_text` (the Adaptive doc's format, section-for-section) and `report_text` (spec + Assessment) are now
  **separate output fields**, so the spec-format guarantee is independently assertable rather than entangled with
  the added narrative.

### Edge cases handled explicitly — the main practical argument for a rule-based renderer

A generative model would produce fluent text for every one of these without signalling that anything was wrong:

- **Attribution summing to near-zero**: below `NEGLIGIBLE_NET_LEAN_PCT` net lean, the narrative says the signals
  substantially offset one another instead of naming a "dominant" driver of an undecided call.
- **Ties between top contributors**: detected within `TIE_TOLERANCE_PCT` and reported *as* a tie ("carry
  effectively equal weight… no single factor dominates"), never resolved by arbitrarily naming whichever sorted
  first. Sorting was additionally made **total** (label as final tiebreak) so ordering cannot shift with upstream
  dict order — a real determinism hole in the previous sort.
- **Missing/degraded data**: new `SignalContribution.is_degraded` flag drives a dedicated "Data quality" line, so a
  well-founded contribution is distinguishable from one standing on a gap. An invalid Component A also renders the
  CI as "unavailable" rather than fabricating an interval.
- **Weight instability flagged mid-explanation**: surfaced as an explicit **leading** caveat, not a footnote —
  when weights are unstable the attribution percentages are themselves the quantity under suspicion, so a reader
  must not take the breakdown at face value.

Also added: `net_lean_pct()`, which reads conviction from `EnsembleForecast`'s own probabilities rather than by
summing the rendered lines. Those are display-filtered, so summing them would understate conviction whenever small
contributions were dropped — a subtle trap the narrative would otherwise have fallen into.

### Done-when bars, all four verified

1. **No external call.** Grepping the module for `api`/`anthropic`/`claude`/`requests`/`httpx`/`urllib` returns
   five hits, **all in docstring prose explaining the removal** — zero in executable code. Verified by three tests
   that strip comments/docstrings before asserting, restrict the import set to
   `{math, dataclasses, typing, numpy, src}`, and reload the module asserting `anthropic`/`httpx`/`requests` are
   absent from `sys.modules`. The full 13-module dependency chain was grepped directly: **no project module in it
   imports a network package.** (`socket`/`urllib` do appear in `sys.modules` — confirmed by direct test to come
   from numpy/pandas themselves, as in any scientific Python stack, not from project code.)
2. **Format matches the spec section-for-section**, asserted on section *order* as well as presence, plus a regex
   on the contribution-line shape and a check that the percentage column is right-aligned.
3. **Determinism**: 25 identical renders compared byte-for-byte, plus a variant test running all seven edge-case
   configurations 8x each, a reversed-input-order test proving tie ordering is stable, and a shared-engine test
   proving no state accumulates between calls.
4. **Every edge case renders readably**: a dedicated test asserts no blank field, no whitespace-only line, and a
   non-empty narrative across six degenerate configurations.

### Testing

`tests/test_explainability.py` rewritten: the seven LLM-path tests (injected fake generators, fallback-on-failure,
numeric-smuggling defence) are gone with the code they covered; everything non-LLM was kept (provable sum identity,
Wilson CI, exact +0.15% gate with strict `>`); 25 new tests added for the four Done-when bars. **52 tests** in this
file, full suite **345 passing** (was 322).

Lint: 35 `ruff` errors vs. 31 before, all in the same three pre-existing categories (`UP006`/`UP045`/`UP035` —
pyupgrade preferring `list[str]`/`str | None` over `typing.List`/`Optional`). The four new ones come from writing
new annotations in the same `typing.List`/`Optional` style every sibling module in this codebase uses; fixing them
in this file alone would make it inconsistent with its siblings, so this is left as a codebase-wide style decision
rather than a local one.

### Open item

`macro_regime_state` is passed to `explain()` as an explicit optional argument rather than parsed back out of the
macro contribution's `detail` string — reconstructing structured state from rendered text would be exactly the
fragile coupling this project avoids. When Layer 8 is eventually wired into a real caller alongside Layers 9/10,
that caller should pass it through from `MacroRegimeResult.state` directly. Omitting it simply skips the macro
sentence rather than guessing.


## Session Update: Phase 2 Partial Integration + 5-Fold Layer Ablation (2026-08-14)

First walk-forward run with Phase 2 conditioning layers actually wired into the backtester, structured as a
leave-one-out ABLATION so each layer's marginal effect is measured rather than assumed. **Layer 14 research-phase
criteria only — the Tier 1 live gate is not evaluated anywhere in this run or this codebase.**

### What got wired in (and what could not)

| Layer | Status | Reason |
|---|---|---|
| 5 Liquidity Trap | WIRED (ensemble input) | Operates on raw OHLCV, which the modeling frame carries. |
| 6 Volatility Regime | WIRED (gate, not ensemble input) | Its spec role is entry gate #4, not one of Layer 7C's five inputs. |
| 7C Ensemble | WIRED | Replaces the Phase 1 hard agreement rule with the real linear blend. |
| 3 Macro Regime | NOT WIRED | No macro data ingested (`data/raw/` is BTC/ETH OHLCV only); no economic calendar. |
| 4 MTF Alignment | NOT WIRED | Needs 1H/4H/1D/1W; `config.yaml` has `["1h","4h"]`. Module raises rather than accept fewer. |
| 9 Risk Engine | NOT WIRED | Needs `stop_distance_pct`/`position_size_pct` from Layers 10/11, which don't exist. |

Layers 3/4 are passed to the ensemble as `None` (excluded + renormalized, the same path Component A uses when its
N>=50 gate withholds a forecast), NOT as fabricated "neutral" results — a neutral distribution actively damps the
blend rather than stepping out of the way.

### Results — BTC/USDT 1h, 2022-08 to 2026-08, 5 folds, spec Optuna budget

| variant | OOS Sharpe | PF | maxDD | trades |
|---|---|---|---|---|
| baseline_phase1 | -12.974 | 0.236 | 0.056 | 3502 |
| all_layers | -8.284 | 0.014 | **1.023** | 2817 |
| no_liquidity_trap_L5 | -8.284 | 0.014 | 1.023 | 2817 |
| no_volatility_gate_L6 | -4.846 | 0.008 | **1.235** | 2813 |
| no_ensemble_L7C | -12.359 | 0.238 | 0.056 | 3499 |

`baseline_phase1` reproduces the documented Phase 1 result (prior entry: Sharpe -12.90, PF 0.231) to within the
expected delta from this session's `vol_spike` correction — confirming at real-data scale that `Phase2Strategy`
with all flags off is behaviourally identical to `PredictiveCoreStrategy`, which is what makes every ablation
delta attributable to the layer that differs.

### The Sharpe-only verdict is MISLEADING, and inverts on the other criteria

The runner's automatic flag (Sharpe-only, as instructed) reports:
- Layer 5: +0.000 (no effect) · Layer 6: **-3.438 (FLAGGED)** · Layer 7C: +4.075 (helps)

That verdict is wrong about which layer is harmful, and the reason is recorded here so it isn't repeated:

**`maxDD > 1.0` means the equity curve went NEGATIVE** (`drawdown = (equity - running_max)/running_max`, so
1.023 puts equity at `-0.023 x peak`). Every variant with the ensemble enabled destroyed the account. **Sharpe
computed over a blown-up equity curve is numerically meaningless**, which is why those Sharpe values swing wildly
(fold 2: -9.78 vs -30.7) on a ~1-trade difference. The +4.690 "improvement" of the full stack over baseline is an
artifact of that instability, not a real gain.

### Per-layer verdicts, each verified rather than inferred

**Layer 7C (Ensemble) — the genuinely destructive layer, and the Sharpe flag misses it.**
It is the only difference between `no_ensemble_L7C` (PF 0.238, maxDD 0.056 — essentially baseline) and
`all_layers` (PF 0.014, maxDD 1.023 — account destroyed). Mechanism, verified by inspecting fitted weights on
real data: the ensemble assigns **component_b = 0.942**, i.e. it is ~94% Component B alone. That removes the
Phase 1 hard agreement rule, which required Component A and B to agree before trading and was acting as a
*filter*. Replacing a filter with a near-pure Component B read, on a classifier with near-zero measured skill,
produces many more directional bets and wipes out the account.

**Layer 5 (Liquidity Trap) — contributes exactly nothing, as configured.**
`all_layers` and `no_liquidity_trap_L5` are byte-identical (same Sharpe to 3dp, same 2817 trades). Verified the
cause rather than assuming a bug: the fitted ensemble weight on `liquidity_trap` is **exactly 0.000000**, so
blending it at zero weight is arithmetically identical to excluding it and renormalizing. The optimizer
independently learned to ignore it. Consistent with a direct measurement on real data: Layer 5 reports
`sweep_just_confirmed` on **44.1% of bars** (197 bearish / 182 bullish — near-balanced), which is noise, not a
rare stop-hunt event. For comparison, the earlier compression+SFP proxy selected 2.04% of candles. Layer 5's
default `level_tolerance_atr=0.25` over a 50-candle lookback is far too permissive for 1h BTC.

**Layer 6 (Volatility Gate) — UNTESTED, not "degrading", despite being the flagged layer.**
Measured over the exact backtest window: the gate blocks **61 of 35,069 bars (0.174%)**. Per-fold trade counts
differ by ~1 between gated and ungated runs. A layer touching 0.17% of bars cannot genuinely move Sharpe by
3.438; that delta is numerical noise on the destroyed equity curve described above. The gate's forced-flat bars
also *split* continuous positions, which is why the gated run shows slightly MORE trades (2817 vs 2813) — the
initially-backwards direction, explained.

### Sample-size collapse — a second reason the Sharpe numbers can't be read at face value

Per-fold OOS trade counts, `all_layers`: **[1215, 1076, 468, 46, 12]** vs baseline **[1273, 1168, 616, 226, 219]**.
Folds 4 and 5 fall to 46 and 12 trades. MASTER_CONTEXT's hypothesis requires a minimum 200-trade out-of-sample
sample; a 12-trade fold produces a meaningless Sharpe that nonetheless enters the aggregate mean. Trade counts
are reported alongside Sharpe in the ablation output specifically to keep this visible.

### Layer 14 status — 4 of 5 criteria now evaluate (was 3)

`layer14_ensemble_weight_stability` now reports **pass** (previously `not_evaluated`) for every ensemble variant —
Layer 7C being wired in is what made it computable. `layer14_risk_on_regime_vs_aggregate` remains
`not_evaluated` because it requires Layer 3, which cannot be wired without macro data.

Every variant **fails** `layer14_oos_sharpe` and `layer14_oos_profit_factor`. The three ensemble variants also
**fail** `layer14_oos_max_drawdown` (>25%); baseline and `no_ensemble_L7C` pass it.

Per the user's 2026-08-14 decision that Phase 1E's "clean pass" requires all five criteria to genuinely evaluate,
**1E still cannot be assessed** — one criterion remains structurally uncomputable until macro ingestion exists.

### Code changes required, each with tests

- `dataset.build_asset_modeling_frame(keep_raw_atr=True)` — emits `RAW_ATR_COLUMN`, needed because Layer 5
  measures distances in ATR units and the percentile-ranked `vol_atr_14` is in [0,1].
- `EnsembleSignalSet.macro_regime/.mtf_alignment/.liquidity_trap` are now `Optional`; `None` means "layer not
  wired up". The raw-Layer-2-feature type guard is fully intact (None is the only permitted non-instance value).
- **Bug found by tests**: `EnsembleMetaModel.predict()` read `signals.macro_regime.macro_weight` unconditionally
  and raised `AttributeError` on the absent path. Now reports 0.0 (no macro conditioning applied).
- **Bug found by tests**: `fit_ensemble_weights` RAISED on a degenerate fold (only one signal present -> flat
  objective -> SLSQP "Inequality constraints incompatible"), which would kill an entire walk-forward run. Now
  falls back to equal weights — correct when every weight vector is equally optimal — with `converged=False` and
  `optimizer_message` recorded rather than applied silently.
- New `backtesting/phase2_strategy.py` (per-layer flags; all-off proves baseline equivalence) and
  `scripts/run_layer_ablation.py` (leave-one-out design, shared model+prediction cache).

Full suite: **384 passing**.

### Next Steps

1. **Do not keep Layer 7C in this form.** It is the destructive layer. The Phase 1 agreement rule was acting as a
   filter; the ensemble removed it. Either restore an agreement/confidence gate on top of the blend, or do not
   enable Layer 7C until Component B has measurable skill (it currently has ~none — see the Stage 1-4 entries).
2. **Re-tune Layer 5 before drawing any conclusion about it.** A 44% confirmed-sweep rate is not a stop-hunt
   detector. Tighten `level_tolerance_atr` / require compression, then re-measure selectivity BEFORE re-running
   the ablation — the current run tested a mis-tuned heuristic, not the hypothesis.
3. **Layer 6 needs a dataset where it fires** to be tested at all. At 0.174% of bars it is unfalsifiable here.
4. **Add a composite ablation criterion.** Sharpe-only flagging demonstrably picked the wrong layer in this run.
   Any future ablation should flag on max drawdown and profit factor too, and refuse to report a Sharpe computed
   over an equity curve that went non-positive.
5. Macro ingestion (Layer 3) and 1D/1W ingestion (Layer 4) remain the blockers to both a full-stack test and to
   Phase 1E being assessable under the all-five-criteria bar.


## Session Update: Negative-Equity Root Cause Found + Under-Powered Fold Exclusion (2026-08-14)

Follow-up to the layer ablation, working the user's prioritised list. **Two items completed** (the safety audit and
the validation-protocol fix); **three items deferred** and recorded below with everything needed to pick them up.

### COMPLETED 1 — Audit: why the hard risk caps did not prevent negative equity

The ablation reported max drawdowns of 1.023 and 1.235. Since `drawdown = (equity - running_max)/running_max`, a
value above 1.0 means equity went **below zero**. MASTER_CONTEXT constraint 6 says the 2%/6%/10% caps are enforced
"at the execution layer independent of any model output", so negative equity should be structurally unreachable.
The audit found three distinct defects, verified by measurement rather than reasoning.

**(a) There are NO execution-layer risk caps. At all.**
Verified by direct grep of `src/backtesting/` and `src/risk_sizing_selection/`: the numbers 2%/6%/10% appear only
inside `risk_engine.py`'s risk-SCORING logic (as inputs to a 0-100 score) and in a package `__init__.py` comment.
Nothing in the execution path enforces anything. The only limiter that exists is
`Phase2Strategy.max_position = 0.02`, which is defective as a cap in three separate ways:
  - it is a **model-layer** parameter, exactly what the spec says must NOT be the thing enforcing caps;
  - it caps position **SIZE**, not **RISK** — MASTER_CONTEXT's 2% is capital-at-risk (size x stop distance), and
    this backtest has no stops at all, so the two quantities are not comparable;
  - it is per-trade only. There is no correlated-direction accounting and no aggregate-exposure accounting, so the
    6% and 10% caps have no implementation whatsoever.

**(b) The equity curve has no bankruptcy floor.**
`metrics.equity_curve_from_trade_pnls` returns `init_cash + realized.cumsum()` — an unbounded cumulative sum. Once
cumulative losses exceed `init_cash` the curve simply goes negative; there is no ruin detection and no halt.
Reproduced directly: 3,000 trades at -$5 each on a $10,000 account crosses zero at trade 2,000 and ends at
-$5,000, reporting `max_drawdown = 1.500`.

**(c) THE DOMINANT CAUSE — sizing and accounting are two different books, and the gap-cost model is unbounded.**
This is the finding that actually explains the magnitude, and it was found by reproducing fold 2 of the
`all_layers` configuration end-to-end (`scripts/diagnose_negative_equity.py`, kept):

```
max |position| = 0.020000        <- the 2% limiter WAS respected
trades = 1151, mean notional $29
net_pnl: mean -$2.34, sum -$2,696
gap_cost total: $2,588           <- 96% of the entire loss
vectorbt (sizing book):  min $9,870  maxDD = 0.013   (1.3%)
reported (trade book) :  min $7,304  maxDD = 0.270   (27%)
```

`WalkForwardBacktester._run_segment` passes fees and slippage into `vbt.Portfolio.from_orders`, but gap costs are
applied AFTERWARDS by `cost_model.apply_gap_cost_to_trades`. So vectorbt sizes every order at 2% of an internal
equity that never sees gap costs, while the reported curve deducts them. Here that produced a **20x divergence**:
vectorbt believed the account was down 1.3% while the real book was down 27%. Position size therefore never
shrinks as the true account drains — the opposite of what percent-of-equity sizing is supposed to do.

And the gap cost itself is unbounded by construction (`costs.py`):

```python
expected_gap_cost_frac = (n_weekend_held  * gap_frequency_weekend  * gap_magnitude_weekend_frac
                        + n_overnight_held * gap_frequency_overnight * gap_magnitude_overnight_frac)
gap_cost_value = expected_gap_cost_frac * notional
```

It is charged **once per bar held, linearly, with no cap**. `n_overnight_held` grows with the holding period, so
`expected_gap_cost_frac` can exceed 1.0 and the "expected gap cost" can exceed the entire position value. The
ablation's positions were held almost continuously (99.8% of bars nonzero, only 2 direction flips in fold 2), so
holding periods were enormous and gap costs accumulated without limit. That is what turns a modest modelled loss
into an arbitrarily large one and drives equity through zero.

**Answer to "why didn't the caps prevent it": because there are no caps, the equity accounting has no floor, and
the cost model can charge more than a position is worth.** Note this is a BACKTEST-ACCOUNTING failure, not
evidence that a live system would blow up — but it does mean every drawdown and Sharpe number from ensemble-
enabled variants is unusable, and that constraint 6 is entirely unimplemented.

### COMPLETED 2 — Under-powered folds excluded from aggregate stats

`WalkForwardBacktester` now takes `min_oos_trades_per_fold` (default **200**, MASTER_CONTEXT's exact minimum
out-of-sample sample size). Folds below it are excluded from `aggregate_oos_sharpe` and reported on
`BacktestReport.under_powered_folds` / `n_folds_in_aggregate_sharpe`, with the exclusion disclosed in `summary()`.
The ablation had folds of 46 and 12 trades feeding the same equal-weighted mean as folds of 1,215.

Deliberately NOT excluded from pooled profit factor or max drawdown: pooling is sample-weighted by construction
(a 12-trade fold contributes 12 trades, not an equal-weighted statistic), and a drawdown that happened, happened.
Only the mean-of-per-fold-Sharpes is distorted by equal-weighting a noise estimate.

Existing tests were updated to pass `min_oos_trades_per_fold=0` explicitly — synthetic fixtures are far below the
production minimum by design, matching how `windows.py` already documents test-scale configuration.

**Also landed in the same edit region (partial delivery of deferred item 3 below):** `SegmentResult.equity_destroyed`
and `BacktestReport.any_equity_destroyed`. `_run_segment` now detects `(equity_curve <= 0).any()` and forces that
segment's Sharpe to `nan`, because `pct_change()` across a zero crossing is arithmetically meaningless. `summary()`
prints an explicit "EQUITY DESTROYED" warning. The ablation script's own flagging is NOT yet updated — see below.

Full suite: **384 passing** (26 in `test_backtesting.py`, including 6 new).

### DEFERRED 1 — Treat Component B's lack of skill as a Layer 14 kill-criterion hit

Not a tuning task. The ablation's fitted ensemble weights on real data were **component_b = 0.942**, and Component
B's own diagnostics across every prior run show a logloss skill score of 0.004-0.016 versus a base-rate baseline —
i.e. essentially no skill. A 94%-weighted signal with no skill means there is nothing to blend, so ensemble
tuning cannot help.

Per MASTER_CONTEXT's own Layer 14 criteria, OOS Sharpe < 0.5 means **"stop, revise hypothesis"** — every variant
in the ablation failed that, plus the profit-factor criterion. The correct next action is a hypothesis revision
(different asset, timeframe, or setup definition per Build Plan 1E), NOT another parameter sweep around a core
that has already been shown across Stages 1-4 not to contain edge in this feature set.

### DEFERRED 2 — Re-tune or respecify Layer 5's sweep-confirmation logic

Measured on real BTC 1h: Layer 5 reports `sweep_just_confirmed` on **44.1% of bars** (197 bearish / 182 bullish —
near-balanced), where the spec describes sweeps as "specific, programmatically detectable events". For scale, the
earlier compression+SFP proxy selected **2.04%** of candles — two orders of magnitude more selective.

The ensemble independently assigned Layer 5 a weight of **exactly 0.000000**, correctly identifying it as noise;
that is why `all_layers` and `no_liquidity_trap_L5` were byte-identical. Likely culprits are
`DEFAULT_LEVEL_TOLERANCE_ATR = 0.25` over a 50-candle lookback (clusters form on almost any 1h data) and the
absence of any compression precondition. Related open question already noted when Layer 5 was built: the Adaptive
doc's hypothesis section defines post-compression as "ATR at or below its 20th percentile", distinct from Layer
6's Compressed state (<30th) — that stricter definition was never wired in. Re-measure selectivity BEFORE
including Layer 5 in any further run; the ablation tested a mis-tuned heuristic, not the hypothesis.

### DEFERRED 3 — Finish disqualifying Sharpe in the ablation harness

The harness half is done (see above). Still outstanding: `scripts/run_layer_ablation.py::_format_comparison`
still ranks and flags purely on Sharpe, which in this run picked **Layer 6** (touches 61 of 35,069 bars = 0.174%,
so it cannot genuinely move Sharpe) over **Layer 7C** (the layer that actually destroyed the account: it is the
only difference between `no_ensemble_L7C` at PF 0.238 / maxDD 0.056 and `all_layers` at PF 0.014 / maxDD 1.023).
The flagging must gate on maxDD/PF sanity first and refuse to rank on Sharpe at all when
`any_equity_destroyed` is true.

### Files

- `scripts/diagnose_negative_equity.py` — new, KEPT. It is the tool that isolated the two-books / unbounded-gap-cost
  root cause and is what any fix must be re-verified against. Manual-run diagnostic, not production code.
- `reports/layer_ablation.json` / `.log` — the ablation results these findings came from.

### Next Steps

1. **Fix the cost model's unbounded gap charge** and unify the two accounting books (feed gap costs into the
   simulator, or cap the charge) — this is what makes every ensemble-variant drawdown number unusable today.
2. **Implement constraint 6 for real**: an execution-layer cap applied by the harness to every position series
   regardless of strategy, with correlated-direction and aggregate exposure accounting, and a bankruptcy floor.
3. Then the three deferred items above.
4. Only after 1-3 is a re-run of the ablation worth the compute — until the accounting is trustworthy, no Sharpe
   or drawdown from an ensemble-enabled variant should be compared against anything.


## Session Update: The Three Deferred Items Completed (2026-08-14)

Closes the three items deferred in the previous entry. Full suite: **406 passing**.

### 1. Ablation flagging now gates on solvency BEFORE Sharpe (was: deferred item 3)

`scripts/run_layer_ablation.py::_format_comparison` rewritten. A variant with `equity_destroyed` (or
`maxDD >= 1.0`, which is itself the signature of equity reaching zero) is **DISQUALIFIED**: its Sharpe is not
printed, not ranked, and not used in any comparison. Layer verdicts are now evaluated in order — (1) does adding
the layer destroy an otherwise-solvent run, (2) profit factor, (3) max drawdown, and only then (4) Sharpe, and
only when both sides stayed solvent. Every flag carries its concrete reasons.

Replaying the actual 2026-08-14 numbers through the corrected logic **inverts the verdict**:

```
FLAGGED: Layer 7C ensemble
   reasons: adding it destroys the account; profit factor 0.238 -> 0.014;
            max drawdown 0.056 -> 1.023
```

Previously the Sharpe-only rule flagged **Layer 6** — which touches 61 of 35,069 bars (0.174%) and therefore
cannot genuinely move Sharpe — while calling Layer 7C, the layer that actually destroyed the account, a "help".
`tests/test_layer_ablation.py::test_the_real_2026_08_14_numbers_now_flag_layer7c_not_layer6` locks this in with
the real numbers.

### 2. Layer 5 sweep detection re-tuned: 44.12% -> 11.53% of bars (was: deferred item 2)

Three measurements drove this, in order, each ruling out a hypothesis:

- **Cluster tolerance was NOT the cause.** Tightening `level_tolerance_atr` from 0.25 to 0.05 moved the sweep
  rate only 44.12% -> 35.4%.
- **A "rejection wick >= X% of candle range" test was implemented, measured, and REMOVED** — it is provably
  redundant here. Confirmation already requires `close < swept_level < high`, which forces the rejection fraction
  above any sane threshold; measured, it moved the rate 44.12% -> 44.35%, i.e. it never bound. The reasoning is
  recorded as a `NOT ADDED, deliberately` comment in `liquidity_trap.py` so it is not re-added.
- **Cluster DENSITY was the real driver.** Real BTC 1h carries a **median of 7 simultaneous resistance clusters
  above price**, spread 0.76-3.01 ATR away (touch counts 2:1696, 3:1478, 4:917, 5:580...). With that many pools
  at that spread, almost any upward wick sweeps something.

Two new conditions, both squarely inside this layer's "equal highs/lows and wick behaviour" scope:

| constant | value | why this value |
|---|---|---|
| `DEFAULT_MIN_PENETRATION_ATR` | 0.25 | Set EQUAL to `DEFAULT_LEVEL_TOLERANCE_ATR` on purpose, not fitted: a cluster's level is only defined to within that tolerance, so a smaller "penetration" is inside the level's own uncertainty band. |
| `DEFAULT_MIN_SWEEP_TOUCHES` | 4 | Median cluster has 3 touches; requiring more than the median restricts sweeps to better-tested pools. A CLUSTER is still "two or more wicks" per spec — this is a separate, stricter bar for a sweep worth acting on. |

Measured tradeoff curve on real BTC 1h (kept so the choice can be revisited without re-deriving it):

```
penetration  touches   sweep rate
       0.00        2     44.12%   <- old behaviour
       0.10        2     32.71%
       0.10        4     21.07%
       0.25        4     11.53%   <- new default
       0.25        5      8.73%
       0.25        6      5.24%
```

**Two honest caveats.** (a) There is no labelled ground truth for "how often sweeps really occur", so the 2.04%
figure from the earlier compression+SFP proxy is a reference point, not a target — tuning to hit it would be
fitting to an arbitrary number. (b) More importantly: **the bearish/bullish split stays near 1.0 at EVERY
threshold tested** (197/182 at the old default, 53/46 at the new one, 24/21 at the strictest). A detector that
stays directionally balanced no matter how selective it becomes may carry no directional information at any
setting. That is consistent with the ensemble having assigned it weight exactly 0.000000, and it is a reason NOT
to re-include Layer 5 in a scoring run until it is shown to carry direction, independent of selectivity.

Mechanism tests now pass `min_sweep_touches=2` explicitly, isolating mechanism from tuning; 7 new selectivity
tests cover penetration (including ATR scale-invariance), touch significance, and the validation bounds.

### 3. Component B's lack of skill recorded structurally (was: deferred item 1)

Added `evaluate_predictive_core_skill` / `PredictiveCoreSkillDiagnostic` to `backtesting/kill_criteria.py`,
following the existing `OverfitDiagnostic` precedent: **explicitly a DIAGNOSTIC, not a sixth Layer 14 criterion.**
MASTER_CONTEXT specifies exactly five and this codebase's discipline is to add checks alongside them rather than
quietly expand the spec's list — `test_skill_diagnostic_is_not_a_layer14_criterion` asserts the criteria list
stays at five.

It flags the specific combination observed in the ablation: a signal with no measurable skill carrying most of the
ensemble weight. On the real numbers:

```
FLAGGED: predictive-core skill score 0.0100 < 0.0200 while 'component_b' carries 94.2% of
the ensemble weight. A no-skill signal dominating the blend means there is nothing to blend
-- ensemble/layer tuning CANNOT recover this. Per MASTER_CONTEXT's Layer 14 response to OOS
Sharpe < 0.5, revise the EDGE HYPOTHESIS (asset, timeframe, or setup definition), not the
parameters.
```

`NO_SKILL_SCORE_THRESHOLD = 0.02` is a documented judgment call (a logloss skill score is fractional improvement
over a base-rate baseline, so 0 is literally "no better than guessing base rates").

**The determination itself, recorded plainly:** every one of the five ablation variants failed
`layer14_oos_sharpe` and `layer14_oos_profit_factor`; three also failed `layer14_oos_max_drawdown`. Component B's
skill score has sat at 0.004-0.016 across every run since Stage 1. **This is a Layer 14 kill-criterion hit and the
correct response is hypothesis revision (Build Plan 1E: different asset, timeframe, or setup definition), not
another parameter sweep.**

### Status of the five Layer 14 criteria across the ablation

| criterion | baseline | all_layers | no_L5 | no_L6 | no_L7C |
|---|---|---|---|---|---|
| oos_sharpe | fail | fail | fail | fail | fail |
| oos_profit_factor | fail | fail | fail | fail | fail |
| oos_max_drawdown | pass | fail | fail | fail | pass |
| risk_on_regime_vs_aggregate | not_evaluated | not_evaluated | not_evaluated | not_evaluated | not_evaluated |
| ensemble_weight_stability | not_evaluated | pass | pass | pass | not_evaluated |

Still **Layer 14 research-phase only**. Tier 1 is not evaluated anywhere in this codebase.

### Next Steps (unchanged in priority from the previous entry)

1. **Fix the cost model's unbounded gap charge and unify the two accounting books** — still the blocker that makes
   every ensemble-variant drawdown number unusable. See the previous entry's audit for the full mechanism.
2. **Implement MASTER_CONTEXT constraint 6 for real** — an execution-layer cap applied by the harness to every
   position series regardless of strategy, with correlated-direction and aggregate exposure accounting, plus a
   bankruptcy floor.
3. **Act on the kill-criterion hit**: revise the edge hypothesis rather than re-running the current one. Layer 5's
   persistent directional balance (item 2 above) is additional evidence pointing the same way.
4. Only after 1-3 is another ablation run worth the compute.


## Session Update: Gap-Cost Bug Fixed — the Cause of Negative Equity (2026-08-14)

Fixes the defect identified in this session's negative-equity audit: gap cost was **96% of the total loss** in the
audited fold ($2,588 of $2,696), with the two accounting books diverged 20x. Full suite: **413 passing**.

### Three defects, not one

The audit found the charge was unbounded; working the fix showed *why* it was unbounded, and that the
unboundedness was the least of it.

**(a) Charged PER BAR, not per SESSION.** `n_weekend_held` counted every bar whose `dayofweek` was Sat/Sun. A
weekend is a single Friday-close-to-Monday-open discontinuity, but it is 48 bars on 1h data and 2,880 on 1m. The
modelled cost therefore scaled with the SAMPLING FREQUENCY rather than the risk, and grew without bound in the
holding period. Fixed with `_session_starts`, which marks the first bar of each contiguous run of the window mask.

**(b) A SYMMETRIC move charged as a one-sided cost.** `_fit_gap_stats` measures `E[|jump|]` — it uses
`np.abs(...)`, so it is as likely to help as to hurt. Charging its full magnitude on every held bar converted a
zero-mean random variable into a systematic one-way drain. Only the execution-degradation share is genuinely
one-sided: `DEFAULT_GAP_ADVERSE_FRACTION = 0.5` (a documented judgment call — you are on the wrong side of a
symmetric move half the time — NOT a source-doc number), exposed as a `gap_adverse_fraction` constructor
parameter.

**(c) DOUBLE-COUNTING.** This is the conceptual root. The realized gap is **already inside vectorbt's `PnL`**,
because PnL is computed from a close series that contains the gap. Deducting a separate "expected gap cost" on top
charged for the same price action twice. (b)'s adverse fraction is what confines the remaining charge to the
execution component PnL does *not* capture.

Also added `MAX_GAP_COST_FRAC_OF_NOTIONAL = 1.0` as a coherence bound — an EXPECTED cost cannot exceed the
position that incurs it. With (a) and (b) fixed this should never bind; if it ever does, the parameters are wrong
and that is worth surfacing rather than silently absorbing.

### Measured effect, on the pathological case that caused the blow-up

Reproduced the long-hold pattern the ablation actually produced (positions held on 99.8% of bars, 2 direction
flips):

| | gap_cost / notional per trade | book divergence | 
|---|---|---|
| before | median **0.1740** (17.4%) | 0.0083 |
| after | median **0.0082** (0.82%) | 0.0003 |

**21x reduction** in the charge, and the two-books divergence collapses to ~zero because the charge itself is now
small. 17.4% of notional per trade in "expected gap cost" was never credible; 0.82% is plausible for execution
degradation across gap-prone sessions.

On a high-turnover pattern (2,934 short trades) the same fix takes gap cost from 4% to 0% of total loss — which
explains why the pathology was invisible in ordinary runs and catastrophic in the ablation: **it scales with
HOLDING PERIOD**, and the ensemble variants held positions almost continuously.

### Testing

7 new tests in `tests/test_backtesting.py`. The strongest is
`test_gap_cost_counts_sessions_not_bars_so_it_is_timeframe_invariant`, which asserts the same calendar span yields
the same number of charged sessions at 1h and 15m sampling — directly proving the unit defect is gone. Others
cover `_session_starts` correctness, boundedness across 7/30/59-day holds, the adverse-fraction scaling
(0.0 -> no charge, 0.5 -> exactly half of 1.0), parameter validation, and that `net_pnl` still deducts the charge
(the correction reduces the cost, it does not remove the mechanism).

### KNOWN REMAINING LIMITATION — do not treat as fixed

Gap cost is still applied AFTER vectorbt has sized its orders, so **position sizing still does not see it** and
the two books still diverge slightly. That divergence is now negligible (0.0003) *because the charge is small*,
not because the architecture was fixed. A full fix requires feeding holding costs into the simulator (or an
iterative re-simulation), which this harness does not support. `equity_curve` and `Portfolio.value()` remain two
different quantities and must not be read as one.

**Separately still unfixed: MASTER_CONTEXT constraint 6.** There are still NO execution-layer risk caps — the
2%/6%/10% limits exist nowhere in the execution path (see the audit entry). This fix removes the mechanism that
drove equity negative; it does not implement the safety layer that should have made negative equity unreachable
regardless.

### Run invalidation

The 5-fold ablation in flight when this fix landed had already imported the old `costs.py`, so its numbers were
computed with the bug. Since the bug accounted for 96% of the loss in the audited fold, those results could not
answer the question they were launched for. That run was **killed and relaunched** against the corrected cost
model. Results from before this entry (`reports/layer_ablation.json`) are superseded for anything involving
drawdown, profit factor, or equity — the per-layer attribution logic they exercised is unaffected.
