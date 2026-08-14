# Project State

**Last Updated:** 2026-08-14

---

# CURRENT STATE — READ THIS FIRST

This file is ~3,800 lines and is an append-only log. **Do not read it end to end.** This block is the
current position; everything below it is history, oldest first. For detail on any item, jump to the named
session entry. The entries before the 2026-08-14 Stage 0 teardown describe a codebase that no longer exists.

## Position

**Governing documents:** `architecture/MASTER_CONTEXT_v2.md` and `architecture/BUILD_PLAN_v2.md`. v1 is
superseded in full.

**Stage 0 (Teardown) — GATE 0 CLEAR.** Awaiting operator sign-off; gates are not self-certified (§10).
**Stage A (Measurement Apparatus) — all six tasks DONE, GATE A NOT SIGNED OFF.** 4 of 5 acceptance
conditions met; the fifth is not literally satisfiable (see Open Decisions).
**Stage B (Data Acquisition) — B.1 DONE. B.2/B.3/B.4 not started. GATE B NOT REACHED.**
**Stage C (the decisive experiment) — BLOCKED on B.2 vendor procurement.**

Test suite: **361 passed** (excludes `tests/test_binance_real_data.py`, which needs network).
**Nothing has been committed.** All work is staged with the suite green.

## Gate status

| gate | status |
|---|---|
| Gate 0 | **CLEAR** — all 5 tasks; suite green; `src/` grep-clean of deleted components |
| Gate A | **4 of 5 conditions met** — leakage test passed, n_eff reported everywhere, corrected CIs published, suite green. Fold-2 reproduction not possible |
| Gate B | **NOT REACHED** — B.1 only; event count not built; ≥200-event kill not evaluated |

## What exists now

**Built and current:** Layer 1 data engine (+ funding, FRED macro, basis, 1D/1W), Layer 2 features (39),
Layers 3/4/5/6, Layer 9 risk *scoring*, Layer 14 harness (single-book, floored), purged K-Fold + embargo,
uniqueness/n_eff, trial registry + DSR, ablation flagging (I3-compliant), Layer 10 sizing **stub that
raises**.

**Deleted at Stage 0 and not coming back:** `HistoricalAnalogEngine` (Component A), the Component B
direction classifier, `EnsembleMetaModel`/`fit_ensemble_weights`, Layer 8 explainability, both strategy
adapters, the ablation runner. ~6,973 lines.

**There is no strategy in the repo. Nothing can generate a position.** By design until Stage D.

## The measurement that matters most right now

**S5 did not survive the corrected apparatus** (Stage A.6). Compressed ≤0.20 went from AUC 0.5390
CI [0.519, 0.559] to **0.5183 CI [0.4986, 0.5374] — the CI now includes 0.5.** The compression gradient
S5 described inverts, and every subset's CI overlaps the others.

Per BUILD_PLAN_v2's sequencing rationale this **does not terminate the project** — Stage A was deliberately
structured not to kill on an S5 failure, and Stage C's flow-data test is the decisive experiment. The result
is carried into Stage C as an input.

Falsification budget: **2 of 3 remaining.** None consumed to date.

## Open decisions that need the operator

1. **Gate A sign-off**, specifically the fold-2 condition. Reproducing fold 2 of `all_layers` is impossible —
   `Phase2Strategy`, the ablation runner, `HistoricalAnalogEngine` and `EnsembleMetaModel` were all deleted at
   Stage 0 on instruction, and the surviving JSON has no positions, trades, or equity curve. The mechanism was
   verified instead on real data at that fold's geometry. **Not recorded as satisfying the original.**
2. **B.2 vendor procurement** — Tardis / Coinglass / Amberdata / Kaiko. **Liquidations are named in §9 as the
   highest-priority acquisition and are unobtainable free.** Longest lead-time item; Stage C is blocked behind
   it. Flagged as Day-1 work six sessions ago, still not begun.
3. **Which dollar index** — DTWEXBGS (ingested, a *proxy*, different basket to DXY), UUP (in `config.yaml`), or
   paid DXY. Three are now reachable; Layer 3's "DXY rising" rule will be evaluated against whichever is chosen.
4. **`realised_return` pricing.** Verified to 1e-16: conditional on `barrier_touched` it is exactly
   `mult × ATR/entry` and carries **no** path information. Gate D's primary metric (top-decile realised EV)
   would rank events by entry-bar volatility, not skill. **Gate D is unreadable until this is resolved.**

## Carried forward (known, unresolved, none blocking)

- Uniqueness weighting shifts the class prior (−2.55pp) because `sl` labels resolve faster than `tp` and so
  score higher uniqueness. AUC is unaffected; logloss-vs-baseline is not comparable across weighted/unweighted.
  Hits Stage D directly.
- Vertical-barrier off-by-one: bar `i+max_candles` is checked for neither TP nor SL yet is used as the exit.
  Affects ~0.8% of labels. Deliberately not changed.
- Outlier threshold is timeframe-blind — 58.4% of 1w bars flagged. Treat 1d/1w outlier flags as absent.
- `config.yaml`'s `backtesting:` section has no reader since Stage 0.
- `WalkForwardTrainer` still drives the superseded `WalkForwardSplitter`.
- FRED publication lag is not modelled.

## Where to read next

| topic | entry |
|---|---|
| What was deleted and why | Stage 0 Teardown; Stage 0 Completion Review |
| Label metadata, overlap, duration distribution | Stage A.1 |
| Uniqueness, n_eff = 14.4% of nominal | Stage A.2 |
| Purged K-Fold, the leakage test | Stage A.3 |
| Harness single-book fix | Stage A.4 |
| Trial registry, DSR, the 2,024 lower bound | Stage A.5 |
| **Corrected ceiling table, S5 result** | **Stage A.6** |
| Data coverage and shortfalls | Stage B.1 |

---

# HISTORICAL LOG (oldest first)

**Everything below predates the current architecture.** Entries before the 2026-08-14 Stage 0 teardown
describe components that have since been deleted; they are retained as the record of how the findings in
MASTER_CONTEXT_v2 §2 were arrived at, not as a description of the codebase.

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


## Session Update: 5-Fold Ablation Re-Run Against the Corrected Cost Model (2026-08-14)

Re-ran the full 5-fold layer ablation after the gap-cost fix. **Scope unchanged: Layers 5/6/7C, not 3-9** — Layers
3 (no macro data), 4 (no 1D/1W timeframes) and 9 (needs Layers 10/11) still cannot be wired. Evaluated exclusively
against **Layer 14 research criteria; Tier 1 is not computed anywhere in this codebase.**
Results: `reports/layer_ablation_v2.json`.

### The gap-cost fix eliminated the negative equity entirely

| variant | maxDD before fix | maxDD after fix | L14 maxDD verdict |
|---|---|---|---|
| baseline_phase1 | 0.056 | **0.030** | pass (was pass) |
| all_layers | **1.023** | **0.073** | **pass** (was fail) |
| no_liquidity_trap_L5 | **1.023** | **0.073** | **pass** (was fail) |
| no_volatility_gate_L6 | **1.235** | **0.083** | **pass** (was fail) |
| no_ensemble_L7C | 0.056 | 0.030 | pass (was pass) |

No variant destroys the account, nothing is disqualified, and `layer14_oos_max_drawdown` now **passes for all
five** (previously three failed). This confirms the audit's conclusion directly: the double-counted, per-bar,
symmetric gap charge WAS the cause of negative equity — not the layers.

Baseline sanity check: trade count is **identical** (3,502) before and after the fix, confirming the correction
touched only cost accounting and not signal generation. The baseline itself improved from Sharpe -12.974 / PF
0.236 to **-7.864 / PF 0.426**.

### Results

| variant | OOS Sharpe (as reported) | OOS Sharpe (common folds) | PF | maxDD | trades |
|---|---|---|---|---|---|
| baseline_phase1 | -7.864 | -10.802 | 0.426 | 0.030 | 3502 |
| all_layers | -15.708 | -15.708 | 0.282 | 0.073 | 2817 |
| no_liquidity_trap_L5 | -15.708 | -15.708 | 0.282 | 0.073 | 2817 |
| no_volatility_gate_L6 | -20.250 | -20.250 | 0.205 | 0.083 | 2813 |
| no_ensemble_L7C | -7.432 | -10.077 | 0.429 | 0.030 | 3499 |

**FLAGGED: Layer 7C ensemble** — profit factor 0.429 -> 0.282, max drawdown 0.030 -> 0.073, and lower Sharpe.

### A methodological flaw the under-powered-fold exclusion introduced

The "as reported" and "common folds" columns differ because **the exclusion averages different variants over
different fold sets**, making the headline numbers non-comparable. The Phase 2 variants lose folds 3 and 4 (46 and
12 trades) while baseline keeps all five — and for those variants folds 3/4 were their BEST folds (+0.50, -0.41),
so excluding them removes their most favourable results while baseline keeps its own mild folds (-3.04, -3.87).

Restricting every variant to the folds that are well-powered in ALL variants (0, 1, 2):

```
Full stack vs baseline, like-for-like:  -4.906 Sharpe  (-10.802 -> -15.708)
as-reported comparison overstated this: -7.845
```

The degradation is real and in the same direction, but the as-reported figure **overstates it by 60%**. The
exclusion fix (correctly) stopped noise folds corrupting a single run's mean, but cross-variant comparison needs a
common fold set. **This is a real follow-up: `_format_comparison` should compare on common well-powered folds, not
on each variant's own subset.** PF and maxDD are unaffected — pooled PF is sample-weighted and maxDD is a max, so
neither depends on which folds entered a mean.

### Per-layer verdicts

**Layer 7C (Ensemble) — DEGRADES, and this is the robust finding.** It is the only difference between
`no_ensemble_L7C` (PF 0.429, maxDD 0.030 — indistinguishable from baseline) and `all_layers` (PF 0.282, maxDD
0.073). Crucially the verdict rests on **profit factor and max drawdown**, which are immune to the fold-set
artifact above, so it does not depend on the Sharpe comparison at all. Mechanism unchanged from the previous run:
the ensemble weights `component_b` at ~0.94, discarding the Phase 1 agreement rule that was acting as a filter.

**Layer 5 (Liquidity Trap) — exact no-op, now confirmed twice under different conditions.** `all_layers` and
`no_liquidity_trap_L5` are byte-identical again (same Sharpe to 3dp, same PF, same maxDD, same 2,817 trades), so
the ensemble is again assigning it weight ~0. This held BOTH at the old 44.1% sweep rate AND after the re-tune to
11.5% — a 4x selectivity improvement changed nothing.

Combined with the bearish/bullish split staying ~1.0 at every threshold measured (197/182 -> 53/46 -> 24/21),
**the earlier "mistuned, not architecturally flawed" reading should be revised toward the architectural side**:
re-tuning improved the detector's specificity without giving it directional content. Do not re-include Layer 5 on
the assumption that further tuning will help.

**Layer 6 (Volatility Gate) — the reported "+4.542 Sharpe, helps" is NOT trustworthy.** It differs from
`all_layers` by **4 trades** (2,817 vs 2,813) yet swings fold-1 Sharpe from -18.81 to -29.20 on a ONE-trade
difference (1,076 vs 1,075). That is not a layer effect; it is Sharpe instability. With ~2% position sizes the
equity curve is mostly flat with occasional small steps, so the returns series has a very small standard
deviation and Sharpe is numerically fragile — this is a *different* fragility from the destroyed-equity case
already guarded against, and it survives even with a healthy drawdown. The gate touches 61 of 35,069 bars
(0.174%). **Verdict: still untested, not "helpful".**

### Layer 14 status (research phase — NOT the Tier 1 live gate)

| criterion | baseline | all_layers | no_L5 | no_L6 | no_L7C |
|---|---|---|---|---|---|
| oos_sharpe | fail | fail | fail | fail | fail |
| oos_profit_factor | fail | fail | fail | fail | fail |
| oos_max_drawdown | **pass** | **pass** | **pass** | **pass** | **pass** |
| risk_on_regime_vs_aggregate | not_evaluated | not_evaluated | not_evaluated | not_evaluated | not_evaluated |
| ensemble_weight_stability | not_evaluated | pass | pass | pass | not_evaluated |

Every variant still fails OOS Sharpe (best: -7.43 against a 0.5 threshold) and profit factor (best: 0.429 against
1.2). **Fixing the accounting bug did not rescue the strategy** — it was inflating how bad the result looked, not
creating it. The Layer 14 kill-criterion determination recorded earlier stands unchanged: this is a
hypothesis-revision trigger, not a tuning problem.

`risk_on_regime_vs_aggregate` remains structurally uncomputable without Layer 3, so under the user's
all-five-must-evaluate bar, **Phase 1E still cannot be assessed**.

### Next Steps

1. **Make cross-variant Sharpe comparison use a common well-powered fold set** (see the methodological flaw
   above) — currently each variant is averaged over its own subset.
2. **Do not keep Layer 7C in this form** — it is the flagged degrader on PF and maxDD, metrics that do not depend
   on the fold-set artifact.
3. **Treat Layer 5 as architecturally unproven, not mistuned** — two independent confirmations of zero
   contribution at very different selectivity levels.
4. **Layer 6 remains unfalsifiable on this dataset** at 0.174% of bars; it needs data where it actually fires.
5. **MASTER_CONTEXT constraint 6 is still unimplemented** — no execution-layer risk caps exist. The gap-cost fix
   removed the mechanism that drove equity negative; it did not add the safety layer that should have made
   negative equity unreachable regardless.
6. Sharpe is numerically fragile at these position sizes even with healthy drawdowns (see the Layer 6 finding).
   Consider reporting trade-level EV/PF as the primary comparison metric and treating Sharpe as secondary until
   position sizing is real (Layer 10).


## Session Update: Signal-Ceiling Re-Measurement — the Label Is Barrier Geometry (2026-08-14)

Re-measured the predictive ceiling of the current 39-feature set on the full ~9-year real BTC/USDT 1h dataset
(78,378 rows), reproducing the Stage 4 methodology so numbers are directly comparable, and extending it two ways:
conditioned on Layer 6 volatility state, and across four triple-barrier geometries. Tool:
`scripts/measure_signal_ceiling.py` (kept). **This entry supersedes Stage 4's interpretation.**

### Updated AUC (unchanged from Stage 4, as expected)

| measure | value |
|---|---|
| Best univariate feature (`mom_di_spread`) | **AUC 0.5172** |
| Median \|AUC-0.5\| across all 39 features | **0.0024** |
| Multivariate ceiling, all bars, generous LightGBM | **AUC 0.5251**, 95% CI [0.514, 0.536] |

Identical to Stage 4 (0.5172 / 0.0024 / 0.5289) — expected, since the only feature change since then was
`vol_spike` on 17 of 78,381 rows.

### THE CENTRAL FINDING: the label is ~pure barrier geometry, not market structure

For a driftless random walk, the probability of touching +tp before -sl is exactly `sl / (tp + sl)`. Relabelling
the same 9 years of real BTC at four geometries:

| geometry | observed up rate | driftless-walk prediction | gap | test AUC | boost rounds |
|---|---|---|---|---|---|
| tp=2.0 / sl=1.0 (current) | 0.337 | 0.333 | +0.004 | 0.5211 | 11 |
| tp=1.0 / sl=1.0 (symmetric) | 0.507 | 0.500 | +0.007 | **0.5094** | 3 |
| tp=1.0 / sl=2.0 (inverted) | 0.675 | 0.667 | +0.008 | 0.5079 | 11 |
| tp=3.0 / sl=1.0 | 0.261 | 0.250 | +0.011 | 0.5136 | 7 |

**Nine years of real BTC produces a label distribution within 0.4-1.1 percentage points of a driftless random
walk at every geometry tested.** The class balance the models have been fighting all project is not a property of
the market — it is arithmetic. The residual +0.4 to +1.1pp is BTC's genuine upward drift: real, and tiny.

**Corollary, and the reason this matters more than the AUC number itself:** at SYMMETRIC barriers, where there is
no class imbalance to exploit, AUC collapses to **0.5094** and early stopping halts after **3 rounds**. Much of
the apparent 0.5251 on asymmetric barriers is therefore the model learning the *base rate*, not direction. The
genuine directional signal in this feature set is close to 0.509 — i.e. essentially nil.

### Signal DOES concentrate in compression (the Layer 6 result)

Multivariate ceiling, same features, conditioned on `vol_atr_pct_rank`:

| subset | n | test AUC | 95% CI | logloss vs base | rounds |
|---|---|---|---|---|---|
| ALL bars (unconditioned) | 78,226 | 0.5251 | [0.514, 0.536] | +0.00102 | 7 |
| Compressed, ATR pct < 0.30 (Layer 6 table) | 27,780 | 0.5238 | [0.506, 0.542] | -0.00082 | 32 |
| **Compressed, ATR pct <= 0.20 (hypothesis text)** | 20,198 | **0.5390** | **[0.519, 0.559]** | **+0.00170** | 5 |
| Expanding, 0.30-0.70 | 27,757 | 0.5140 | [0.494, 0.533] | +0.00030 | 2 |
| Extended, ATR pct > 0.70 | 22,689 | 0.5101 | [0.488, 0.532] | +0.00027 | 4 |

Read carefully, because the honest claim is narrower than it first looks:

- **Expanding and Extended have CIs that straddle 0.5** -> no detectable signal in either.
- **Compressed (<=0.20) has a CI entirely above 0.5** ([0.519, 0.559]) and the largest logloss improvement over
  baseline. Something is there.
- **BUT** that CI overlaps heavily with the unconditioned CI [0.514, 0.536], so "compression is significantly
  better than unconditioned" is NOT established. The defensible statement is: *signal is present in compression
  and absent in expansion/extension*, which is a gradient consistent with the hypothesis, not proof of it.
- The stricter 0.20 threshold from the Adaptive doc's hypothesis section beats Layer 6's own 0.30 table threshold
  (0.5390 vs 0.5238), and 0.30 is the only subset whose logloss is WORSE than baseline.

This is the empirical basis for treating Layer 6 as the next priority: compression is condition 1 of the
four-condition hypothesis, it is the ONLY one computable from data this project ingests, and it is the only
volatility state where the ceiling is measurably above chance.

### Why the models behave as they do

- **Early stopping halts after 2-32 rounds** in every configuration. A model finding real structure trains for
  hundreds. Halting almost immediately is the clearest single indicator that there is nothing to fit.
- **`vol_state_compressed` is currently one of 39 undifferentiated input features, never a filter.** The model is
  asked to predict every candle in 9 years, so a signal confined to ~26% of bars is diluted by the 74% where the
  measurement above shows there is none.
- **Percentile-ranking over a trailing 252 bars destroys absolute level information** (`feature_calculator`'s
  "no raw values leave this module" rule). Every feature is a within-window rank, so the model cannot see regime
  magnitude, only relative position.

### Next Steps

1. **Rebuild around compression as a FILTER, not a feature** — train and evaluate only on candles where
   `vol_atr_pct_rank <= 0.20`. This is the one conditioning step available today and the only subset with a
   ceiling measurably above chance.
2. **Reconsider the triple-barrier label.** It is currently ~pure geometry. Any label whose class balance is
   predicted to within 1pp by `sl/(tp+sl)` is measuring the barrier, not the market. Consider fixed-horizon
   returns, or barriers scaled to a compression-conditional volatility estimate rather than trailing ATR.
3. **Improve Layer 6 itself** (see the dedicated next entry / discussion): duration and slope of compression, not
   just a point-in-time percentile; reconcile the 0.20 vs 0.30 thresholds; fix Spike, which fires on 0.174% of
   bars and is currently unfalsifiable.
4. Everything previously recorded stands: Layer 7C degrades and is flagged, Layer 5 contributes exactly nothing
   at two very different selectivity settings, constraint 6 is unimplemented.

## Session Update: Stage 0 Teardown (2026-08-14)

**First session executed under BUILD_PLAN_v2 / MASTER_CONTEXT_v2.** This is Stage 0 (Teardown), gating on
Gate 0. No measurement was produced this session and none was intended: Stage 0 is deletion, and the deliverable
is a smaller repo plus a green suite, not a number. Nothing was committed.

### Test suite count

| point | tests |
|---|---|
| Before teardown | **413 passed** |
| After teardown | **262 passed** |
| Net | **-151** |

(Both figures exclude `tests/test_binance_real_data.py`, which requires live network access. Zero failures at
both points. MASTER_CONTEXT_v2 section 8 records "384 passing"; the real pre-teardown number was 413.)

Net diff: **24 files changed, 295 insertions, 6,973 deletions.**

### Deleted outright (13 files)

Per MASTER_CONTEXT_v2 section 8, on operator instruction:

| file | lines | why |
|---|---|---|
| `src/predictive_core/analog_engine.py` | 429 | Component A / k-NN. Distance metric weighted by Component B importances, inherits its uselessness |
| `src/predictive_core/ensemble.py` | 961 | `EnsembleMetaModel`, `fit_ensemble_weights` (S7; destroyed the account in ablation) |
| `src/predictive_core/explainability.py` | 832 | Layer 8. Orphan: `EnsembleSignalSet` was its only input type |
| `src/backtesting/phase2_strategy.py` | 475 | Orphan: 100% consumer of A + B-direction + 7C |
| `src/backtesting/predictive_core_strategy.py` | 385 | Orphan: same |
| `scripts/run_layer_ablation.py` | 414 | Orphan: drove `Phase2Strategy` |
| `scripts/run_walkforward_backtest.py` | 270 | Orphan: drove `PredictiveCoreStrategy` |
| `scripts/diagnose_negative_equity.py` | 98 | Orphan: drove `Phase2Strategy` |
| `tests/test_ensemble.py` | 799 | Covered deleted component only |
| `tests/test_explainability.py` | 764 | Covered deleted component only |
| `tests/test_layer_ablation.py` | 374 | Covered deleted component only |
| `tests/test_phase2_strategy.py` | 298 | Covered deleted component only |
| `tests/test_predictive_core_strategy.py` | 116 | Covered deleted component only |

### Cut surgically

- **`src/predictive_core/boosted_core.py` (439 -> 240 lines).** Removed `train_direction_classifier`,
  `_direction_objective`, `_direction_to_class_idx`, `DirectionFoldResult`, `DIRECTION_CLASS_LABELS`, the
  `IsotonicRegression` calibration block and its `calibrators_` attachment, and `_compute_class_weights` (the
  balanced class-weighting machinery built for the 65/34 imbalance, S1). `FoldResult` lost its `.direction`
  field; `WalkForwardTrainer` no longer trains a classifier.
- **Retained in the same file:** `train_quantile_regressor`, `_quantile_objective`, `QuantileFoldResult`,
  `QUANTILES`, and the Optuna infrastructure. Section 8 names "Component B **direction classifier**"
  specifically, and section 8's KEEP list retains Optuna infrastructure. The magnitude/duration quantile heads
  were not named and were not deleted -- but they now have **no caller in `src/`**, since both strategy adapters
  are gone.

### Two judgment calls made explicitly, both retentions, both reversible

Section 8's DELETE list includes "calibration work premised on that imbalance." That phrase is ambiguous across
two different things, and they were treated differently:

1. **`calibration.py` (ECE / Brier) -- RETAINED.** ECE and Brier are geometry-independent measurements of any
   probabilistic classifier; they are not premised on any class balance. The imbalance-premised artefact was the
   *isotonic calibrator fit to the 65/34 three-class problem*, which was deleted with the classifier.
   MASTER_CONTEXT_v2 section 1 ranks measurement apparatus above models, and BUILD_PLAN_v2 D.3 requires
   "calibration on the event set" for the Stage D meta-label model.
2. **`skill_diagnostics.py` -- RETAINED.** These are the diagnostics that *measured* the direction classifier as
   skill-less (0.004-0.016), i.e. they produced the number that killed their own caller. BUILD_PLAN_v2 Stage C
   requires "logloss vs baseline" for all three ceiling-test arms.

Both now have **no caller in `src/`**. Both carry a `STAGE 0 STATUS` docstring block recording that, so an
unused module is not mistaken for a live one. If the operator disagrees, deleting them is a two-file change.

### Callers found by the Gate 0 grep sweep, and what was done with each

Every reference to a deleted symbol, and its disposition. **`src/` is now completely clean -- the sweep returns
zero matches.**

| caller | reference | disposition |
|---|---|---|
| `src/predictive_core/__init__.py` | exported A, B-direction, calibration symbols | Rewritten; exports reduced to labelling, splitter, calibration, quantile regressor |
| `src/predictive_core/dataset.py` | 5 docstring refs to A / `PredictiveCoreStrategy` | Docstrings rewritten to name surviving consumers |
| `src/predictive_core/labeling.py` | `attrs['label_horizon']` rationale cited `PredictiveCoreStrategy` | Rewritten; contract retained and re-pointed at Stage A.3, which needs real per-sample `t1` |
| `src/predictive_core/skill_diagnostics.py` | argmax rule cited `PredictiveCoreStrategy` | Generalised to "a consuming strategy" |
| `src/risk_sizing_selection/risk_engine.py` | prereqs cited `analog_engine`; pattern cited `explainability.py` | Rewritten; added a Stage 0 caveat that Layer 9 now has no live caller until Stage E |
| `src/macro_mtf_liquidity_vol/macro_regime.py` | 3 illustrative refs to `HistoricalAnalogEngine` | Names removed, points preserved |
| `tests/test_macro_regime.py` | tz-bug regression test cited A by name | Reworded to "the since-deleted k-NN analog engine" |
| `src/backtesting/__init__.py` | none | Untouched -- never imported the strategies |
| `tests/test_backtesting.py` | 18 apparent hits | **Untouched.** All are the generic `SignalStrategy.ensemble_weights()` protocol and the Layer 14 stability criterion, driven by in-file fake strategies. No dependency on `EnsembleMetaModel`. Verified before concluding, not assumed |

**One intentional exception to "zero references":** `tests/test_predictive_core.py` names six deleted tests in
its header, as the record of what Stage 0 pruned and why. That is a deletion manifest, not a dependency. Flagged
for the operator rather than quietly scrubbed.

### Pruned rather than deleted

`tests/test_predictive_core.py` (458 -> 90 lines). Deleted: `test_minimum_sample_gate_enforced`,
`test_deduplication_is_asset_aware_and_uses_real_time`, `test_predict_accepts_tz_aware_query_timestamp`,
`test_deduplicate_matches_naive_reference_and_is_fast` and its `_naive_deduplicate_reference` helper (all
Component A), both class-weight tests, and the `main()` narrative script. Retained: splitter fold-isolation,
`apply_triple_barrier` label-horizon metadata. Rewritten:
`test_walk_forward_trainer_runs_end_to_end_with_calibration` -> `..._runs_quantile_regressors_end_to_end`, with
the isotonic-calibration assertions removed and pinball-loss assertions kept.

### Gate 0 status: NOT CLEAR -- 3 of 5 tasks done

| task | status |
|---|---|
| 0.1 Delete A, B-direction, `EnsembleMetaModel`, `fit_ensemble_weights` | **DONE** -- grep clean in `src/` |
| 0.2 Delete class-weighting machinery and imbalance-premised calibration | **DONE** -- see judgment calls above |
| 0.3 Remove `Phase2Strategy.max_position`; leave a hard `NotImplementedError` stub where real caps go | **NOT DONE** |
| 0.4 Update Layer 6 threshold 0.30 -> 0.20, single-sourced | **NOT DONE** |
| 0.5 Prune tests; record count before/after | **DONE** -- 413 -> 262 |

0.3 and 0.4 were **not attempted** because the operator's instruction this session scoped the work to section
8's DELETE list. They are not blocked -- both are small and understood:

- **0.3.** `Phase2Strategy.max_position` no longer exists (the class was deleted), so the first half is
  satisfied incidentally. The second half is not: **there is nowhere in the repo that the `NotImplementedError`
  stub belongs.** `risk_engine.py` is a *scorer* -- it accepts `position_size_pct` as an input and never produces
  one. There is now **no sizing arithmetic anywhere in the codebase.** Satisfying 0.3's acceptance ("tests fail
  loudly if anything sizes without caps") requires creating a new module, which is a build decision, not a
  teardown one.
- **0.4.** `COMPRESSED_MAX_PERCENTILE = 0.30` at `src/macro_mtf_liquidity_vol/volatility_classifier.py:111`,
  already single-sourced. A one-line change plus a duplicate sweep of `config.yaml`. S6 settled that 0.20 is
  correct.

### Also unresolved, and on the critical path

**`trials.jsonl` does not exist anywhere in the repo.** MASTER_CONTEXT_v2 section 6 requires every evaluated
configuration to be logged there, and DSR -- a Stage E gate -- cannot be computed honestly without it. Zero
model configurations were evaluated this session (teardown produces no measurements), so nothing was owed to it
today, but the file's absence means **every configuration evaluated before this point is unlogged and
unrecoverable**, and the trial count feeding any future DSR will understate reality. Creating it is Stage A.5,
which defines its schema; it was deliberately not created ahead of that.

### What the repo is now

Layers 1 (data), 2 (features), 3 (macro regime), 4 (MTF), 5 (liquidity trap), 6 (volatility), 9 (risk scoring),
and 14 (backtest harness) survive. The predictive core is now labelling + splitting + dataset assembly +
quantile regression + two unreferenced measurement modules. **There is no strategy in the repo.** Nothing can
currently generate a position, by design -- Stage D builds the primary rule and the meta-label model, and Stage
E builds the risk caps and exits beneath it.

### Next Steps

1. **Operator decision on Gate 0's remaining two tasks** (0.3's stub location, 0.4's one-line threshold change).
   Gate 0 is not self-certifiable by the agent (section 10).
2. Then **Stage A -- Measurement Apparatus**, the highest-priority work in the project: A.1 label metadata
   (`t0`/`t1`/`barrier_touched`/`realised_return`), A.2 uniqueness weighting, A.3 Purged K-Fold + embargo
   replacing `WalkForwardSplitter`, A.4 the four harness defect fixes (S11/S12/I3), A.5 `trials.jsonl` + DSR,
   A.6 recompute the Stage 5 ceiling under the corrected apparatus.
3. **Stage B data acquisition should start in parallel now** -- BUILD_PLAN_v2's sequencing rationale puts it on
   Day 1 because vendor lead time is the longest pole. Nothing in Stage 0 or A blocks it.


## Session Update: Stage 0 Tasks 0.3 and 0.4 -- GATE 0 CLEAR (2026-08-14)

Completes the two tasks left outstanding by the teardown entry above. Suite: **262 -> 270 passed**, zero
failures. Nothing committed.

### 0.3 -- Execution-layer risk cap stub

`Phase2Strategy.max_position` was already gone (the class was deleted in the teardown), so the removal half was
satisfied incidentally. The stub half required a new home, because **there is no sizing arithmetic anywhere in
the repo** and `risk_engine.py` is a scorer that consumes `position_size_pct` rather than producing one.

Created **`src/risk_sizing_selection/position_sizing.py`** -- Layer 10, every entry point raising
`ExecutionRiskCapsNotImplemented(NotImplementedError)`:

- `calculate_position_size()` -- the only sanctioned path from signal to size. Raises.
- `apply_execution_risk_caps()` -- would clamp to 2/6/10%. Raises.
- `SINGLE_TRADE_RISK_CAP_PCT = 2.0`, `CORRELATED_DIRECTION_RISK_CAP_PCT = 6.0`,
  `AGGREGATE_OPEN_RISK_CAP_PCT = 10.0` -- declared in the module that will enforce them, rather than existing
  only inside a scoring formula that cannot. Currently declarations, not controls.

The module docstring records why `max_position` was never a risk cap, in the two independent senses: it capped
SIZE not RISK (risk = size x stop distance, and there are still no stops -- Layer 13/Stage E.2), and it sat in
the MODEL layer where a model output could route around it.

Created **`tests/test_position_sizing.py`** (7 tests), asserting a refusal rather than a behaviour:

- both entry points raise, including when called with plausible guessed signatures (the raise cannot be routed
  around by passing arguments that look real);
- the exception remains a `NotImplementedError` subclass so generic handlers still catch it;
- the message names S10 and all three cap figures, so a traceback says WHY sizing refused rather than merely
  that something is unfinished;
- the cap constants equal MASTER_CONTEXT's exact figures;
- a reflective test enumerates `__all__` and asserts EVERY public callable refuses -- so adding a working sizing
  function without also implementing caps fails here.

**These tests are expected to fail when Stage E.1 lands.** That is intended: at that moment they should be
replaced by E.1's acceptance test (a malicious model output demanding 50% size comes back clamped), not quietly
deleted to keep a suite green. The test module says so in its own docstring.

### 0.4 -- Layer 6 compression threshold 0.30 -> 0.20 (S6)

**The constant WAS duplicated. Two separate duplications were found, both in `features/feature_calculator.py`:**

| duplication | detail |
|---|---|
| Band edges | `feature_calculator.py:290-294` hardcoded `0.3` and `0.7` in three independent boolean masks, entirely unaware of Layer 6's `COMPRESSED_MAX_PERCENTILE`/`EXTENDED_MIN_PERCENTILE` |
| Spike constants | `feature_calculator.py:34-35` re-declared its own `SPIKE_LOOKBACK_CANDLES = 5` / `SPIKE_RATIO_THRESHOLD = 2.0` -- under a comment claiming they were "kept as named constants so `vol_spike` and the Layer 6 classifier reference the same numbers", which was true only by coincidence of both literals happening to match |

This mattered more than a tidiness issue. Layer 6's `classify_volatility_state` assigns Expanding via a final
`else`, so moving its compression edge is self-consistent by construction. But `feature_calculator` builds three
*independent* masks, so changing only its compressed edge would have left every bar in [0.20, 0.30) belonging to
**no state at all** -- a silent hole in a model input column, not a visible error.

Resolution: `volatility_classifier.py` is now the declared single source; `feature_calculator.py` imports all
four constants and declares none. No cycle -- `volatility_classifier` imports only `math`/`dataclasses`/`typing`.

The comparison stays strict (`<`), not `<=` as the hypothesis text writes it. Verified rather than argued (see
below): exactly 0.20 never occurs.

### Verification against real data (not asserted -- measured)

Ran the changed pipeline over the cached 9-year real BTC/USDT 1h frame (78,666 raw rows, 78,381 with a defined
`vol_atr_pct_rank`):

| check | result |
|---|---|
| `vol_state_compressed` now fires on | **20,201 rows** (was 27,791 at <0.30) |
| Band [0.20, 0.30) moved out of Compressed | **7,590 rows = 9.7% of the dataset** |
| Agreement with hypothesis-text `<= 0.20` | 20,201 -- **identical** |
| Partition integrity across all 78,381 rows | **every row in exactly one state**; states-per-row counts `{1: 78381}`. No gap opened |
| Rows where `vol_atr_pct_rank == 0.20` exactly | **0** -- `rank(pct=True)` over 252 bars yields k/252 for integer-or-half-integer k, and 0.20 would need k=50.4. The `<` vs `<=` distinction is empirically empty on this data |

The 20,201 / 27,791 figures reconcile with the previously recorded 20,198 / 27,780; the small difference is the
denominator (78,381 feature rows here vs 78,226 modeling rows after labelling and neutral-class exclusion).

**This is a state change for ~10% of all bars in a column that feeds every model.** Any measurement taken before
this change that used `vol_state_compressed` or Layer 6's state is not directly comparable to one taken after.

### Gate 0 status: CLEAR -- all 5 tasks done

| task | status |
|---|---|
| 0.1 Delete Component A, B-direction, `EnsembleMetaModel`, `fit_ensemble_weights` | **DONE** -- `src/` grep-clean |
| 0.2 Delete class-weighting machinery and imbalance-premised calibration | **DONE** |
| 0.3 Remove `max_position` as a risk mechanism; hard `NotImplementedError` stub | **DONE** |
| 0.4 Layer 6 threshold 0.30 -> 0.20, single-sourced | **DONE** -- two duplications found and collapsed |
| 0.5 Prune tests; record count before/after | **DONE** -- 413 -> 262 -> 270 |

Gate 0's acceptance conditions: test suite green (**270 passed, 0 failed**); repo contains no reference to
deleted components (`src/` returns zero matches; the one intentional exception is the deletion manifest in
`tests/test_predictive_core.py`'s header, flagged rather than scrubbed); counts recorded here.

**Gate 0 is not self-certified by the agent (MASTER_CONTEXT_v2 section 10). It requires operator sign-off before
Stage A begins.**

### Carried forward unchanged

- **`trials.jsonl` still does not exist.** Zero model configurations were evaluated across either Stage 0
  session. Creation is Stage A.5, which defines its schema.
- `calibration.py` and `skill_diagnostics.py` remain retained-but-uncalled, each carrying a `STAGE 0 STATUS`
  docstring. So does `train_quantile_regressor`.
- Nothing in the repo can generate a position. By design until Stage D.

### Next Steps

1. **Operator sign-off on Gate 0.**
2. **Stage A -- Measurement Apparatus** (A.1 label metadata `t0`/`t1`/`barrier_touched`/`realised_return`, A.2
   uniqueness weighting, A.3 Purged K-Fold + embargo, A.4 the S11/S12/I3 harness fixes, A.5 `trials.jsonl` +
   DSR, A.6 recompute the ceiling under corrected apparatus). Expect A.6 to widen the compressed-regime CI
   substantially -- BUILD_PLAN_v2 A.6 states in advance that `[0.519, 0.559]` straddling 0.5 is a likely and
   honest outcome, not a failure.
3. **Stage B data acquisition in parallel, starting now** -- vendor lead time is the longest pole and nothing in
   Stage 0 or A blocks it.


## Session Update: Stage 0 Completion Review -- 2 Stale-Doc Defects Found and Fixed (2026-08-14)

Audit of the Stage 0 work above against BUILD_PLAN_v2's task table and MASTER_CONTEXT_v2 section 8, re-deriving
each acceptance condition from the source documents rather than from the preceding entries. Suite **270 passed**,
unchanged. Nothing committed.

### Two real defects found -- both stale documentation asserting the superseded threshold

The code change for 0.4 was correct, but two places still *stated* 0.30, which is the exact divergence class that
task 0.4 exists to eliminate:

1. **`volatility_classifier.py` header docstring (lines 2-11)** -- the module's own "MASTER_CONTEXT's exact
   table" still read `Compressed: ATR < 30th percentile` and `Expanding: 30th-70th percentile`. The authoritative
   prose description of Layer 6 contradicted the constant defined 120 lines below it. Rewritten to state 0.20,
   the 0.20-0.70 Expanding band, and why (S6), with an explicit note that the rest of the table is unchanged.
2. **`scripts/measure_signal_ceiling.py:124`** -- labelled the `< 0.30` arm "Layer 6 table", which now points at
   a value Layer 6 does not use. Relabelled: 0.20 is "LIVE Layer 6", 0.30 is "SUPERSEDED value". Added the
   previously-missing `0.20 <= pct <= 0.70` Expanding arm, so the live band is actually measurable (only the old
   0.30-0.70 band was).

   The thresholds themselves were deliberately left hardcoded rather than imported from Layer 6, with a comment
   recording why: this is a measurement tool, and importing the constant would silently redefine "Compressed"
   every time Layer 6 is retuned, making future runs incomparable with the numbers already recorded in this file.

### Verified independently (not restated from the entries above)

| check | method | result |
|---|---|---|
| Deleted modules genuinely gone | `importlib.import_module` on all 5 | all raise `ModuleNotFoundError` |
| No symbol leaks via package re-export | `hasattr` on `src.predictive_core` for 5 deleted names | all absent |
| Layer 6 constant truly single-sourced | identity check across modules | `volatility_classifier.COMPRESSED_MAX_PERCENTILE is feature_calculator.COMPRESSED_MAX_PERCENTILE` -> `True`, value `0.2` |
| No deleted-component refs in code/config | grep over `*.py,*.yaml,*.yml,*.toml,*.cfg,*.ini` | zero. All remaining hits are in `docs/project_state.md` (historical log) and `architecture/` (the specs that ordered the deletion) |
| `config.yaml` clean | full read | no compression threshold, no analog/ensemble model config. The two `ensemble_instability_*` keys are Layer 14 kill-criteria config for the RETAINED harness, not `EnsembleMetaModel` |
| No stray band literals | grep for `0.3`/`0.7`/`0.2` across `src/` | remaining hits are unrelated constants: Layer 3's `CORRELATION_REDUCED_WEIGHT_THRESHOLD`/`DEFAULT_RISK_OFF_CORRELATION_THRESHOLD`, the ensemble-weight-delta kill threshold, and an Optuna learning-rate bound |
| Suite green | full run | **270 passed, 0 failed** |

### Confirmed in scope, and correctly NOT done at Stage 0

Section 8's FIX table contains six entries; only ONE belongs to Stage 0. Checked so that "the FIX table isn't
done" is not mistaken for incomplete teardown:

| FIX entry | belongs to | status |
|---|---|---|
| Layer 6 threshold 0.30 -> 0.20 | **Stage 0.4** | **DONE** |
| `equity_curve_from_trade_pnls` bankruptcy floor (S11) | Stage A.4 | correctly not started |
| Harness two-book accounting (S12) | Stage A.4 | correctly not started |
| Gap cost model cap (S11) | Stage A.4 | correctly not started |
| Ablation flagging maxDD/PF before Sharpe (I3) | Stage A.4 | correctly not started |
| `DEFAULT_LEVEL_TOLERANCE_ATR` 0.25 retune (S8) | Stage B.4 | correctly not started |

### Gate 0: CLEAR (re-confirmed after fixes)

All five tasks done; both Gate 0 acceptance conditions hold (suite green; no reference to deleted components in
code or config, recorded here). Test count across the whole of Stage 0: **413 -> 262 -> 270**.

Still requires operator sign-off -- Gate 0 is not self-certified by the agent (section 10).

### Carried forward, unchanged

- **`trials.jsonl` still does not exist.** Zero model configurations evaluated in any Stage 0 session. Stage A.5.
- `calibration.py`, `skill_diagnostics.py`, `train_quantile_regressor` retained but uncalled, each flagged in its
  own docstring.
- **`config.yaml`'s entire `backtesting:` section is now unread** -- its only consumer was the deleted
  `run_walkforward_backtest.py`. Not a Stage 0 defect (nothing in section 8 names it), but it is dead config that
  Stage A.4 should re-wire or remove when it touches the harness.
- Nothing in the repo can generate a position. By design until Stage D.


## Session Update: Stage A.1 -- Label Metadata + Overlap Measurement (2026-08-14)

Stage A, task A.1, gating on **Gate A** (apparatus, not hypothesis). Suite **270 -> 284 passed**. Nothing
committed. Gate A is NOT reached -- A.2 through A.6 remain.

### What A.1 added

`apply_triple_barrier` now emits four per-sample columns:

| column | meaning |
|---|---|
| `t0` | event time. Also the index, but emitted as a column because `pool_assets` yields a NON-UNIQUE index (BTC and ETH share timestamps), so post-pooling code cannot recover event time from the index alone |
| `t1` | barrier-touch or vertical-barrier time |
| `barrier_touched` | `'tp'` / `'sl'` / `'vertical'` |
| `realised_return` | signed fractional return t0 -> t1, long-only convention |

`t1` is resolved by BAR POSITION, then converted to a timestamp, rather than by wall-clock arithmetic on `t0`.
This matters because exchange outages leave real gaps in the index (`data_engine` classifies them), so
`t0 + 48h` is not reliably a bar that exists. Every t1 is therefore guaranteed to be an actually-observed
timestamp -- asserted in `test_t1_is_always_an_observed_bar_timestamp` and
`test_t1_survives_a_gapped_irregular_index`.

`duration` already encoded this offset as a bar COUNT, which is why the backfill was possible at all -- but a
count is unusable for purging without resolution to a timestamp. An internal assertion in the labeller now
raises if `t1 != t0 + duration` for any row, because the tp/sl and vertical branches reach `touch_idx` by
different routes and a divergence would silently corrupt every future purge boundary.

Added `tests/test_label_metadata.py` (14 tests).

### Backfill note

There is no persisted labelled dataset in this repo -- `data/raw/` holds raw OHLCV only. "Backfill" therefore
means regenerating the labelled frame through the same `build_asset_modeling_frame` path every prior
measurement used, now carrying the metadata. Tool: `scripts/measure_label_overlap.py` (kept).

### MEASURED: label duration (t1 - t0), in bars

Full real datasets, `max_candles=48` (config.yaml's value).

| statistic | BTC/USDT 1h | ETH/USDT 1h |
|---|---|---|
| n labels | 78,378 | 35,073 |
| min | 1 | 1 |
| **median** | **5** | **5** |
| mean | 7.48 | 7.15 |
| p75 | 10 | 9 |
| **p90** | **17** | **16** |
| p99 | 45 | 45 |
| **max** | **48** | **48** |

Heavily right-skewed: the median label resolves in 5 bars, but the tail runs to the 48-bar vertical barrier.

### Why the distribution has that shape (inspected, not assumed)

| route | BTC n | share | median dur | p90 dur |
|---|---|---|---|---|
| `sl` | 51,804 | 66.10% | 4 | 14 |
| `tp` | 25,947 | 33.10% | 7 | 19 |
| `vertical` | 627 | **0.80%** | 48 | 48 |

The short median is barrier geometry, not fast markets. The stop sits at 1xATR and the target at 2xATR, so the
nearer barrier is hit first and hit sooner -- `sl` resolves in a median 4 bars against `tp`'s 7. **The vertical
barrier is almost never reached: 0.8% of labels.** The 48-bar horizon is therefore nearly irrelevant to the
duration distribution; it binds on fewer than 1 in 100 samples. Consistent with S1 (the 65/34 split is
`sl/(tp+sl)` arithmetic), and this is the same fact viewed in the time domain.

### MEASURED: overlap

Concurrency `c_t` = number of label windows `[t0_i, t1_i]` spanning bar t.

| statistic | BTC | ETH |
|---|---|---|
| bars spanned by >=1 label | 78,379 of 78,379 | 35,074 of 35,074 |
| **mean concurrency** | **8.48** | **8.15** |
| median concurrency | 7 | 7 |
| p90 concurrency | 16 | 15 |
| max concurrency | 43 | 42 |

Mean concurrency (8.48) equals mean duration + 1 (7.48 + 1), as it must when one label starts per bar -- an
internal consistency check, not an independent finding.

**Roughly 8-9 labels read every bar.** Nominal n is overstating independent information by close to an order of
magnitude.

### PREVIEW of effective n -- NOT the A.2 implementation

Computed directly from the concurrency counts to size the problem before A.2 is built. A.2 implements this
properly and wires it into training weights; nothing here touches the training path.

| | BTC | ETH |
|---|---|---|
| nominal n | 78,378 | 35,073 |
| mean uniqueness | 0.1442 | 0.1503 |
| **effective n** | **11,306 (14.42%)** | **5,270 (15.03%)** |

**This is higher than BUILD_PLAN_v2 A.6 predicted.** That section states "effective n is plausibly 3-5% of
nominal". Measured is ~14-15%, roughly 3x more favourable. CI widening at A.6 should therefore be milder than
the plan anticipates -- on the order of sqrt(1/0.1442) ~ 2.6x, not sqrt(1/0.04) = 5x. The compressed-regime CI
`[0.519, 0.559]` (half-width 0.020) would widen to roughly +/-0.053, i.e. about `[0.487, 0.591]` -- **still
straddling 0.5.** A.6 must compute this properly rather than rely on that scaling; it is recorded here as an
expectation set in advance, so the real number is not read as a surprise either way.

### FINDING: `realised_return` is barrier geometry, not trade quality

Flagged because it directly affects Stage D's primary metric.

Verified on all 77,751 barrier-resolved BTC rows, to floating-point precision (max abs error 1.2e-16):
conditional on `barrier_touched`, `realised_return` is EXACTLY `mult * ATR_entry / close_entry`. Every `tp` row
is `+2*ATR/entry`; every `sl` row is `-1*ATR/entry`. Values differ between rows only because ATR/price differs,
**never because the trade went better or worse.**

This was caught by an I4 inspection, not by reasoning: the first script run printed "78,378 distinct values"
next to a docstring claiming the column was degenerate. The two were both right and both misleading -- the
column is not constant, it is a deterministic function of the entry bar carrying zero path information. The
first attempt to verify this also failed for an unrelated reason worth recording: `vol_atr_14` in the modeling
frame is PERCENTILE-RANKED, not raw; the raw series is `raw_atr` and only present when
`build_asset_modeling_frame(..., keep_raw_atr=True)`.

Consequence for Stage D: BUILD_PLAN_v2 D.3 makes **top-decile realised EV the PRIMARY metric** and Gate D
requires it > 0. Computed against these labels, that metric would rank events by their entry-bar ATR/price
ratio -- i.e. by volatility exposure -- not by model skill. Before Gate D can mean anything, `realised_return`
needs to respond to the path (priced against actual fills, or against the exit engine's real exit). **Recorded
now rather than at Stage D, because Stage D's gate is unreadable without it.**

Sanity check consistent with S1: overall mean realised_return is -0.000101 (BTC), i.e. ~0. With 66.1% at
-ATR/entry and 33.1% at +2*ATR/entry, the geometry cancels to zero by construction.

### Also noted, not changed

The barrier-scan window is `highs[i+1 : i+max_candles]` -- bars i+1 through i+max_candles-1, which is
`max_candles - 1` bars -- while the vertical barrier resolves at bar `i+max_candles`. **Bar i+max_candles is
checked for neither TP nor SL touch**, but is used as the vertical exit. A TP-touching wick on exactly that bar
is recorded as a vertical resolution at the close instead. Affects at most the 0.8% of labels that reach the
vertical barrier. **Not changed** -- it would alter every existing label and therefore every recorded
measurement in this file. Flagged for an explicit decision at A.6, when labels are regenerated anyway.

### Gate A status: NOT REACHED

| task | status |
|---|---|
| A.1 Label metadata (t0, t1, barrier_touched, realised_return) | **DONE** |
| A.2 Sample uniqueness / `n_eff` in training weights | not started (measured as preview only) |
| A.3 Purged K-Fold + embargo | not started |
| A.4 Harness defect fixes (S11/S12/I3) | not started |
| A.5 `trials.jsonl` + DSR | not started -- file still absent |
| A.6 Recompute Stage 5 ceiling under corrected apparatus | not started |

Gate A additionally requires a synthetic leakage test (inject a deliberately leaked feature; confirm the old
splitter shows inflated AUC and the new one does not). Not built.

### Next Steps

1. **A.2 -- uniqueness weighting.** The measurement above is the input: mean uniqueness 0.144, so training
   weights will down-weight the average sample ~7x. Report `n_eff` alongside nominal n everywhere (I6).
2. **A.3 -- Purged K-Fold + embargo**, now unblocked: `t1` exists, so the purge predicate
   (`[t0_i, t1_i]` overlaps the test fold) is finally expressible. Embargo h = 0.01 x total observations.
3. **Decide the `realised_return` pricing question** before Stage D, not at it.
4. **Decide the vertical-barrier off-by-one** at A.6, when labels are regenerated.


## Session Update: Stage A.2 -- Sample Uniqueness and Effective n (2026-08-14)

Stage A, task A.2, gating on **Gate A**. Suite **284 -> 302 passed**. Nothing committed.

### THE NUMBERS

BTC/USDT 1h, full labelled dataset, `max_candles=48`, compression at the live Layer 6 threshold
(ATR pct <= 0.20, per S6):

| set | nominal n | n_eff | ratio | mean c_t |
|---|---|---|---|---|
| **FULL dataset** | **78,378** | **11,305.6** | **0.1442** | 8.48 |
| **COMPRESSED, own concurrency** | **20,201** | **3,639.9** | **0.1802** | 6.72 |
| COMPRESSED, in-situ (full-data concurrency) | 20,201 | 2,286.4 | 0.1132 | -- |

The two compressed rows answer different questions and are both reported because choosing one silently would
hide a methodological decision:

- **own concurrency** -- overlap among the compressed labels themselves. What an estimator trained ONLY on this
  subset actually sees. This is the figure to use when the compressed set is the training set.
- **in-situ** -- each compressed label's uniqueness within the full label set, counting the non-compressed
  neighbours it overlaps. Lower, because those neighbours are real overlap even when excluded from training.

The compressed subset's own-concurrency ratio (0.1802) is higher than the full dataset's (0.1442): compressed
bars are lower-volatility, so barriers set at multiples of a smaller ATR are still hit at a similar rate, while
removing the non-compressed labels removes overlap partners. Mean concurrency drops 8.48 -> 6.72 accordingly.

Per the instruction, no interpretation of what these imply is offered here. Implications are A.6's.

### Implementation

New module `src/predictive_core/uniqueness.py`, implementing AFML ch. 4 exactly as specified:

    indicator 1[t,i] -> concurrency c_t -> uniqueness u[t,i] -> avg u_bar_i -> n_eff = sum u_bar_i

`average_uniqueness()` returns a `UniquenessResult` carrying `average_uniqueness`, `nominal_n`, `effective_n`,
`ratio`, `max_concurrency`, `mean_concurrency`. `uniqueness_sample_weights()` returns weights normalised to
mean 1.0 (relative weighting unchanged; absolute scale kept so `min_child_samples`/`min_child_weight` stay
comparable with an unweighted baseline) plus the unnormalised result, so n_eff is never inflated by
normalisation.

**Not built as the literal indicator matrix.** At 78,378 labels that matrix is ~6.1e9 cells, which is why
AFML's pandas reference implementation does not scale here. Uses a difference array plus prefix sums --
exactly equivalent, O(n log n), dominated by the sort.

### Two correctness properties, both tested

1. **Uniqueness is per-asset.** A BTC label and an ETH label spanning the same hour do not read the same
   information. `average_uniqueness(..., groups=asset)` computes concurrency independently within each asset.
   Omitting `groups` on a pooled frame silently halves every uniqueness value --
   `test_labels_on_different_assets_do_not_dilute_each_other` asserts both the grouped (u_bar=1.0, n_eff=2.0)
   and ungrouped (u_bar=0.5, n_eff=1.0) results so the difference is pinned, and
   `test_grouping_still_penalises_overlap_within_one_asset` confirms grouping is not a blanket exemption.

   This is the same failure mode as the cross-asset bug in the since-deleted k-NN `_deduplicate`
   (project_state.md, 2026-08-12), where row adjacency in a pooled frame was mistaken for temporal adjacency
   within one instrument.

2. **Uniqueness is a property of the sample SET, not of the market.** Subset the labels and the survivors
   become more unique, because the labels they overlapped with are gone. `restrict_to=` makes that choice
   explicit rather than implied by whichever frame was passed. This is exactly the 0.1802 vs 0.1132 gap above.

### Tests (18 new, in `tests/test_uniqueness.py`)

Checked against hand-derived arithmetic rather than against the implementation:

- disjoint windows -> u_bar = 1.0, n_eff = nominal exactly
- k coincident labels -> u_bar = 1/k, n_eff = 1.0, for k in {2,3,5,9}
- partial overlap (A spans bars 0-2, B spans 1-3, c = [1,2,2,1]) -> both u_bar = 2/3, n_eff = 4/3, derived
  longhand in the test docstring
- randomised property sweep: 0 < u_bar <= 1 and 0 < n_eff <= nominal across 25 random configurations
- rejects t1 < t0, rejects mismatched lengths, empty input returns n_eff 0.0 not NaN

### I6 wiring

`QuantileFoldResult` gained `n_eff_train`, `n_eff_val`, `n_eff_test`, `uniqueness_weighted`, and
`n_eff_ratio_train()`. Every result object now carries effective n alongside nominal n.

`train_quantile_regressor(use_uniqueness_weights=True)` (default on) computes u_bar from the TRAIN partition's
`t0`/`t1` only -- never val, never test -- grouped per asset, and passes it as LightGBM's `weight=` on the
training Dataset in both the Optuna objective and the final fit. The validation Dataset stays unweighted, so
early stopping and the Optuna score judge the model on the raw val distribution; weighting changes what the
loss optimises against, not how the result is evaluated. Same isolation discipline the module already applied.

**Honest degradation, verified:** on a frame without `t0`/`t1`, the result reports
`uniqueness_weighted=False` and `n_eff_* = None` -- never a fabricated n_eff equal to nominal. An absent
measurement is not a measurement of no overlap (I5). Asserted in
`test_missing_label_metadata_reports_none_not_a_fabricated_n_eff`.

Verified end-to-end on a pooled 2-asset synthetic frame: `uniqueness_weighted=True`, n_eff reported for all
three partitions and strictly below nominal in each (1000 -> 368.3, 300 -> 112.3, 497 -> 192.6).

Tool: `scripts/measure_effective_n.py` (kept).

### Gate A status: NOT REACHED

| task | status |
|---|---|
| A.1 Label metadata | **DONE** |
| A.2 Sample uniqueness, u_bar as training weight, n_eff in results | **DONE** |
| A.3 Purged K-Fold + embargo | not started |
| A.4 Harness defect fixes (S11/S12/I3) | not started |
| A.5 `trials.jsonl` + DSR | not started -- file still absent |
| A.6 Recompute Stage 5 ceiling under corrected apparatus | not started |

Gate A also requires the synthetic leakage test (inject a leaked feature; old splitter shows inflated AUC, new
one does not). Not built -- it belongs with A.3.

### Next Steps

1. **A.3 -- Purged K-Fold + embargo.** Unblocked since A.1. Purge training samples whose `[t0_i, t1_i]`
   overlaps the test fold; embargo h = 0.01 x total observations. Replace `WalkForwardSplitter`'s splitting
   logic, keep its per-fold reporting structure.
2. **Build the Gate A synthetic leakage test alongside A.3**, not after it.
3. Carried forward: `realised_return` pricing question (Stage D's primary metric depends on it), the
   vertical-barrier off-by-one (decide at A.6), `trials.jsonl` absent, `config.yaml`'s `backtesting:` section
   unread.


## Session Update: Stage A.3 -- Purged K-Fold + Embargo, and the Gate A Leakage Test (2026-08-14)

Stage A, task A.3, gating on **Gate A**. Suite **302 -> 315 passed**. Nothing committed.

### THE LEAKAGE TEST -- the number Gate A asks for

Construction: the label is an i.i.d. coin flip at the bar where it resolves, and the feature identifies WHICH bar
resolves it. Two samples resolving on the same bar therefore carry the same label. **No genuine signal exists**,
so achievable AUC without leakage is 0.5000 exactly -- for this realisation, not merely in expectation.

Geometry: n=8,000, max horizon 600 bars, K=40 folds (200-bar test blocks), embargo 1%.

**Leak channel** -- test samples sharing a resolution bar with a training sample:

| fold | UNPURGED | PURGED | of n_test |
|---|---|---|---|
| 0 | 63 | **0** | 200 |
| 1 | 93 | **0** | 200 |
| 2 | 102 | **0** | 200 |
| ... | ~30-50% | **0** | 200 |

**Test AUC** (truth = 0.5000):

| | mean AUC | inflation over truth |
|---|---|---|
| **UNPURGED** | **0.5184** | **+0.0184** |
| **PURGED** | **0.5021** | **+0.0021** |

Purging removes ~89% of the fabricated skill and closes the leak channel completely. Training data retained
after purge+embargo: 88-92% per fold.

Encoded as two permanent tests in `tests/test_purged_kfold.py`:
`test_gate_a_purging_closes_the_leak_channel_completely` (deterministic -- counts the channel, no model, no
thresholds) and `test_gate_a_unpurged_splitter_reports_skill_that_does_not_exist` (the statistical half).

### THREE THINGS THAT WENT WRONG BUILDING THIS, ALL FOUND BY INSPECTION

Recorded because each looked convincing before it was checked.

**1. The first leak construction was confounded and its result was meaningless.** It used a random-walk price
level as the feature and the forward move as the label, on the reasoning that independent increments make true
AUC exactly 0.5. That is true in expectation over paths, NOT for the single realised path a model actually sees
-- a realised walk contains genuine level/return structure (a V-shaped stretch makes low levels precede rises).
The tell was fold 0 showing AUC 0.72 under BOTH splitters, i.e. purging changing nothing. Had only the mean been
read, "0.5552 vs 0.5172" would have looked like a clean pass. Replaced with the i.i.d.-coin construction, where
0.5 holds for the specific realisation.

**2. The first implementation was walk-forward, and A.3's embargo clause is dead code in that shape.** A.3
specifies dropping training samples with `t0 in [T1, T1 + h]` -- AFTER the held-out block. In a pure
walk-forward layout no training sample is ever after the test block, so that clause can never fire. Every fold
reported `embargoed 0`, which is what surfaced it. Training data exists on both sides of a block only in
K-fold, which is what AFML ch. 7 specifies and what the embargo is designed for. Rebuilt as `PurgedKFold`.

**3. My own purge accounting was wrong and overstated the damage.** `n_train_before` was computed as
`len(rest) + n_purged + n_embargoed`, but `rest` was already the pre-purge candidate set -- a double-count. It
reported "train 8,441" on an 8,000-row frame, which is impossible on its face. Retention was understated as
84.8% when it is 91.8%. Fixed, and a balance assertion (`len(train) == n_before - purged - embargoed`) now
raises rather than printing an impossible number.

### FINDING: the old splitter's real leak is into VALIDATION, not test

Measured on its own geometry (train 2000 / val 400 / test 400):

| fold | train->val shared | train->test shared |
|---|---|---|
| 0 | 74 | 4 |
| 1 | 78 | 1 |
| 2 | 70 | 1 |
| ... | 70-101 | 1-6 |

`WalkForwardSplitter` purges **neither**. train->test stays small only because the 400-bar val block partly
buffers it -- an accident of window sizing, not a property of the splitter, and it stops holding as soon as the
horizon exceeds the val block. train->val is large regardless.

That is not a lesser problem. Validation drives early stopping and Optuna hyperparameter selection, so the old
splitter contaminates the partition that CHOOSES the model, then reports that model's score on test as though
it were clean. Both leaks are closed by the replacement, which purges train against val AND test.

### The same measurement at the project's real geometry

| | value |
|---|---|
| unpurged leak channel | **0.06%** of each test fold |
| purged leak channel | **0.00%** |
| training retained after purge+embargo | **97.9%** |

At horizon 48 against ~15,700-bar folds the channel touches only fold edges. **Narrow is not zero, and it is not
bounded** -- it widens with horizon and with fold count, and the cost of closing it is 2.1% of training data.

### Implementation

`src/predictive_core/purged_kfold.py`:

- `purge_and_embargo()` -- free function. PURGE `t0_i <= B1 AND t1_i >= B0`; EMBARGO `B1 < t0_i <= B1 + h`.
  Embargo bound resolved against the frame's own BAR AXIS, not wall-clock arithmetic, because exchange outages
  leave real gaps and `B1 + h hours` can land in a hole.
- `PurgedKFold(n_splits, embargo_pct=0.01, purge=True, val_size=0)` -- contiguous time blocks, training drawn
  from BOTH sides, optional validation carved from the training side and itself treated as a held-out block
  that training is purged against.
- `PurgedFold` -- per-fold reporting detail A.3 asks to keep: `n_train_before`, `n_purged`, `n_embargoed`,
  `train_retention`, `summary()`.
- Refuses to split a frame lacking `t0`/`t1` rather than silently skipping purging -- a splitter that skipped
  silently would be indistinguishable from the one it replaces, which is how the leak survived two phases.
- `_assert_no_overlap` checks LABEL WINDOWS, not t0 ordering -- the property that actually prevents the leak,
  asserted directly rather than inferred from construction.

`WalkForwardSplitter` marked SUPERSEDED in its docstring, retained solely as the control arm in the leak
comparison, with an explicit "do not use it to produce results".

13 new tests, including hand-checkable purge/embargo semantics, training-on-both-sides, embargo actually
firing, accounting balance, and every row tested exactly once.

Tool: `scripts/leakage_test.py` (kept).

### Gate A status: NOT REACHED -- 3 of 6, but the leakage requirement is MET

| requirement | status |
|---|---|
| A.1 Label metadata | **DONE** |
| A.2 Uniqueness weighting, n_eff in results | **DONE** |
| A.3 Purged K-Fold + embargo | **DONE** |
| **Gate A: synthetic leakage test verified** | **DONE** -- 0.5184 -> 0.5021, channel closed |
| A.4 Harness defect fixes (S11/S12/I3) | not started |
| A.5 `trials.jsonl` + DSR | not started -- file still absent |
| A.6 Recompute Stage 5 ceiling under corrected apparatus | not started |
| Gate A: harness fixes verified by reproducing fold 2 of `all_layers` | not started (needs A.4) |
| Gate A: `n_eff` reported everywhere, corrected CIs published | partial -- n_eff wired, CIs are A.6 |

### Next Steps

1. **A.4 -- harness defect fixes.** Bankruptcy floor on `equity_curve_from_trade_pnls` (S11), single accounting
   book (S12), gap-cost cap at position notional, ablation flagging on maxDD/PF before Sharpe (I3).
2. **A.5 -- `trials.jsonl` + DSR.** Still absent; every configuration evaluated to date is unlogged.
3. **A.6 -- recompute the ceiling** under purged CV + uniqueness weighting, and publish corrected CIs.
4. **Wire `PurgedKFold` into `WalkForwardTrainer`**, which still drives the superseded `WalkForwardSplitter`.
   Deliberately not done in this task: it changes what every training run measures, and belongs with A.6 where
   the recomputation happens.
5. Carried forward: `realised_return` pricing (Stage D's primary metric), vertical-barrier off-by-one (A.6),
   `config.yaml`'s unread `backtesting:` section.


## Session Update: Stage A.5 -- Trial Registry and Deflated Sharpe Ratio (2026-08-14)

Stage A, task A.5, gating on **Gate A**. Suite **315 -> 338 passed**. Nothing committed.
**`trials.jsonl` now exists** and is not gitignored (verified) -- it must travel with the repo.

### THE TRIAL COUNT

**Documented lower bound on distinct configurations already evaluated: 2,024.**

Labelled an underestimate, because it is one.

| source | count |
|---|---|
| Optuna trials, documented runs | 1,840 |
| Non-Optuna configurations, documented | 174 |
| Ablation variants reconstructed as full records | 10 |
| **TOTAL (lower bound)** | **2,024** |

Optuna breakdown:

| run | studies | trials |
|---|---|---|
| Stage 1 walk-forward (2-trial sanity check) | 20 | 40 |
| Stage 2 walk-forward (full 15-trial budget) | 20 | 300 |
| Stage 3 walk-forward (class-weighted re-run) | 20 | 300 |
| Layer ablation v1 (2 distinct fits) | 40 | 600 |
| Layer ablation v2 (2 distinct fits) | 40 | 600 |

### TWO COUNTING ASSUMPTIONS THAT WERE WRONG ON THE FIRST PASS

The first computation returned **6,554**. It was inflated ~3.2x by two unchecked assumptions. Both were caught
by checking the evidence rather than reasoning from the plan.

1. **Studies per fold: assumed 7, actually 4.** The guess was direction(1) + magnitude(3 quantiles) +
   duration(3). But `Phase2Strategy.fit()` -- read before deletion, and `PredictiveCoreStrategy` was identical
   here -- trains direction ONCE and `train_quantile_regressor` ONCE for `target_col='magnitude'` only. Duration
   was never trained by the strategies. 1 + 3 = 4.

2. **Ablation variants: assumed 5 full fits each, actually 2.** The wall-clock timings in the logs settle it:
   v1 ran 2203.2s, 458.3s, then 8.9s / 13.7s / 4.0s; v2 ran 3302.3s, 654.9s, then 10.2s / 21.0s / 7.8s. A
   sub-15-second variant cannot have retrained anything -- the runner reused an already-fitted model. Counting
   all five as full fits inflated the bound 2.5x on this line alone.

An inflated figure presented as a lower bound is not a conservative error. It would overstate the deflation
term and make DSR look more rigorous than the evidence supports.

### WHAT IS AND IS NOT RECOVERABLE

**Reconstructed with full per-configuration detail (10 records now in `trials.jsonl`):** the 5 variants of each
ablation run, from `reports/layer_ablation.json` and `layer_ablation_v2.json` -- real metrics, per-fold Sharpes,
equity_destroyed flags. Written with `source='backfill'` and a `note` citing the file, so reconstructed entries
are distinguishable from live ones. `n_eff` is `null` on all ten: uniqueness did not exist when they ran and the
labelled frame was not persisted. Null, not backfilled with nominal n (I5).

**Counted only, not reconstructable:** the Stage 1-3 walk-forward runs and the signal-ceiling measurements are
described in `project_state.md` with their metrics, but their individual configurations are gone.

**Unrecoverable entirely, and this is the important one:**
- **Every individual Optuna trial's hyperparameters.** `mlruns/` contains only `.gitkeep` -- MLflow was
  configured and never written to. No Optuna storage was persisted; every study was built in-memory per call
  and discarded. For each of the ~1,840 documented trials, the winning parameters are gone along with the 14
  losing ones.
- Any run whose output was neither written to `reports/` nor narrated in `project_state.md`. Unknown and
  unbounded.

**The bound also overcounts in one direction, stated rather than netted away:** identical Optuna configurations
re-proposed across folds are counted separately here, whereas the live registry hashes them and collapses
duplicates. Both directions are reported; they are not combined into a single "adjusted" figure, because that
would present a reconstruction as a measurement.

### Implementation

**`src/predictive_core/trial_registry.py`** -- append-only JSONL, one object per line:
`{config_hash, timestamp, stage, params, n_eff, result_metrics, nominal_n, source, note}`.

- `config_hash` is SHA-256 over a canonicalised payload: sorted keys, floats normalised, so cosmetic
  differences (key order, `5` vs `5.0`) do not inflate the count. `stage` participates in the hash -- the same
  hyperparameters at Stage C and Stage D are two searches of the hypothesis space.
- `distinct_configuration_count()` is what feeds DSR: distinct, not total lines, since re-evaluating an
  identical configuration does not widen the space searched.
- **No update function and no delete function, deliberately.** Corrections are appended as superseding records.
  A count that can be silently reduced is worse than no count, because DSR computed from it looks rigorous
  while overstating significance.
- Appends are flushed and `fsync`ed: a trial that ran but went unrecorded biases the registry in the one
  direction that flatters DSR.
- A corrupt trailing line (crash mid-append) costs one record, not the file -- tested.
- `optuna_logging_callback()` logs **every** trial, not the best one. A search that tried 15 and kept 1 has
  searched 15; logging only the winner is exactly how DSR gets flattered.

Wired into `train_quantile_regressor(trial_registry_stage=...)`. Default `None` (no logging) so tests and
exploratory calls do not pollute the registry; any run whose numbers are reported must set it.

**`src/predictive_core/dsr.py`** -- Bailey & Lopez de Prado:

    SR*  = sqrt(V) * [(1-gamma)*Phi^-1(1 - 1/N) + gamma*Phi^-1(1 - 1/(N*e))]
    DSR  = PSR(SR*) = Phi[(SR - SR*)*sqrt(T-1) / sqrt(1 - g3*SR + ((g4-1)/4)*SR^2)]

`DSRResult` carries every input that determined the answer, so a reported DSR can be audited without re-running
it, plus `passes_gate` (> 0.5).

**The part that is easy to fudge, and is refused:** `V`, the cross-trial variance of Sharpe ratios, is not
derivable from one strategy's returns and is not optional -- SR* scales with sqrt(V). Passing
`var_trial_sharpes=None` **raises**. The caller must supply a float or explicitly opt in to the zero-skill
approximation `V ~ 1/T` via the literal string `"null"`, and `DSRResult.var_source` records which was used. A
DSR computed from an invented V looks rigorous and is not.

No annualisation anywhere; SR is per-observation, and the docstring warns that passing an annualised Sharpe with
a per-observation T inflates DSR by ~sqrt(periods_per_year).

23 new tests, including: SR* matches the closed form; SR* grows with N; PSR against itself is exactly 0.5; PSR
falls under negative skew and fat tails; **DSR falls monotonically as the trial count rises**; a zero-mean
return series selected from 2,000 trials fails the gate; and DSR refuses to guess V.

Tools kept: `scripts/backfill_trials.py`.

### What this means for the DSR gate, factually

Gate E requires DSR > 0.5 against the full `trials.jsonl` count. That count starts at a **documented lower bound
of 2,024 before Stage C has run a single new configuration**. At N=2,024 the deflation term is already
substantial, and it only grows. No interpretation beyond that here -- Gate E is far off and the returns series
it would be applied to does not exist yet.

### Gate A status: NOT REACHED -- 4 of 6

| task | status |
|---|---|
| A.1 Label metadata | **DONE** |
| A.2 Uniqueness weighting, n_eff in results | **DONE** |
| A.3 Purged K-Fold + embargo | **DONE** |
| Gate A: synthetic leakage test | **DONE** (0.5184 -> 0.5021) |
| A.5 `trials.jsonl` + DSR | **DONE** |
| A.4 Harness defect fixes (S11/S12/I3) | **not started** |
| A.6 Recompute Stage 5 ceiling under corrected apparatus | **not started** |
| Gate A: fold-2 harness reproduction, sizing/reported maxDD agree | not started (needs A.4) |
| Gate A: corrected CIs published | not started (A.6) |

### Next Steps

1. **A.4 -- the last unstarted apparatus task.** Bankruptcy floor on `equity_curve_from_trade_pnls` (S11),
   single accounting book (S12, 20x divergence), gap-cost cap at position notional, ablation flagging on
   maxDD/PF before Sharpe (I3). Gate A's fold-2 reproduction check depends on it.
2. **A.6 -- recompute the ceiling** under purged CV + uniqueness weighting, publish corrected CIs, and log every
   configuration to the registry via `trial_registry_stage`.
3. **Wire `PurgedKFold` into `WalkForwardTrainer`** (still on the superseded splitter) -- belongs with A.6.
4. Carried forward: `realised_return` pricing (Stage D's primary metric), vertical-barrier off-by-one (A.6),
   `config.yaml`'s unread `backtesting:` section.


## Session Update: Stage A.4 -- Harness Defect Fixes (2026-08-14)

Stage A, task A.4, gating on **Gate A**. Suite **338 -> 348 passed**. Nothing committed.

### Fix 3 was ALREADY DONE -- verified, not redone

The gap-cost fixes landed in the 2026-08-14 gap-cost session and are present in `costs.py`:
per-SESSION counting replacing per-bar linearity (lines 339-346), an adverse-fraction correction
(`DEFAULT_GAP_ADVERSE_FRACTION = 0.5`) for charging a symmetric move as a one-sided cost, and the notional cap
at line 369 (`MAX_GAP_COST_FRAC_OF_NOTIONAL = 1.0`). Re-implementing would have been duplicated work; the
brief's description of the defect matches the pre-fix state, not the current one.

**This matters for interpreting fix 2's verification** -- see below.

### Fix 1 -- bankruptcy floor (S11)

`equity_curve_from_trade_pnls` returned `init_cash + cumsum()` with nothing bounding it below. The
`equity_destroyed` FLAG existed and was set, but the CURVE was never floored, so maxDD could exceed 1.0 --
a drawdown above 100% describes losing more than everything and continuing to trade.

**Ruin is now absorbing**, not merely clipped: the curve is pinned at 0 from the first crossing onward. A
pointwise clip would let a later cumsum recovery lift the curve back off the floor and understate the damage.
Post-fix maxDD <= 1.0 by construction, matching section 7's canonical definition.

`floor_at_zero=False` retained solely so a test can still produce the pre-fix behaviour and assert the fix
changed something.

**An existing test failed on this fix, correctly.** `test_ruin_detection_disqualifies_sharpe_and_is_reported`
asserted `max_drawdown(eq) > 1.0` as "the signature of negative equity" -- exactly the impossible value the
floor removes. Rewritten as `test_ruin_is_floored_at_zero_and_maxdd_cannot_exceed_one`, now asserting maxDD ==
1.0 exactly, equity never negative, ruin absorbing, and the unfloored path still producing the impossible value
so a regression fails loudly.

### Fix 2 -- one book (S12 / I7)

The harness ran vectorbt once with fees+slippage, extracted trades, subtracted gap cost afterwards, and built a
SECOND equity curve from the net PnL. `size_type='targetpercent'` sizes every order as a percentage of
vectorbt's internal equity -- so sizing read a book that never saw gap cost, while results were computed from a
different one.

Fixed with a two-pass run: pass 1 discovers the trades (gap cost depends on holding period, unknowable before
they exist), pass 2 folds the per-trade gap charge into the fee array vectorbt already consumes, as an
exit-bar fee fraction. `pf.value()` is then the single equity curve, and no cost is applied outside it.

Charged at exit rather than entry, deliberately, and this is the conservative choice: the capital reduction
lands at the end of the hold rather than being spread across it, so intra-trade equity is not flattered.

Not iterated to a fixed point -- charging gap cost lowers equity, which lowers targetpercent order sizes, which
slightly lowers the charge. The residual is small and is reported rather than assumed away.

### VERIFICATION -- what could and could not be done

**The literal check Gate A asks for is impossible, and this is not a shortcut.** `all_layers` was produced by
`Phase2Strategy` driven by `scripts/run_layer_ablation.py`, using `HistoricalAnalogEngine` and
`EnsembleMetaModel`. All four were deleted at Stage 0 on operator instruction.
`reports/layer_ablation_v2.json` retains fold 2's Sharpe (-0.162) and trade count (468) but no positions, no
trades, and no equity curve -- nothing sufficient to rebuild it.

What was verified instead: the DEFECT itself, on real BTC data at fold 2's geometry (2y train / 6m val / 6m
test from 2022-08-01, test window 2025-08-01 -> 2026-02-01, 4,417 bars), using a deterministic
moving-average rule so the run reproduces exactly. S12 is a property of the harness's accounting, not of the
strategy that fed it.

**BEFORE -- two books:**

| | value |
|---|---|
| sizing book maxDD (what vectorbt sized off) | 0.008747 |
| reported book maxDD (what was published) | 0.008140 |
| final equity, sizing book | $9,970.53 |
| final equity, reported book | $9,966.07 |
| difference | **$4.46 = exactly the gap cost** |

**AFTER -- one book:**

| | value |
|---|---|
| single book maxDD (sized off AND reported) | 0.008870 |
| final equity | $9,966.16 |

Sizing and reporting now read the same series, so the two numbers are one number by construction. There is no
second book left to diverge from.

**THE DIVERGENCE HERE IS 0.93x, NOT THE 20x RECORDED AS S12, AND THAT DIFFERENCE IS REAL RATHER THAN A
MEASUREMENT ARTEFACT.** S12's 20x was measured against the ORIGINAL unbounded per-bar gap model, where gap cost
was 96% of fold 2's loss. That model was already replaced (fix 3, above), so the gap cost that drove the
divergence is now small and bounded. The structural violation was still present and is now fixed; its magnitude
at current parameters is simply much smaller.

Demonstrated rather than argued, by sweeping gap-cost magnitude:

| adverse frac | gap cost | sizing maxDD | reported maxDD | divergence |
|---|---|---|---|---|
| 0.5 (current) | $4.46 | 0.008747 | 0.008140 | 0.93x |
| 2.0 | $17.82 | 0.008747 | 0.008454 | 0.97x |
| 10.0 | $89.11 | 0.008747 | 0.014101 | 1.61x |
| 40.0 | $356.45 | 0.008747 | 0.039987 | 4.57x |

**The sizing-book maxDD is IDENTICAL -- 0.008747 -- at every level, from $4.46 of gap cost to $356.45.** That
single column is S12 stated exactly: the book that sized the orders could not see gap cost at all, no matter how
large it got. The reported book moves; the sizing book does not.

Tool: `scripts/verify_single_book.py` (kept).

### Fix 4 -- ablation flagging (I3)

The old flagging lived in `scripts/run_layer_ablation.py`, deleted at Stage 0. Rebuilding it inside the next
runner would rebuild the defect, so the rule now lives in `src/backtesting/ablation_flagging.py`, tested, and
whatever Stage E runner appears must call it.

`rank_variants()` **refuses to rank at all** when any variant destroyed equity -- stronger than excluding the
insolvent ones, because an ablation compares variants against each other and a comparison set containing a
destroyed account has no common scale. Solvent sets rank on maxDD, then profit factor, then Sharpe.
`flag_degrading_layers()` consults Sharpe ONLY when both the full-stack and leave-one-out runs are solvent.

A NaN maxDD counts as insolvent: an unmeasurable drawdown is not a good drawdown.

**Regression test replays the real 2026-08-14 ablation numbers and reaches the opposite verdict to the old
rule.** The old Sharpe-only rule flagged Layer 6 (removing it raised Sharpe -8.284 -> -4.846) and reported
Layer 7C as the biggest HELP (+4.075 Sharpe). Both runs had maxDD above 1.0, so both Sharpes were arithmetic on
a destroyed account. The new rule refuses the ranking, flags 7C on solvency/maxDD/PF (1.023 -> 0.056 and
0.014 -> 0.238 when removed), and never consults Sharpe for either layer because no comparison was admissible.

### Gate A status: ALL SIX TASKS DONE -- gate NOT self-certified

| task | status |
|---|---|
| A.1 Label metadata | **DONE** |
| A.2 Uniqueness weighting, n_eff in results | **DONE** |
| A.3 Purged K-Fold + embargo | **DONE** |
| A.4 Harness defect fixes | **DONE** (fix 3 was already done; verified) |
| A.5 `trials.jsonl` + DSR | **DONE** |
| A.6 Recompute Stage 5 ceiling under corrected apparatus | **NOT STARTED** |

Gate A's explicit acceptance conditions:

| condition | status |
|---|---|
| Purged CV verified by synthetic leakage test | **MET** -- 0.5184 -> 0.5021, channel closed |
| `n_eff` reported everywhere | **MET** for the trainer; A.6 must carry it into the ceiling results |
| Corrected CIs published | **NOT MET** -- A.6 |
| Harness fixes verified by reproducing fold 2 of `all_layers` | **NOT LITERALLY POSSIBLE** -- see above. Mechanism verified on real data at that fold's geometry instead |
| Test suite green | **MET** -- 348 passed |

**Gate A is not self-certified by the agent (section 10).** The fold-2 condition in particular needs an
operator ruling: the literal artifact cannot be regenerated because Stage 0 deleted the strategy that made it,
and I am not going to record a substitute as though it satisfied the original.

### Next Steps

1. **Operator ruling on Gate A**, specifically whether the mechanism-level verification of the harness fixes
   substitutes acceptably for reproducing fold 2 of a deleted strategy.
2. **A.6 -- the last apparatus task.** Recompute the unconditioned and conditioned ceiling measurements under
   purged CV + uniqueness weighting; publish corrected CIs; log every configuration to `trials.jsonl` via
   `trial_registry_stage`. Expect the compressed-regime CI to widen ~2.6x (measured n_eff ratio 0.1442, not the
   3-5% the plan anticipated) and quite possibly to straddle 0.5 -- BUILD_PLAN_v2 A.6 states in advance that
   this is the honest outcome, not a failure.
3. **Wire `PurgedKFold` into `WalkForwardTrainer`** (still on the superseded splitter) -- belongs with A.6.
4. Carried forward: `realised_return` pricing (Stage D's primary metric is unreadable without it), the
   vertical-barrier off-by-one, `config.yaml`'s unread `backtesting:` section.


## Session Update: Stage A.6 -- Corrected Ceiling Measurement (2026-08-14)

Stage A, task A.6, the final apparatus task. Suite **348 -> 349 passed**. Nothing committed.
All 5 configurations logged to `trials.jsonl` (registry now 15 records, 5 with n_eff).

### THE CORRECTED TABLE

BTC/USDT 1h, 78,226 labelled rows (neutral excluded), 39 features. Purged K-Fold K=5, 1% embargo,
uniqueness-weighted training, CI bootstrap drawing n_eff observations rather than n.

| subset | nominal n | n_eff | ratio | AUC | corrected 95% CI | LL vs base | rounds |
|---|---|---|---|---|---|---|---|
| ALL bars (unconditioned) | 78,226 | 11,334 | 0.145 | 0.5244 | **[0.5134, 0.5358]** | -0.00036 | 20.8 |
| Compressed ATR pct < 0.30 (old Layer 6) | 27,780 | 4,857 | 0.175 | 0.5139 | **[0.4973, 0.5305]** | -0.00264 | 16.8 |
| **Compressed ATR pct <= 0.20 (LIVE Layer 6)** | 20,198 | 3,641 | 0.180 | **0.5183** | **[0.4986, 0.5374]** | -0.00106 | 6.8 |
| Expanding 0.30-0.70 | 27,757 | 6,053 | 0.218 | 0.5156 | [0.5006, 0.5313] | -0.00126 | 14.6 |
| Extended ATR pct > 0.70 | 22,689 | 4,572 | 0.202 | 0.5095 | [0.4929, 0.5267] | -0.00217 | 18.2 |

CI widening, shown against the naive bootstrap so the correction is visible:

| subset | naive CI | naive width | corrected width | factor |
|---|---|---|---|---|
| ALL bars | [0.5202, 0.5287] | 0.0085 | 0.0224 | **2.65x** |
| Compressed < 0.30 | [0.5070, 0.5208] | 0.0138 | 0.0332 | 2.41x |
| Compressed <= 0.20 | [0.5097, 0.5267] | 0.0170 | 0.0389 | 2.29x |
| Expanding | [0.5085, 0.5228] | 0.0143 | 0.0307 | 2.15x |
| Extended | [0.5016, 0.5177] | 0.0161 | 0.0337 | 2.10x |

Widening of 2.1-2.65x, matching the sqrt(1/0.144) ~ 2.63 predicted at Stage A.2. The prediction was recorded in
advance and the measurement matched it.

### S5 SURVIVAL -- reported separately, as an INPUT TO STAGE C, not a verdict

**S5 does not survive the correction.** The prior finding was: "Signal is present under compression (ATR pct
<= 0.20, AUC 0.5390, CI [0.519, 0.559]) and absent under expansion/extension (CIs straddle 0.5)." That
statement rested on the compressed CI sitting entirely above 0.5 while the others did not.

Under corrected apparatus:

- **Compressed <= 0.20: AUC 0.5390 -> 0.5183, CI [0.519, 0.559] -> [0.4986, 0.5374]. The CI now INCLUDES 0.5.**
- Compressed < 0.30 also includes 0.5 ([0.4973, 0.5305]).
- Extended includes 0.5 ([0.4929, 0.5267]).
- **Expanding does NOT include 0.5** ([0.5006, 0.5313]) -- the one subset that previously straddled 0.5 and now
  marginally does not. Its lower bound clears 0.5 by 0.0006, which is not a finding, it is a boundary.
- Only the unconditioned set has a CI clearly above 0.5 ([0.5134, 0.5358]).

The gradient S5 described -- signal present in compression, absent in expansion -- **inverts** under the
corrected apparatus, and both directions sit inside their intervals. The defensible statement is now: no
volatility subset shows a ceiling distinguishable from the unconditioned one, and the compressed subset in
particular is no longer distinguishable from chance.

This was pre-registered as a likely outcome (BUILD_PLAN_v2 A.6: "This is the honest number, and it straddling
0.5 is a likely outcome, not a bug"). It is recorded and carried into Stage C per the plan's sequencing
rationale, which makes Stage C's flow-data test the decisive experiment precisely so that an S5 failure does not
end the project. **No termination decision follows from this result.**

Note also: boost rounds are 6.8-20.8 across every subset. S4 established that 2-32 rounds means nothing to fit.
That diagnostic is unchanged by the corrected apparatus.

### FINDING: uniqueness weighting shifts the class prior, and it is barrier geometry doing it

Every subset shows `LL vs base` NEGATIVE -- the model's logloss is worse than a constant base-rate predictor,
where Stage 5's unconditioned run had it POSITIVE (+0.00102). Inspected rather than reported as a result.

Measured directly, unconditioned set, same purged CV, weighting toggled:

| | AUC | logloss | vs baseline | mean prediction |
|---|---|---|---|---|
| uniqueness-weighted | 0.5244 | 0.63931 | **-0.00036** | 0.3138 |
| unweighted | 0.5214 | 0.63871 | **+0.00024** | 0.3359 |

- unweighted training base rate: **0.33688**
- uniqueness-WEIGHTED base rate: **0.31138** (a -2.55pp shift)

**Cause:** `sl` labels resolve in a median 4 bars, `tp` labels in 7 (measured at Stage A.1). Shorter windows
overlap less, so they score HIGHER uniqueness and take MORE weight. Since `sl` is the down class, weighting
systematically down-weights "up" and the model learns a prior 2.5pp below the one the test set carries. Every
subset shows the same signature (calibration gap -0.015 to -0.023).

**AUC is rank-based and immune to this** -- it actually rose slightly under weighting (0.5244 vs 0.5214). Only
logloss and calibration are affected.

Consequences, recorded rather than resolved here:

1. **`LL vs base` is not comparable across weighted and unweighted runs.** The Stage 5 -> A.6 change in that
   column is a calibration artifact, not a deterioration in what the model knows. Future logloss comparisons
   under uniqueness weighting need either a weighted baseline or a recalibration step. Not chosen here, because
   picking whichever variant looks better after seeing both numbers is exactly the failure mode this project
   guards against -- both are reported.
2. **Uniqueness weighting is not class-neutral under asymmetric barriers.** Any label whose duration correlates
   with its class will inherit this. It affects the Stage D meta-label model directly, whose label ("did this
   event reach target before stop") has exactly that structure.

### DEFECT FOUND AND FIXED IN MY OWN A.3 CODE

The first A.6 run scored 62,581 of 78,226 rows -- exactly 4/5. Cause: `PurgedKFold` carved its validation set
from the rows BEFORE the test block, but fold 0's block starts at row 0, so `before` was empty, val came back
empty, and callers skipped fold 0 entirely. Every out-of-fold estimate silently lost 1/K of the data, and the
earliest 20% of history was never scored.

Fixed to fall back to the rows AFTER the block, which is legitimate in K-fold where training data exists on both
sides and purge+embargo apply identically. Regression test
`test_first_fold_still_gets_a_validation_set` added. Full n restored to 78,226.

Surfaced only because the scored-row count was checked against nominal rather than assumed. It would not have
changed any conclusion below, but it would have quietly discarded 20% of the data in every future measurement.

### GATE A -- APPARATUS CRITERIA

Gate A tests apparatus correctness, not whether S5 survives.

| criterion | status |
|---|---|
| Purged CV verified by synthetic leakage test (old splitter inflated, new not) | **MET** -- 0.5184 -> 0.5021, leak channel closed to 0 |
| `n_eff` reported everywhere alongside nominal n | **MET** -- in `QuantileFoldResult`, in every A.6 row, in every `trials.jsonl` record |
| Corrected CIs published | **MET** -- table above, 2.1-2.65x wider, naive CIs shown alongside |
| Harness fixes verified by reproducing fold 2 of `all_layers` | **NOT LITERALLY POSSIBLE** -- `Phase2Strategy`, `run_layer_ablation.py`, `HistoricalAnalogEngine`, `EnsembleMetaModel` were all deleted at Stage 0. Mechanism verified instead on real BTC data at that fold's geometry: sizing-book maxDD held constant at 0.008747 across gap costs from $4.46 to $356.45 while the reported book moved, then reconciled to one book |
| Test suite green | **MET** -- 349 passed |

**Four of five met. The fifth is not refusable-or-passable by me** -- the artifact cannot be regenerated because
Stage 0 deleted the code that produced it, on instruction. **Gate A is not self-certified by the agent (section
10) and requires operator sign-off.**

### Next Steps

1. **Operator sign-off on Gate A**, including a ruling on the fold-2 criterion.
2. **Stage B and Stage C.** Per BUILD_PLAN_v2's sequencing rationale, Stage A was explicitly structured NOT to
   kill the project on an S5 failure, and Stage C's flow-data ceiling test is the decisive experiment. Stage B
   data acquisition is the long pole and has not started -- vendor evaluation was flagged as Day 1 work five
   sessions ago and remains untouched.
3. **Carry the S5 result into Stage C's interpretation**, per the plan.
4. **Decide the uniqueness/calibration interaction** before Stage D, since the meta-label model has the same
   duration/class correlation.
5. Carried forward: `realised_return` pricing (Stage D's primary metric), the vertical-barrier off-by-one (this
   was the point at which labels were regenerated, and it was NOT changed -- still open), `config.yaml`'s unread
   `backtesting:` section, `WalkForwardTrainer` still on the superseded splitter.


## Session Update: Stage B.1 -- Free-Source Ingestion (2026-08-14)

Stage B, task B.1, working toward **Gate B**. Suite **349 -> 361 passed**. Nothing committed.
All sources persisted to `data/raw/`.

### COVERAGE, GAPS, ROW COUNTS

| source | rows | start | end | outliers | gaps by class |
|---|---|---|---|---|---|
| BTC/USDT 1h (reference) | 78,666 | 2017-08-17 | 2026-08-13 | 71 | data_error=25, illiquid_period=3 |
| BTC/USDT 1d OHLCV | 3,284 | 2017-08-17 | 2026-08-13 | 271 | none |
| BTC/USDT 1w OHLCV | 469 | 2017-08-20 | 2026-08-09 | 274 | none |
| Funding rate 8h | 7,592 | 2019-09-10 | 2026-08-14 | 0 | none |
| Perp-spot basis 1h | 58,198 | 2019-12-23 | 2026-08-14 | 0 | data_error=13, illiquid_period=2 |
| FRED dxy_proxy (DTWEXBGS) | 5,164 | 2006-01-02 | 2026-08-07 | 0 | data_gap=8, market_closed=1,121 |
| FRED us10y (DGS10) | 16,138 | 1962-01-02 | 2026-08-12 | 0 | data_gap=4, market_closed=3,593 |
| FRED vix (VIXCLS) | 9,251 | 1990-01-02 | 2026-08-13 | 0 | data_gap=3, market_closed=1,987 |

### SOURCES SHORTER THAN THE BTC 1h WINDOW

| source | shortfall |
|---|---|
| **Funding rate 8h** | starts **754 days later** (2019-09-10 vs 2017-08-17) |
| **Perp-spot basis 1h** | starts **858 days later** (2019-12-23 vs 2017-08-17) |
| BTC/USDT 1w | starts 2 days later -- weekly bucketing, not a data limitation |

All three FRED series comfortably cover the reference window.

**This is the constraint MASTER_CONTEXT_v2 section 9 predicted, now measured.** Binance BTC perpetuals begin
2019-09, so every derivatives-flow series is capped at ~7 years against OHLCV's 9. Basis is bounded tighter
still (2019-12) because `markPriceKlines` history starts later than the funding series. **Any conditioned event
set built on flow features loses the 2017-2019 period, and Gate B's >= 200 independent events must be counted
over the flow window, not the OHLCV window.**

### THE DXY SUBSTITUTION -- flagged, not silently accepted

**DXY is not published by FRED.** It is a proprietary ICE index. `DTWEXBGS` (Fed Nominal Broad USD Index) is
ingested instead, and it is a DIFFERENT INDEX rather than a renamed one:

| | DXY | DTWEXBGS |
|---|---|---|
| basket | 6 currencies, ~58% EUR | 26 economies, trade-weighted |
| weights | fixed since 1973 | reweighted annually |
| frequency | intraday | daily only |

They correlate strongly but are not interchangeable, and DTWEXBGS's heavier emerging-market weighting behaves
differently in exactly the risk-off episodes Layer 3 exists to detect. Layer 3's "DXY rising" rule will be
evaluated against a proxy, and that must be stated wherever the result is reported.

Note also that `config.yaml` uses **UUP** (an ETF tracking DXY futures) via yfinance -- a THIRD series. Three
different dollar measures are now reachable in this repo. The choice needs to be made once and recorded, not
left to whichever fetcher a caller reaches for.

### FINDING: the outlier flag is uninformative on coarse bars

| source | outliers | rate |
|---|---|---|
| BTC/USDT 1h | 71 of 78,666 | 0.1% |
| BTC/USDT 1d | 271 of 3,284 | **8.3%** |
| BTC/USDT 1w | 274 of 469 | **58.4%** |

`flag_outliers(threshold_pct=10.0)` treats a 10% bar move as extreme. On 1h that is genuinely rare; on 1w it is
routine for BTC, so the flag saturates and 58% of weekly bars are "outliers" -- which is the same as flagging
nothing.

**Deliberately NOT retuned.** The brief specified the existing Layer 1 pipeline, and quietly changing a
threshold so a number looks reasonable is the failure mode this project guards against. It needs a
timeframe-aware threshold before 1d/1w outlier flags are used for anything, and until then those two columns
should be treated as absent rather than as clean.

### Macro gap classification needed its own calendar

`crypto_fetcher.classify_gaps` assumes a CONTINUOUS market. A weekend in 24/7 crypto is a genuine anomaly; a
weekend in Treasury yields is the market being shut. Running daily macro through the crypto classifier would
have reported roughly 2,900 false `data_error` gaps on DGS10 alone.

`fred_fetcher.classify_macro_gaps` therefore classifies against the exchange calendar -- `normal` (<= 1 business
day), `market_closed` (<= 4 calendar days, i.e. weekends and long weekends), `data_gap` (anything longer). The
`market_closed` counts in the table above are weekends, not defects; the `data_gap` counts (8 / 4 / 3) are the
real holes.

### Alignment rules, forward only

Funding and macro are lower-frequency series joined onto 1h bars, and both forward-fill ONLY. A funding rate
settled at 08:00 is unknown at 04:00, so back-filling it into 00:00-08:00 would leak the outcome of an interval
into bars inside that interval -- **lookahead that no splitter can catch, because it is baked into the
feature.** Bars before the first observation stay NaN, never 0 (I5): a zero funding rate is a real market state,
not an absence.

Tested directly (`test_funding_is_forward_filled_never_back_filled`,
`test_bars_before_the_first_settlement_are_nan_not_zero`).

Also recorded: FRED publishes on the observation date with a release lag, and this module does NOT model that
lag. A strict point-in-time backtest must shift these series by their real publication delay.

### Three defects found and fixed during ingestion

1. **Binance treats `startTime=0` as unset.** The first funding run returned 500 rows starting 2026-03-01 --
   5 months, silently, with no error. Verified directly: `startTime=0` -> 500 rows from 2026-03-01;
   `startTime=1568000000000` -> 1000 rows from 2019-09-10. Fixed with a positive epoch floor. Also fixed the
   stop condition: pagination broke on `len(batch) < limit`, but the endpoint returns short pages mid-history,
   so it ended years early. Now stops only when the cursor stops advancing. **7,592 rows recovered vs 500.**
2. **`_timeframe_minutes` could not parse `'1w'`** -- both the lookup table and the regex fallback stopped at
   `'d'`. Hypothesis condition 4 requires 1H/4H/1D/1W, so the weekly timeframe raised rather than resolving.
3. **pandas deprecated lowercase `'w'` in `resample`.** It still works today with a FutureWarning and will stop
   in a later pandas. Added `_pandas_resample_alias` so ccxt's vocabulary stays at the call sites, with a test
   that promotes the FutureWarning to an error -- it fails now rather than breaking silently on upgrade.

12 new tests. Tools kept: `scripts/ingest_stage_b1.py`,
`src/data_engine/derivatives_fetcher.py`, `src/data_engine/fred_fetcher.py`.

### Gate B status: NOT REACHED

| requirement | status |
|---|---|
| B.1 free sources ingested with documented coverage + gap classification | **DONE** |
| B.2 paid vendor evaluation (liquidations, OI history, long/short ratios) | **NOT STARTED** |
| B.3 flow feature construction | not started |
| B.4 sweep detector re-specification (`DEFAULT_LEVEL_TOLERANCE_ATR`, S8) | not started |
| Gate B: fully-conditioned event count, reported for BOTH windows | not started |
| Gate B kill: < 200 independent events over the flow window | not evaluated |

### What is still missing, and it is the expensive half

B.1 covers the free sources. **Liquidations -- named in MASTER_CONTEXT_v2 section 9 as the highest-priority
acquisition and the single most likely source of information absent from OHLCV -- are not obtainable from any
free endpoint and remain un-ingested.** So do open-interest history and long/short ratios beyond ~30 days.
These need a vendor (Tardis, Coinglass, Amberdata, Kaiko), and that evaluation was flagged as Day-1
critical-path work six sessions ago and has still not begun. Stage C, the decisive experiment, is blocked
behind it.

### Next Steps

1. **B.2 vendor evaluation -- the long pole.** Compare Tardis / Coinglass / Amberdata / Kaiko on history depth,
   liquidation granularity (per-event vs aggregated), OI interval, and cost. This is an operator procurement
   decision, not an agent task.
2. **Decide the dollar-index question** (DTWEXBGS vs UUP vs paid DXY) and record it once.
3. **B.3 flow features** from what is now ingested (funding level/ROC/extremity percentile, basis level and
   z-score) -- buildable today, and enough to give Stage C's C-2 arm partial content even before liquidations
   land.
4. **B.4 sweep detector retune** -- independent of vendor data, buildable now.
5. Carried forward: timeframe-aware outlier threshold, `realised_return` pricing (Stage D), vertical-barrier
   off-by-one, uniqueness/calibration interaction, `config.yaml`'s unread `backtesting:` section.

