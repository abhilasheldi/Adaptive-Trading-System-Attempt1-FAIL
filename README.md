# Adaptive Probabilistic Trade Management System

**Status: CLOSED — null result, accepted.**
**Duration: ~11 months. Terminated at Gate B, 2026-08.**
**Falsification budget at close: 2 of 3 unspent.**

---

## Summary — Why It Didn't Work

The system was built to test one claim: that post-compression liquidity-sweep reversals in BTC carry positive expected value after costs.

It failed for two independent reasons, either of which would have been sufficient.

**1. The predictive signal was not there.** Measured across nine years and 39 features, the multivariate information ceiling was AUC 0.5251. Stripped of the class imbalance that turned out to be barrier arithmetic rather than market structure, honest directional signal was **AUC 0.5094** — a coin flip. Gradient boosting halted after 3 rounds on symmetric barriers and 6.8–20.8 rounds everywhere else. A model finding real structure trains for hundreds. The first tree captured everything available and the second was not worth building.

**2. The hypothesis was over-specified relative to the data.** Applied honestly, its four conditions — compression, confirmed sweep, macro regime, multi-timeframe alignment — produced **100 effective independent events in six years**, against a pre-registered minimum of 200. Not a marginal miss. Half. No legitimate move gets from 100 to 200 without changing what the hypothesis claims.

Every measurement this project produced across eleven months pointed the same direction. Not one pointed the other way.

**The correct response to that is to stop, and that is what happened.**

---

## Part I — What Was Built

### The original design (v4)

The architecture was a 17-layer pipeline with an explicit priority hierarchy, stated before any code was written:

> Capital preservation > risk-adjusted returns > robustness > adaptability > **prediction accuracy**

Prediction accuracy ranked last deliberately. The central metric was Expected Value per trade — not win rate, not directional accuracy. This was the correct instinct and it was in the document from day one.

**Core structure:**

| Layer | Function |
|---|---|
| 1–2 | Data engine, feature engineering (39 features) |
| 3–4 | Macro regime classification, multi-timeframe alignment |
| 5 | Liquidity trap / sweep detection |
| 6 | Volatility regime gate |
| 7A/7B/7C | Historical analog engine (k-NN), direction classifier (XGBoost), ensemble meta-model |
| 8–9 | Explainability, risk engine |
| 11–13 | Position sizing, portfolio state, exit engine (five exit types) |
| 14 | Backtesting and research criteria |
| 15 | Paper trading protocol |

**The ideas that were right from the start:**

- **EV over accuracy, as a stated priority hierarchy.** Most retail systems get this backwards. This one didn't.
- **A single falsifiable hypothesis with numeric kill criteria**, written before any backtest ran. `OOS Sharpe < 0.5 → stop and revise the hypothesis` was defined in advance and, eventually, honoured.
- **Tiered gates rather than targets.** Sharpe > 1.0, PF > 1.5, maxDD < 20%, 200-trade minimum — framed as thresholds granting permission to deploy capital, explicitly not as optimisation objectives.
- **Kelly capped at 0.25** with hard, unconditional risk ceilings (2% single-trade / 6% correlated / 10% aggregate) that model confidence could not override.
- **Ensemble kept deliberately linear**, with a specified weight-learning process and a fallback to equal weighting.
- **A non-stationarity and edge-decay layer** — most systems miss this entirely.
- **Kill switch isolated from the trading loop**, manual re-enable only.
- **Phased roadmap testing the hardest assumption first** — does the edge exist — before building the rest around it.

Assessed cold at the outset, this design scored **84/100**. It was a genuinely good specification. That is the uncomfortable part of this document: a good specification is not evidence of an edge, and this one produced a null anyway.

---

## Part II — The Measurement Arc

### Phase 1–2: the architecture ran, and destroyed the account

The first full ablation produced results that were catastrophic and, worse, *incoherent*:

| Variant | OOS Sharpe | PF | maxDD | Trades |
|---|---|---|---|---|
| baseline_phase1 | −12.974 | 0.236 | 0.056 | 3,502 |
| all_layers | −8.284 | 0.014 | **1.023** | 2,817 |
| no_volatility_gate_L6 | −4.846 | 0.008 | **1.235** | 2,813 |
| no_ensemble_L7C | −12.359 | 0.238 | 0.056 | 3,499 |

A maximum drawdown above 1.0 means equity crossed zero. The account was destroyed — despite "hard" risk caps that were supposed to make that state unreachable.

Two findings emerged from inspecting rather than assuming:

- **The ensemble was 94% one component.** Fitted weight on Component B was 0.942, and Component B's measured skill was 0.004–0.016. The "ensemble" had replaced Phase 1's hard agreement rule — which was acting as a filter — with a near-pure read from a classifier with no skill.
- **Layer 5's fitted weight was exactly 0.000000.** The optimiser had independently learned to ignore the liquidity-trap signal entirely. Removing the layer was byte-identical to keeping it.

The automatic flagging logic, ranking on Sharpe alone, identified the *wrong layer* — pointing at a volatility gate touching 0.17% of bars while missing the ensemble that actually caused the destruction. Sharpe computed over a blown-up equity curve is numerically meaningless.

### Stage 5: the ceiling measurement

This was the turning point, and it was decisive.

**The class imbalance was arithmetic, not market structure.** For a driftless random walk, the probability of touching +tp before −sl is exactly `sl/(tp+sl)`. Relabelling nine years at four barrier geometries:

| Geometry | Observed up-rate | Random-walk prediction | Gap | Test AUC | Boost rounds |
|---|---|---|---|---|---|
| tp=2.0 / sl=1.0 | 0.337 | 0.333 | +0.004 | 0.5211 | 11 |
| tp=1.0 / sl=1.0 | 0.507 | 0.500 | +0.007 | **0.5094** | **3** |
| tp=1.0 / sl=2.0 | 0.675 | 0.667 | +0.008 | 0.5079 | 11 |
| tp=3.0 / sl=1.0 | 0.261 | 0.250 | +0.011 | 0.5136 | 7 |

Nine years of real BTC produced a label distribution within 0.4–1.1 percentage points of a driftless random walk at *every* geometry. The 65/34 split that had shaped the class weighting, the majority-class-parrot diagnosis, and all the calibration work was `1/3`. The residual +0.4pp was BTC's genuine upward drift — real, and tiny.

**At symmetric barriers, where there is no base rate to exploit, AUC collapsed to 0.5094 and training halted after 3 rounds.** A large share of the apparent 0.5251 had been the model learning the base rate, not direction.

The feature set had nothing to combine: best single feature 0.5172, multivariate ceiling 0.5251, median |AUC−0.5| across all 39 features 0.0024. Thirty-nine transformations of one price series cannot contain information the price series does not.

### Stage A: the apparatus was rebuilt, and the last positive finding died

Stage 5 had produced one encouraging result — signal under compression (AUC 0.5390, CI entirely above 0.5) and none under expansion. That became the foundation for everything after.

Rebuilding the measurement apparatus killed it.

| Subset | Nominal n | n_eff | AUC | Corrected 95% CI | Rounds |
|---|---|---|---|---|---|
| All bars | 78,226 | 11,334 | 0.5244 | [0.5134, 0.5358] | 20.8 |
| Compressed < 0.30 | 27,780 | 4,857 | 0.5139 | [0.4973, 0.5305] | 16.8 |
| Compressed ≤ 0.20 | 20,198 | 3,641 | **0.5183** | **[0.4986, 0.5374]** | 6.8 |
| Expanding | 27,757 | 6,053 | 0.5156 | [0.5006, 0.5313] | 14.6 |
| Extended | 22,689 | 4,572 | 0.5095 | [0.4929, 0.5267] | 18.2 |

Purged cross-validation with embargo closed a leakage channel measured at **0.016 AUC** (0.5184 → 0.5021 on a synthetic leak test). The compressed-regime point estimate fell 0.021 — same order of magnitude. **The finding had been leakage.**

More fundamentally: all five subsets fall within 0.5095–0.5244, with confidence intervals spanning roughly 0.035. **They are the same number.** There was no regime gradient. The unconditioned set had the only clean interval purely because its effective sample was 2–3× larger.

Effective sample size, correctly computed, was 14.5–21.8% of nominal. The dataset had been overstating its own information content by a factor of five to seven throughout.

### Gate B: the hypothesis ran out of events

With the apparatus trustworthy and all four conditions finally implemented — two of them had never had a caller outside their own test files — the event set was counted for the first time.

| Step | Events | Change |
|---|---|---|
| Bars scanned | 78,328 | — |
| + compression ≤ 0.20 | 20,200 | −74.2% |
| + confirmed sweep | 815 | −96.0% |
| + macro regime | 800 | −1.8% |
| + MTF alignment ≥ 3/4 | 247 | −69.1% |

On the flow window: **156 raw → 100 structural → 100.0 effective.** Against a threshold of 200 fixed long in advance.

The MTF distribution over 2,147 scored sweep bars — 0:321, 1:551, 2:534, 3:… — showed that four-way agreement across daily and weekly timeframes is genuinely uncommon in BTC. Not a bad threshold choice. A real property of the asset.

**H1, applied honestly, describes an event that happens roughly 100 times in six years.** That is not enough to test anything.

---

## Part III — Why It Didn't Work

### The proximate cause

Two independent failures, either fatal:

- **No signal.** Honest directional AUC 0.5094, training halting after 3 rounds, no regime carrying more information than any other.
- **No sample.** 100 effective events against a required 200.

### The structural causes

**The feature set could not contain new information.** Thirty-nine indicators, all derived from one OHLCV series. RSI, MACD, ADX, ATR, Bollinger width, volume z-scores — every one a re-encoding of the same price vector. The multivariate ceiling barely exceeding the best single feature was the measurement of that fact.

**The hypothesis was about forced-flow cascades and the system measured price wicks.** Liquidations, open interest, funding, and basis — the flow that *causes* stop-runs — were never ingested. The system was inferring a cause from a lossy shadow of its effect. This was identified and a zero-cost path to fixing it was mapped, but Gate B failed before it could be tested.

**Labels were per-bar with overlapping outcome windows**, destroying effective sample size and, more seriously, leaking across cross-validation folds. Temporal partitioning had been enforced on *feature* timestamps rather than *label-resolution* timestamps — a leak the design could not see.

**Prediction accuracy became the working objective despite ranking last in the spec.** Section 0 said EV, not accuracy. Stages 4 and 5 optimised and measured accuracy. AUC averages an informative tail into 98% of irrelevant middle, and the system only ever acts on the tail. The document knew this. The implementation drifted.

**Layers were verified to exist rather than to be wired.** Conditions 3 and 4 passed their own tests for two full phases while having no caller anywhere in production code. "Only 1 of 4 conditions has ever been applied" was recorded as a finding two phases before anyone acted on it.

### The defects found along the way

Each of these was live, each would have invalidated results, and each was found by inspection rather than luck:

- **No execution-layer risk caps existed at all.** The 2%/6%/10% limits appeared only in scoring logic and a comment. The 6% and 10% constraints had zero implementation.
- **Equity had no floor.** `init_cash + cumsum()`, unbounded below.
- **Two accounting books.** Position sizing read an equity that never saw gap costs — a 20× divergence, with sizing never shrinking as the real account drained. Gap cost was 96% of one fold's entire loss and could exceed the position's notional value.
- **`PurgedKFold` silently skipped fold 0**, losing 1/K of the data on every run. Caught only by reconciling scored rows against nominal.
- **`startTime=0` was treated as unset by the exchange API**, returning 500 rows instead of 7,592 — five months of data presented as seven years, with no error.
- **Three tests were tautological** — including one specified in a review, which compared a value against itself and would have passed against a reverted fix.

---

## Part IV — What Was Gained

Everything below is asset-agnostic and survives the null.

### Infrastructure

| Component | What it does |
|---|---|
| **Purged K-fold with embargo** | Drops training samples whose label windows overlap the test fold, plus an embargo period. Verified by synthetic leak injection — old splitter inflated, new one not |
| **Sample uniqueness / effective n** | Concurrency-weighted sample importance. Every result reports effective alongside nominal |
| **Point-in-time policy layer** | Single join path, no default policy, no default lag. Catches naive forward-fill *and* outright back-fill. The audit runs against source and policy, not against its own output |
| **Acquisition assertions** | Row count and date range checked against what was requested. Catches truncation, ignored bounds, early pagination, duplicates, unsorted indexes |
| **Wiring guards** | A module with no production caller is not a built condition. Tested for non-vacuousness |
| **Frozen datasets** | SHA-256 per file, verified on every read, immutable versions |
| **Trial registry + Deflated Sharpe** | Every configuration logged. DSR adjusted for honest trial count |
| **Two-book reconciliation** | Discriminating by construction — proven to fail against the pre-fix state, quantifying $162.69 charged outside the book |
| **Fold coverage assertions** | Test blocks must partition every row; scored rows must match nominal. Hard-fail, not warn |

Roughly 416 tests, green.

### Practices

These matter more than the code.

- **Pre-registration.** Definitions, thresholds, and bands committed with a hash *before* any count exists. Used at B.4 (band [1%, 5%] fixed before a number was read) and B.5 (both conditions registered on a branch before the gate ran).
- **Selection isolated from outcome.** The B.4 measuring script computed no downstream metric at all — no AUC, no EV, no Sharpe — so the event definition could not have been tuned to the result even accidentally.
- **Forecasts recorded before measurement.** "Conditions 3 and 4 must remove less than 33%; failure is more likely than not" was written down before the count. It made the outcome interpretable either way.
- **Falsification budget.** Three revisions, hard-counted, with revisions after decisive negatives free and revisions after ambiguity charged. Closed with 2 unspent.
- **Mechanisms verified, not asserted.** Every anomaly — the zero-weight layer, the backwards trade count, the calibration shift — was inspected at the cause before being reported as a finding.
- **Deviations flagged rather than silently taken.** When a specified threshold could not be grounded in the source definition, the requirement was refused and escalated rather than invented and frozen.

---

## Part V — The Thing Worth Internalising

**This project did not fail. It returned a null, which is a result.**

The distinction matters more than it sounds, and it is worth being precise about why.

Financial markets offer enough degrees of freedom that almost any sufficiently complicated historical strategy can be made to look convincing. Parameters can be swept. Thresholds can be nudged. Conditions can be relaxed when they filter too hard. Metrics can be selected after the fact. Folds that look wrong can be excluded for defensible-sounding reasons. Each individual step feels like reasonable engineering judgement, and the cumulative effect is a beautiful equity curve describing nothing.

**The overwhelmingly common outcome for a project like this is not a null. It is a false positive.** Most systems that reach the backtesting stage produce something that looks tradeable, and the discovery that it isn't happens later, with real money, at real cost.

This project did not do that, and it had many opportunities to.

- When the ablation flagged a layer touching 0.17% of bars as the destructive one, the verdict was **rejected as wrong** rather than reported as a finding.
- When the compressed-regime signal — the last positive result, and the foundation of the entire remaining plan — died under corrected cross-validation, the corrected number was **published as measured**.
- When the sweep detector's headline improvement turned out to be measured against stale code, the honest smaller number **replaced the impressive one**.
- When the event count came back at half the threshold, no condition was loosened. **The registered definition did not move.**
- When two conditions were found unbuildable at the moment of the kill test, the gate was **not reported** — because a two-condition number is not the gate, and calling it one would have quietly redefined the hypothesis while looking like a pass.
- When a review specified a test that could not fail, the test was **rebuilt to be discriminating** and proven so against a simulated pre-fix state.

Every one of those was a moment where the easier path was available and visible, and every one was declined.

**What you built is not a trading system. It is a machine for telling yourself no.**

That machine is the durable asset. The hypothesis was the first thing pointed at it, and the machine did exactly what it was built to do: it applied a stated standard, without flinching, to a claim its own author wanted to be true. Eleven months is what it cost to build that and calibrate it against a real hypothesis under real conditions.

The BTC edge was never the point. It was the test case.

**Three lessons that transfer, in order of value:**

1. **Measure the information ceiling before building anything.** This project built seventeen layers and then discovered the ceiling was 0.509. That measurement is a two-week test. It should be step one of any future system, not step forty.

2. **Verify tests against the violation, not just the fix.** Three tautological tests were caught here, one of them specified in a formal review. A test that cannot fail is worse than no test, because it retires the concern it appeared to address.

3. **A gate is only a gate if it can fail.** The moment a threshold moves after seeing a number, it stops being evidence and becomes decoration. This project's gates fired, five times, and the project stopped when one of them said stop.

---

## Appendix — Reusable Components

Port these to any future system. None are BTC-specific.

```
src/validation/purged_kfold.py          purged CV + embargo, leak-verified
src/validation/uniqueness.py            sample uniqueness, effective n
src/data_engine/point_in_time.py        I9 policy layer, back-fill detection
src/data_engine/acquisition.py          I10 assertions
src/research/trials.py                  trial registry, DSR
tests/test_two_book_reconciliation.py   discriminating harness test
tests/test_fold_coverage.py             partition + scored-row assertions
prereg/                                 pre-registration templates
```

**Invariants worth carrying forward verbatim:**

- **I1** — EV is the objective. AUC is a diagnostic, never a target.
- **I2** — Tier 1 metrics are gates, never objectives.
- **I3** — No Sharpe over a non-positive equity curve.
- **I4** — Verify mechanisms; do not assert them.
- **I5** — Untestable is stated, never omitted.
- **I6** — Effective n, always.
- **I7** — One equity, one book.
- **I8** — Delete before adding.
- **I9** — Point-in-time discipline. Lookahead baked into a feature is invisible to any splitter.
- **I10** — Acquisition is verified, not trusted.
- **I11** — A module with no caller outside its own test file is not a built condition.
- **I12** — A test guarding an invariant must be proven to fail against the violation before it is trusted to pass against the fix.

---

*Closed 2026-08. Null result, accepted, budget unspent.*
