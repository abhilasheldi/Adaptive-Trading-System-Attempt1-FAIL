# BUILD PLAN v2

**Entry state:** End of Phase 2. Layers 1, 2, 6, 7A/7B/7C, 9 (partial), 14 built. Layers 3, 4 blocked on data. Layer 5 built but mistuned. Layers 11, 12, 13 and execution-layer risk caps do not exist.

**Read MASTER_CONTEXT_v2.md first.** This plan assumes its Settled Findings and Invariants.

---

## Sequencing Rationale

Two changes from the v5 draft ordering, both driven by lead time and information value:

1. **Data acquisition starts immediately and runs parallel to apparatus work.** Procurement, API integration, and vendor evaluation have wall-clock lead time that does not consume developer attention. Starting it at Stage B serialises the critical path unnecessarily.

2. **The flow-data ceiling test is the decisive experiment, and it is promoted.** If O1 fails (compressed signal does not survive CI correction) but O2 succeeds (flow features lift the ceiling), the project continues. If both fail, it terminates. Structuring Stage A as a hard kill gate would terminate the project before running its most informative test.

**Stage A therefore gates on apparatus correctness, not on the compressed finding surviving.** The compressed finding's survival becomes an input to Stage C's interpretation, not a standalone kill.

---

## Stage 0 — Teardown (3–5 days)

Delete before building. Dead components reappear in ensembles.

**Tasks**

| # | Task | Acceptance |
|---|---|---|
| 0.1 | Delete `HistoricalAnalogEngine`, Component B direction classifier, `EnsembleMetaModel`, `fit_ensemble_weights` | Grep returns zero references outside git history |
| 0.2 | Delete class-weighting machinery and imbalance-premised calibration | Same |
| 0.3 | Remove `Phase2Strategy.max_position` as a risk mechanism; leave a hard `NotImplementedError` stub where real caps will go | Tests fail loudly if anything sizes without caps |
| 0.4 | Update Layer 6 threshold 0.30 → 0.20 | Constant is single-sourced, not duplicated |
| 0.5 | Prune the test suite of tests that only covered deleted components; record count before/after | Suite green; new count recorded in `project_state.md` |

**Gate 0:** Test suite green. Repo contains no reference to deleted components. Recorded in `project_state.md`.

---

## Stage A — Measurement Apparatus (2–3 weeks)

Nothing measured under the current apparatus is trustworthy. This is the highest-priority work in the project.

### A.1 — Label metadata

Triple-barrier labelling must emit, per sample: `t0` (event time), `t1` (barrier-touch or vertical-barrier time), `barrier_touched ∈ {tp, sl, vertical}`, `realised_return`.

`t1` is the critical addition — purging is impossible without it, and its absence is why the existing temporal partitioning leaks.

### A.2 — Sample uniqueness

Implement per López de Prado (AFML ch. 4):

```
indicator matrix   1[t,i] = 1 if label i's window [t0_i, t1_i] spans bar t
concurrency        c_t    = Σ_i 1[t,i]
uniqueness         u[t,i] = 1[t,i] / c_t
avg uniqueness     ū_i    = Σ_t u[t,i] / Σ_t 1[t,i]
effective n        n_eff  = Σ_i ū_i
```

Use `ū_i` as training sample weight. Optionally add return-attributed weights: `w_i = |Σ_{t∈[t0,t1]} r_t / c_t|`, normalised to mean 1.

**Report `n_eff` alongside nominal n in every result** (Invariant I6).

### A.3 — Purged K-Fold with embargo

For each test fold `[T0, T1]`:
- **Purge:** drop training samples where `[t0_i, t1_i]` overlaps `[T0, T1]`
- **Embargo:** additionally drop training samples with `t0_i ∈ [T1, T1 + h]`, `h = 0.01 × total observations`

Replace the existing walk-forward CV. Keep the walk-forward *reporting* structure (per-fold detail is valuable); replace the *splitting* logic.

### A.4 — Harness defect fixes

| Fix | Detail |
|---|---|
| Bankruptcy floor | `equity_curve_from_trade_pnls` floors at 0; sets `equity_destroyed` |
| Single book | Reconcile vectorbt sizing equity with reported equity. All costs — including gap — applied inside the book sizing reads |
| Gap cost cap | Cap total gap charge at position notional. Re-derive the per-bar linearity assumption; document whatever replaces it |
| Ablation flagging | Gate on maxDD and PF before Sharpe. Refuse to rank on Sharpe when `any_equity_destroyed` |

### A.5 — Trial registry and DSR

`trials.jsonl`: one line per evaluated configuration — `{config_hash, timestamp, stage, params, n_eff, result_metrics}`. Optuna trials auto-log. Implement `deflated_sharpe_ratio(returns, n_trials, skew, kurtosis)`.

### A.6 — Recompute Stage 5 under corrected apparatus

Re-run the unconditioned and conditioned ceiling measurements with purged CV, uniqueness weighting, and corrected CIs.

**Expected outcome, stated in advance so it is not read as failure:** effective n is plausibly 3–5% of nominal, which widens the compressed-regime CI substantially. `[0.519, 0.559]` may become something like `[0.49, 0.59]`. **This is the honest number, and it straddling 0.5 is a likely outcome, not a bug.**

**Gate A (apparatus, not hypothesis):**
- Purged CV verified by a synthetic leakage test: inject a deliberately leaked feature, confirm the old splitter shows inflated AUC and the new one does not
- `n_eff` reported everywhere; corrected CIs published
- Harness fixes verified by reproducing fold 2 of `all_layers` and confirming sizing-book and reported maxDD now agree
- Test suite green

**Gate A does not kill the project.** Its output — whether S5 survives — is recorded and carried into Stage C.

---

## Stage B — Data Acquisition (starts Day 1, parallel; 3–4 weeks wall-clock)

### B.1 — Free and immediate

| Source | Endpoint | Note |
|---|---|---|
| Funding rates | Binance `/fapi/v1/fundingRate` | Full history, 8h intervals |
| Macro | FRED (DXY, US10Y, VIX) | Unblocks hypothesis condition 3 |
| 1D / 1W bars | Existing OHLCV pipeline | Unblocks hypothesis condition 4 |
| Perp–spot basis | Computed from spot + mark price | Derived, not fetched |

### B.2 — Paid, on the critical path

Liquidations, open interest history, and long/short ratios beyond ~30 days require a vendor. Evaluate Tardis, Coinglass, Amberdata, Kaiko on: history depth, liquidation granularity (per-event vs aggregated), OI interval, and cost.

**Start vendor evaluation on Day 1.** This is the longest lead-time item in the project.

### B.3 — Flow feature construction

Minimum viable set, all percentile-ranked against trailing 252-bar history per Layer 2 convention:

- Liquidation volume (long/short separately), z-scored vs trailing 20
- Liquidation cluster proximity: distance from price to nearest recent liquidation cluster
- OI change over 1/4/24 bars; OI-to-volume ratio
- Funding rate level and rate of change; funding extremity percentile
- Basis level and z-score
- Long/short account ratio and its rate of change

**Rationale for prioritising liquidations:** H1 is a hypothesis about forced-flow cascades. Detecting them from the price wick they leave is inferring a cause from a lossy shadow of its effect. This is the single most likely source of information absent from OHLCV.

### B.4 — Sweep detector re-specification

Named suspects from prior sessions:

- `DEFAULT_LEVEL_TOLERANCE_ATR = 0.25` — a quarter-ATR tolerance matches most wicks. Sweep candidate values `{0.05, 0.10, 0.15}` and report confirmation rate for each. **Do not select on downstream performance** — select on confirmation rate landing in low single digits, which is a definitional target set in advance, not an optimisation.
- No compression precondition applied — add ATR pct ≤ 0.20 as a gate on sweep detection
- The "ATR ≤ 20th percentile" definition from the source document was never wired in; wire it

**Gate B:**
- Flow data ingested with documented coverage window and gap classification
- Fully-conditioned event set built; **event count reported for both the 9-year OHLCV window and the ~6-year flow window**
- Sweep confirmation rate in low single digits
- **Kill: fully-conditioned independent event count < 200 over the flow window → H1 fails on sample grounds.** Terminate or spend one falsification-budget revision (this would be a *decisive negative* and therefore free — see MASTER_CONTEXT §6).

---

## Stage C — The Decisive Ceiling Test (1–2 weeks)

The highest-information experiment remaining. Run it before building anything else.

**Design:** on the fully-conditioned event set, under purged CV with uniqueness weighting, measure the multivariate ceiling three ways:

| Arm | Features | Purpose |
|---|---|---|
| C-1 | 39 OHLCV features only | Baseline on events (vs 0.5251 on all bars) |
| C-2 | Flow features only | Is flow independently informative? |
| C-3 | OHLCV + flow | The number that decides the project |

Report AUC with corrected CI, logloss vs baseline, `n_eff`, and boost rounds before early stopping for each arm.

**Read the boost-round count as a primary diagnostic, not a footnote.** S4 established that 2–32 rounds means nothing to learn. If C-3 still halts under 40 rounds, that is a negative result regardless of what AUC says.

**Gate C — the project's main decision point:**

| Outcome | Action |
|---|---|
| C-3 CI entirely above 0.5 **and** C-3 materially exceeds C-1 **and** boost rounds > 40 | Proceed to Stage D |
| C-3 CI above 0.5 but no lift over C-1 | Flow adds nothing. H1 falsification condition 5 met — decisive negative, free revision or terminate |
| C-3 CI includes 0.5 | Decisive negative. Terminate or free revision |

**Do not proceed to Stage D on a marginal Gate C.** A marginal result here is exactly the ambiguity that consumes falsification budget in MASTER_CONTEXT §6. Marginal means stop.

---

## Stage D — Meta-Label Model (3–4 weeks)

Only reached if Gate C passes cleanly.

### D.1 — Primary rule (deterministic, no ML)

```
IF   atr_percentile ≤ 0.20
AND  confirmed_sweep(tolerance = tuned in B.4)
AND  mtf_alignment ≥ 3/4
THEN emit {side, entry, stop, target}
```

No learned parameters. This is the event definition, and it must be frozen before D.2 begins.

### D.2 — Secondary model (the only learned component)

- **Input:** feature vector at event time + event metadata + flow features
- **Label:** binary — did this event reach target before stop?
- **Output:** `P(success)` → size via `m = 2Φ(z) − 1` where `z = (p − ½)/√(p(1−p))`, then hard-capped by Stage E risk limits
- **Training:** purged CV, uniqueness-weighted, trial-logged

### D.3 — Measurement (Invariant I1 governs)

Primary: **top-decile realised EV after 75th-percentile costs.** Secondary: lift over base rate at operating threshold, calibration on the event set, `n_eff`.

**AUC is reported and not optimised.**

**Gate D:** top-decile realised EV > 0 after costs, out-of-sample, on ≥ 200 independent events.

---

## Stage E — Execution, Risk, and Exits (4 weeks)

The system has never been tested. This stage builds the parts that were specified but never implemented.

### E.1 — Real execution-layer risk caps

Per S10, these do not exist. Build them as a hard gate the order path cannot bypass:

- 2% max single-trade risk (risk, not size — requires stops from E.2)
- 6% max correlated-direction exposure — **portfolio-level, currently zero implementation**
- 10% max aggregate open risk — **currently zero implementation**

Enforced below the model layer. No model output can raise them. Test: assert that a deliberately malicious model output demanding 50% size is clamped.

### E.2 — Exit engine (Layer 13)

Five types per v4: target achieved, forecast deterioration, risk increase, opportunity cost, time decay. **There are currently no stops at all**, which is why S10's size cap was never a risk cap.

Log full feature state at every exit decision — this becomes the dataset for backtesting exit logic independently of entry logic.

### E.3 — Full-system backtest

Complete system, purged CV, realistic costs, fixed-fractional sizing (Stage 1 only — no Kelly).

**Gate E:** v4 Tier 1, all six, on untouched out-of-sample data, plus DSR > 0.5 adjusted for the full `trials.jsonl` count.

---

## Stage F — Decision

Clear Gate E → paper trading per Layer 15 (100 completed trades, ≥ 2 regime transitions, no acceleration).
Fail any gate → one falsification-budget revision, or terminate.

---

## Revision Pre-Registration Template

Complete **before** examining any data under a revised hypothesis. Commit it, then measure.

```
REVISION [n] of 3          Date: ________
Budget consumed: [YES — marginal result] / [NO — decisive negative]

TRIGGER
  Which measurement motivated this? (cite gate + numbers)
  Was its CI decisive or marginal?

CHANGE
  H1 before: ____________________
  H1 after:  ____________________
  What specifically differs?

KILL CRITERIA FOR THIS REVISION
  Numeric, pre-specified: ____________________

DATA
  Which slice will be used?
  Which slice remains untouched for final validation?

Committed by: ________  Commit hash: ________
```

---

## Effort Summary

| Stage | Duration | Blocking? |
|---|---|---|
| 0 — Teardown | 3–5 days | Yes |
| A — Apparatus | 2–3 weeks | Yes |
| B — Data | 3–4 weeks wall-clock | Parallel from Day 1 |
| C — Ceiling test | 1–2 weeks | **Main decision point** |
| D — Meta-label | 3–4 weeks | Only if C passes cleanly |
| E — Execution/risk/exits | 4 weeks | Only if D passes |
| F — Decision | — | — |

**Critical path to the decisive answer: roughly 6–9 weeks.** That is the number that matters — it is when you find out whether this project has an edge, and it is deliberately front-loaded so that a negative answer arrives cheaply.
