# BUILD PLAN v2

**Entry state:** Stage 0 and Stage A complete (Gate A pending operator sign-off). Stage B.1 complete. Layers 1, 2, 6, 7A/7B/7C, 9 (partial), 14 built. Layers 3 and 4 unblocked by B.1. Layer 5 built but mistuned. Layers 11, 12, 13 and execution-layer risk caps do not exist.

**Read MASTER_CONTEXT_v2.md first.** This plan assumes its Settled Findings and Invariants.

---

## Changelog

| Date | Trigger | Change |
|---|---|---|
| 2026-08-15 (b) | Binance Data Vision source review | **No paid data is required (S23).** Vendor decision table deleted from Gate B — the gate is now purely the event count, with no procurement branch. B.2 becomes free bulk acquisition, **3–5 days with no lead time** (was 1–2 weeks wall-clock). Critical path to the decisive answer shortens by ~1–2 weeks to **4–7 weeks from Gate B**. Sequencing rationale items 1 and 3 updated: the lead-time argument for parallelism is retired. **B.2 and B.3 bodies are stubs awaiting the operator's §3/§4 text** — the obsolete vendor content was deleted rather than left standing, since S23 falsifies it. |
| 2026-08-15 | A.6 results; B.1 ingestion; operator rulings 1–7 | Amendments applied inline. Gate A criterion 5 replaced with the two-book reconciliation test (both outstanding tests now built and green — see Stage A). Stage B resequenced to **B.0 → B.4 → B.5 → decision → B.2 → B.3** so the free kill test precedes procurement. B.5 gates on **effective** event count over the **four**-condition set, BTC only. FRED publication lag promoted to blocking in B.0. Gate C recalibrated, C-4 arm added, "materially exceeds C-1" clause removed. Stage D given the stratified-uniqueness fix. Duplicate effort summary removed. |

---

## Sequencing Rationale

Two changes from the v5 draft ordering, both driven by lead time and information value:

1. **Data acquisition starts immediately and runs parallel to apparatus work.** *Superseded 2026-08-15:* this was written when acquisition meant procurement with wall-clock lead time. No paid data is required (S23), so acquisition is now a bounded engineering task with no lead time to hide behind. The parallelism argument no longer applies; B.2 simply follows Gate B.

2. **The flow-data ceiling test is the decisive experiment, and it is promoted.** If O1 fails (compressed signal does not survive CI correction) but O2 succeeds (flow features lift the ceiling), the project continues. If both fail, it terminates. Structuring Stage A as a hard kill gate would terminate the project before running its most informative test.

**Stage A therefore gates on apparatus correctness, not on the compressed finding surviving.** That structuring paid off: O1 resolved negative at A.6 (S5 retired) and the project correctly did not terminate. The compressed finding's survival is an input to Stage C's interpretation, not a standalone kill.

A third change, added 2026-08-15:

3. **Stage B's free kill test is promoted ahead of acquisition work.** All four hypothesis conditions use OHLCV + FRED only, so Gate B's kill criterion is computable from data in hand. The original B.2 → B.3 → B.5 order serialised acquisition ahead of a free "no" — which mattered more when B.2 was a purchase, but still holds now that it is a multi-day download: there is no reason to ingest flow data for an event set that may not exist.

---

## Stage 0 — Teardown (3–5 days) — **COMPLETE**

Delete before building. Dead components reappear in ensembles.

| # | Task | Acceptance |
|---|---|---|
| 0.1 | Delete `HistoricalAnalogEngine`, Component B direction classifier, `EnsembleMetaModel`, `fit_ensemble_weights` | Grep returns zero references outside git history |
| 0.2 | Delete class-weighting machinery and imbalance-premised calibration | Same |
| 0.3 | Remove `Phase2Strategy.max_position` as a risk mechanism; leave a hard `NotImplementedError` stub where real caps will go | Tests fail loudly if anything sizes without caps |
| 0.4 | Update Layer 6 threshold 0.30 → 0.20 | Constant is single-sourced, not duplicated |
| 0.5 | Prune the test suite of tests that only covered deleted components; record count before/after | Suite green; new count recorded in `project_state.md` |

**Gate 0:** Passed.

---

## Stage A — Measurement Apparatus (2–3 weeks) — **COMPLETE, GATE PENDING SIGN-OFF**

Nothing measured under the previous apparatus was trustworthy. This was the highest-priority work in the project.

### A.1 — Label metadata

Triple-barrier labelling emits, per sample: `t0`, `t1`, `barrier_touched ∈ {tp, sl, vertical}`, `realised_return`. `t1` is the critical addition — purging is impossible without it.

### A.2 — Sample uniqueness

Per López de Prado (AFML ch. 4):

```
indicator matrix   1[t,i] = 1 if label i's window [t0_i, t1_i] spans bar t
concurrency        c_t    = Σ_i 1[t,i]
uniqueness         u[t,i] = 1[t,i] / c_t
avg uniqueness     ū_i    = Σ_t u[t,i] / Σ_t 1[t,i]
effective n        n_eff  = Σ_i ū_i
```

`ū_i` is the training sample weight. **Report `n_eff` alongside nominal n in every result** (I6). This same machinery now defines the independent-event count at B.5 — see MASTER_CONTEXT §7.

### A.3 — Purged K-Fold with embargo

For each test fold `[T0, T1]`:
- **Purge:** drop training samples where `[t0_i, t1_i]` overlaps `[T0, T1]`
- **Embargo:** additionally drop training samples with `t0_i ∈ [T1, T1 + h]`, `h = 0.01 × total observations`

Replaces the splitting logic of the walk-forward CV; the walk-forward *reporting* structure is kept.

### A.4 — Harness defect fixes

| Fix | Detail |
|---|---|
| Bankruptcy floor | `equity_curve_from_trade_pnls` floors at 0; sets `equity_destroyed` |
| Single book | vectorbt sizing equity reconciled with reported equity. All costs — including gap — applied inside the book sizing reads |
| Gap cost cap | Total gap charge capped at position notional; per-bar linearity re-derived |
| Ablation flagging | Gates on maxDD and PF before Sharpe; refuses to rank on Sharpe when `any_equity_destroyed` |

### A.5 — Trial registry and DSR

`trials.jsonl`: one line per evaluated configuration — `{config_hash, timestamp, stage, params, n_eff, result_metrics}`. Optuna trials auto-log. `deflated_sharpe_ratio(returns, n_trials, skew, kurtosis)` implemented.

### A.6 — Recompute Stage 5 under corrected apparatus

Complete. Outcome: **S5 and S6 retired.** CIs widened 2.1–2.65×, matching the `sqrt(1/0.144) ≈ 2.63` prediction recorded in advance at A.2. Compressed ATR ≤ 0.20 fell 0.5390 → 0.5183 with CI [0.4986, 0.5374] — includes 0.5. Pre-registered as a likely outcome and therefore **not read as failure**.

### Gate A — apparatus, not hypothesis

| Criterion | Status |
|---|---|
| Purged CV verified by synthetic leakage test (old splitter inflated, new not) | **MET** — 0.5184 → 0.5021, leak channel closed |
| `n_eff` reported everywhere alongside nominal n | **MET** |
| Corrected CIs published | **MET** — 2.1–2.65× wider, naive CIs shown alongside |
| **Two-book reconciliation** (substitute for the retired fold-2 criterion) | **MET, 2026-08-15** — see below |
| **Fold-coverage assertion** | **MET, 2026-08-15** — see below |
| Test suite green | **MET** — 374 passed |

**Two-book reconciliation (substitute for the retired fold-2 criterion).**
The original criterion required code deleted at Stage 0 and is not regenerable. Its purpose was verifying the S12 fix, which does not need that specific artifact. Substitute, now built as `tests/test_two_book_reconciliation.py`: a synthetic strategy holds positions across bars carrying non-zero gap cost, runs through the harness, and the sizing book and reported equity are asserted to agree.

The test is **discriminating, not tautological**. Post-fix the reported curve *is* `pf.value()`, so equality is true by construction and would pass against a reverted harness. The property that separates one book from two is visible in the order sizes: `size_type='targetpercent'` sizes off the portfolio's own equity, so a gap charge *inside* the book changes the sizes, while a charge subtracted afterwards leaves them identical. The test prices identical positions with the gap channel live and off, and asserts both equity and sizes move. Verified against a simulated revert: two assertions fail, quantifying the out-of-book charge at **$162.69** (reported equity moved $611.54 while trades accounted for $448.85).

**Fold-coverage assertion (added after the A.3 defect).**
`PurgedKFold._assert_test_blocks_partition_every_row` hard-fails when held-out blocks overlap or leave rows unscored, checked at generator exhaustion (partial iteration is a caller's choice and does not trip it). `assert_scored_rows_match_nominal` covers the other end: a splitter can partition correctly and a caller can still drop a fold. Wired into `measure_ceiling_corrected.py`, replacing a silent `continue`. The original PurgedKFold skipped fold 0 and lost 1/K of the data, caught only by manual reconciliation. This is now a guard, not a habit.

**Gate A does not kill the project.** Its output — that S5 did not survive — is recorded and carried into Stage C. **Gate A is not self-certified by the agent (MASTER_CONTEXT §10) and requires operator sign-off.**

---

## Stage B — Data Acquisition and Event Set

**Resequenced 2026-08-14.** The original order (B.2 vendor → B.3 features → B.5 event set) serialises a paid procurement decision ahead of a free kill test. All four hypothesis conditions use OHLCV + FRED only — **Gate B's kill criterion is computable before spending anything.**

New order: **B.0 → B.4 → B.5 → decision → B.2 → B.3**

### B.0 — Hardening — **COMPLETE 2026-08-18**

**Renumbered by the operator's 2026-08-18 task list.** The previous draft had B.0.4 = dollar index, B.0.5a/b = outlier audit, B.0.6 = FRED lag. The operator's numbering is authoritative and is used below; the outlier audit is retained as B.0.7 since it was already complete and is not in the new list.

| # | Task | Status |
|---|---|---|
| B.0.1 | Freeze dataset to `data/frozen/`, hash in `project_state.md` | **DONE** — v2, `a5f81b5b8585d982…` |
| B.0.2 | I9 point-in-time; observation timestamp declared separately from value timestamp | **DONE** — `data_engine/point_in_time.py` |
| B.0.3 | I10 acquisition assertion (row count + date range vs requested) | **DONE** — `data_engine/acquisition.py` |
| B.0.4 | **FRED publication lag — BLOCKING.** ALFRED vintages, else shift ≥ 1 business day | **DONE** — 1 business day; ALFRED hook raises (no API key) |
| B.0.5 | DTWEXBGS confirmed; **UUP path deleted entirely**; substitution stated in reports | **DONE** — route closed in `macro_fetcher`, not just unconfigured |
| B.0.6 | Base currency GBP via FRED DEXUSUK; equity/P&L/drawdown/caps in GBP; both reported | **DONE** — `backtesting/currency.py` |
| B.0.7 | Audit whether the outlier flag is consumed downstream | **DONE 2026-08-15** — written, never read (S25). Cosmetic |
| B.0.7b | Timeframe-aware outlier threshold (rolling MAD, fixed multiplier, set in advance) | **Deferred — non-blocking**, per B.0.7 |

**Why B.0.4 was blocking and B.0.7b is not.** Macro gates event *formation* (MASTER_CONTEXT §4, four conditions), so unmodelled publication lag contaminates the event set itself and every downstream measurement inherits it — strictly worse than contaminating a feature, which stays confined to one arm. The outlier flag by contrast turned out to be write-only, so no series is destroyed and MTF alignment is unaffected.

**One business day is a floor, not the truth.** Real FRED release delays vary by series and vintage. `fetch_alfred_vintage` raises rather than silently falling back, so a vintage-accurate run can never be confused with an approximate one in the output. It needs a `FRED_API_KEY`, which this environment does not have.

**What B.0.6 changes, and what it does not.** Percentage risk caps are currency-invariant — 2% of GBP equity buys the same position as 2% of the equivalent USD equity, because the rate cancels. So the 2/6/10% numbers are unchanged; what changes is the base they apply to, pinned by `currency.RiskCapBasis` for E.1 to read. **Gate metrics are not invariant**: Sharpe, maxDD and PF all move because FX variance enters the return series, and the GBP figure is the one the Tier 1 gates read. Both are reported side by side so the FX contribution is visible rather than absorbed.

### B.1 — Free sources — **COMPLETE**

Funding, basis, 1d/1w, FRED all ingested. See MASTER_CONTEXT §9.

### B.4 — Sweep detector retune — **COMPLETE 2026-08-18**

**Selected: `level_tolerance_atr = 0.05`, `min_penetration_atr = 0.25`.** Confirmation rate **4.03%** on compressed bars (2.94% on all bars), **815** two-condition events over the full OHLCV window, **516** over the flow window.

Selection was on confirmation rate against a **1–5% band fixed in advance**, with no downstream metric computed by the measuring script at all — so the event definition could not have been tuned to the outcome. Exactly one of eight configurations landed in the band.

**S8's 44.1% was stale.** The penetration and `min_sweep_touches` selectivity requirements landed 2026-08-14, after S8 was recorded. The unretuned detector measures **10.93%**, not 44.1%, so B.4 began from a much better starting point than the brief assumed. Both numbers are real; they describe different code.

**The coupled sweep is non-monotonic and never reaches the band** — tightening tolerance 0.25 → 0.15 *raises* the compressed rate 13.04% → 17.27%, because tolerance and penetration were tied and pull in opposite directions. Decoupling them is what made the band reachable. `penetration ≥ tolerance` is now enforced as the bound that survives.

**Stability verified across time.** Splitting the flow window at its midpoint (2023-04-18), the selected config lands in-band on both halves independently — H1 4.29% [3.86, 4.77], H2 2.93% [2.55, 3.37], both Wilson CIs entirely inside [1%, 5%]. The pooled figure is not an averaging artifact.

**But the rate declines significantly between halves** — down roughly a third, with non-overlapping CIs, and compressed bars fall too (7,526 → 6,579). Event availability is not uniform across the flow window; the recent regime supplies fewer events on both counts. Recorded before B.5 so a thin recent-fold count is read as this known drift rather than a new defect.



`DEFAULT_LEVEL_TOLERANCE_ATR = 0.25` fires at 44.1% confirmation (S8) — a quarter-ATR tolerance matches most wicks. Sweep candidate values `{0.05, 0.10, 0.15}` and report confirmation rate for each.

**Select on confirmation rate landing in low single digits — a definitional target set in advance — never on downstream performance.** Additionally: apply ATR pct ≤ 0.20 as a precondition on sweep detection, and wire the "ATR ≤ 20th percentile" definition from the source document, which was never connected.

0.20 is a priori and **prohibited from re-tuning** (S6 retired).

### B.5 — Event count (the free kill test) — **GATE B FAILED 2026-08-18**

**100.0 effective events on the flow window against a threshold of 200.** Raw 156 → structural 100 → effective 100.0. Half the requirement; not marginal. H1 falsification condition 2 is met.

Conditions 3 and 4 were unblocked by pre-registration (`prereg/`, commit `a5a8fbd8`, committed **before** any count), a QQQ ingest, and a production caller in `conditions.py`. Cumulative filter: 78,328 bars → 20,200 compressed → 815 sweep → 800 macro (−1.8%) → **247 MTF (−69.1%)**. MTF ≥ 3/4 is the binding condition; macro is nearly inert because Risk-Off is an AND of three and occurs on 2.7% of bars — both predicted before counting.

**Under §6 this is a decisive negative**, so a revision responding to it is free. **Stopped at the gate.**

<details><summary>Superseded: the earlier BLOCKED status</summary>


**Two of the four conditions cannot be evaluated.** `macro_regime` is blocked on data (`btc_dominance_20d_trend_pct` and `btc_qqq_corr_20d` have no ingested source; both are legs of the Risk-On definition). `mtf_alignment` is blocked on specification (`score_mtf_alignment` consumes already-derived per-timeframe directional readings, and nothing derives them). Neither module has a caller outside its own test file. **This is S13 recurring** — the count of applied conditions has gone from 1 of 4 to 2 of 4.

**Upper bound measured instead.** Conditions only remove events, so the two-condition effective count bounds the gate number from above: **298.9 effective** on the flow window (516 raw → 326 structural → 298.9 effective, eff/raw 0.917, inside the 0.7–0.95 band predicted in advance).

**Gate B is INDETERMINATE — not passed, not failed.** The bound clears 200 so the decisive-negative branch does not fire, but the gate number does not exist. Conditions 3 and 4 together may remove no more than **33%** for the gate to hold; the structural + effective haircut alone already removed 42%.

**Two operator decisions block re-running it:** the condition-4 direction mapping (must be pre-registered — it determines the gate number), and the dominance leg of condition 3 (procure, drop-and-record, or restate — no free historical source exists; S23's "no paid data required" covered derivatives flow, not macro).



Build the fully-conditioned event set. All four conditions use data in hand:

```
atr_percentile ≤ 0.20
confirmed_sweep(tolerance = tuned in B.4)
macro_regime ∈ {risk-on, transitional}
mtf_alignment ≥ 3/4
```

**Independent event counting** per MASTER_CONTEXT §7 — structural collapse within a compression episode, then `Σ ū_i` over event label windows using the A.2 machinery. No new constant is introduced.

Report, over each of:
  - Flow window (2019-12-23 → 2026-08) — **this is the gate**
  - Funding-only window (2019-09-10 → 2026-08) — in case basis proves droppable
  - Full OHLCV window — reference only

and for each window, report:
  - **Four-condition effective count — the gate number**
  - Four-condition raw count
  - **Three-condition count (macro dropped) — diagnostic only**, so what macro costs in sample is visible
  - Count after each condition individually and cumulatively, so it is visible which condition does the filtering
  - ETH OHLCV-only count — information only, **not** part of any gate (MASTER_CONTEXT §9)

**The three-condition figure cannot be substituted for the gate number after the fact.** It exists to make the cost of the macro condition visible, not to provide a fallback.

</details>

### GATE B — kill criterion — **EVALUATED: FAILED**

| Effective event count | Action | |
|---|---|---|
| **< 200** | **H1 fails on sample grounds. Stop.** Decisive negative → free revision or terminate | ← **100.0 measured** |
| 200–400 | Proceed. Sample is thin | |
| > 400 | Proceed | |


Evaluated on the **effective, four-condition, BTC-only, flow-window** count.

| Effective event count | Action |
|---|---|
| **< 200** | **H1 fails on sample grounds. Stop.** Decisive negative → free revision or terminate |
| 200–400 | Proceed. Sample is thin; expect wide CIs at Stage C and budget for an ambiguous result |
| > 400 | Proceed |

**Purely an event count, with no procurement branch (revised 2026-08-15).** The vendor decision table is deleted: no paid data is required (S23), so there is nothing to buy and no window to buy narrowly. The gate is a sample-size kill test and nothing else.

**Do not loosen conditions to reach 200.** That inverts the gate into a target and voids the result. This includes: dropping the macro condition, adding ETH, relaxing the sweep tolerance chosen in B.4, and substituting the raw count for the effective one. **Data now being free does not soften this** — a free rescue after seeing the number inverts the gate exactly as a paid one would.

### B.2 — Free bulk acquisition (only after Gate B passes)

> **AWAITING OPERATOR TEXT.** The amendment directs "replace B.2 and B.3 with §3 and §4 above"; that source text was not supplied to this session. The obsolete vendor-evaluation content has been **deleted rather than left standing**, because S23 falsifies it and leaving it would make this plan contradict §2 and §9. What remains below are only the sub-task anchors that MASTER_CONTEXT already cross-references — they are placeholders, not a reconstruction of the operator's plan. **Do not treat this section as specified.**

No procurement. Sources are Binance Data Vision bulk dumps (MASTER_CONTEXT §9).

| # | Task | Status |
|---|---|---|
| B.2.2 | **Observation-timestamp semantics of `metrics`** (O5). Is `create_time` a point-in-time snapshot or a window aggregate? Determines whether I9 can be honoured; a wrong answer produces a Stage C false positive indistinguishable from success | Anchor only — awaiting text |
| B.2.4 | **Measure the `metrics` start date.** The flow window is bounded by whichever of perp–spot basis (2019-12-23) or this is later | Anchor only — awaiting text |
| B.2.6 | **Validate the OI-cascade liquidation proxy** against whatever `liquidationSnapshot` window exists | Anchor only — awaiting text |

Complete event-level liquidation history is the one genuine gap: `liquidationSnapshot` publishing reportedly stalled ~2024-03, and the live `!forceOrder@arr` stream is sampled rather than complete (S24), so any forward-collected series systematically undercounts. Both facts constrain what the proxy at B.2.6 can be validated against, and the undercount must be stated wherever a liquidation-derived feature is reported (I5).

### B.3 — Flow features

> **AWAITING OPERATOR TEXT** — see the note at B.2. The feature list below is retained from the prior plan because nothing has falsified it; only the *availability* premise changed. Treat the list as provisional against the operator's §4.

Funding and basis features are buildable now. OI, long/short ratio and taker buy/sell volume ratio become buildable once B.2 lands — **free, not vendor-gated (S23)**. Liquidation features depend on the B.2.6 proxy plus the partial `liquidationSnapshot` window. **I9 applies to every one.** All percentile-ranked against trailing 252-bar history per Layer 2 convention:

- Liquidation volume (long/short separately), z-scored vs trailing 20
- Liquidation cluster proximity: distance from price to nearest recent liquidation cluster
- OI change over 1/4/24 bars; OI-to-volume ratio
- Funding rate level and rate of change; funding extremity percentile
- Basis level and z-score
- Long/short account ratio and its rate of change
- Taker buy/sell volume ratio and its rate of change — **new, from the same `metrics` file**

**Rationale for prioritising liquidations:** H1 is a hypothesis about forced-flow cascades. Detecting them from the price wick they leave is inferring a cause from a lossy shadow of its effect. This is the single most likely source of information absent from OHLCV.

---

## Stage C — The Decisive Ceiling Test (1–2 weeks)

The highest-information experiment remaining. Run it before building anything else.

**Design:** on the fully-conditioned event set, under purged CV with uniqueness weighting, measure the multivariate ceiling four ways:

| Arm | Features | Sample | Purpose |
|---|---|---|---|
| C-1 | 39 OHLCV only | Event set | Baseline on events |
| C-2 | Flow only | Event set | Is flow independently informative? |
| C-3 | OHLCV + flow | Event set | The number that decides the project |
| C-4 | OHLCV + flow | All bars, flow window | **Added 2026-08-14.** Separates "flow adds nothing" from "conditioning adds nothing" |

Report AUC with corrected CI, logloss vs baseline, `n_eff`, and boost rounds before early stopping for each arm.

**Read the boost-round count as a primary diagnostic, not a footnote.** S4 and S18 established that 2–32 rounds means nothing to learn. If C-3 still halts under 40 rounds, that is a negative result regardless of what AUC says.

### Gate C — the project's main decision point, recalibrated 2026-08-14

The original "materially exceeds C-1" clause was written when C-1 was expected near 0.539. S5 is retired and C-1 is now expected near 0.518-and-noise, against which exceeding is easy and meaningless. **The clause is removed.**

| Outcome | Action |
|---|---|
| C-3 CI entirely above 0.5 **and** boost rounds > 40 | Pass → Stage D |
| C-3 CI above 0.5 but rounds < 40 | Negative. S4/S18 band — nothing to fit, regardless of AUC |
| C-3 CI includes 0.5, C-4 CI above 0.5 | Flow informative, conditioning is not. Revise the event definition, not the feature set |
| C-3 and C-4 CIs both include 0.5 | Decisive negative. Terminate or free revision |

**Marginal means stop.** A CI whose lower bound clears 0.5 by less than 0.005 is a boundary, not a finding — Expanding cleared by 0.0006 at A.6 and is not a signal. This is exactly the ambiguity the falsification budget exists to protect against.

---

## Stage D — Meta-Label Model (3–4 weeks)

Only reached if Gate C passes cleanly.

### D.1 — Primary rule (deterministic, no ML)

```
IF   atr_percentile ≤ 0.20
AND  confirmed_sweep(tolerance = tuned in B.4)
AND  macro_regime ∈ {risk-on, transitional}
AND  mtf_alignment ≥ 3/4
THEN emit {side, entry, stop, target}
```

No learned parameters. This is the event definition, and it must be frozen before D.2 begins.

### D.2 — Secondary model (the only learned component)

- **Input:** feature vector at event time + event metadata + flow features
- **Label:** binary — did this event reach target before stop?
- **Output:** `P(success)` → size via `m = 2Φ(z) − 1` where `z = (p − ½)/√(p(1−p))`, then hard-capped by Stage E risk limits
- **Training:** purged CV, uniqueness-weighted, trial-logged

**Required: stratified uniqueness weighting (S17).** The meta-label has the same duration/class correlation as the Stage 5 labels — losses resolve faster than wins, so loss labels overlap less, score higher uniqueness, and drag the weighted prior down (0.3114 vs 0.3369 at A.6). Left unhandled, this miscalibrates every probability the sizing formula consumes, and sizing is `m = 2Φ(z) − 1` on exactly those probabilities.

Fix: normalise `ū_i` to mean 1 **separately within each class** before training. Preserves the overlap correction, removes the prior shift. Report weighted and unweighted base rates side by side, as A.6 did.

Do not use post-hoc recalibration as the primary fix — it adds a fitted step to a pipeline that is already short on effective sample.

### D.3 — Measurement (I1 governs)

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

| Stage | Duration | Status |
|---|---|---|
| 0 — Teardown | 3–5 days | **Complete** |
| A — Apparatus | 2–3 weeks | **Complete** — both outstanding tests built and green; **Gate A pending operator sign-off** |
| B.0 — Hardening | 0.5 day + B.0.6 | **Next.** B.0.5a done; B.0.6 (FRED lag) blocking |
| B.1 — Free sources | — | **Complete** |
| B.4 — Sweep retune | 3–5 days | Unblocked |
| B.5 — Event count | 2–3 days | **Free kill test — Gate B** |
| B.2 — Free bulk acquisition | 3–5 days, no lead time | Only if Gate B passes |
| B.3 — Flow features | 1 week | Partially unblocked |
| C — Ceiling test | 1–2 weeks | **Main decision point** |
| D — Meta-label | 3–4 weeks | Only if C passes cleanly |
| E — Execution/risk/exits | 4 weeks | Only if D passes |
| F — Decision | — | — |

**Critical path to the free kill test: ~2 weeks.** Critical path to the decisive answer: **4–7 weeks from Gate B passing**, shortened by roughly 1–2 weeks now that B.2 is a 3–5 day bulk download with no procurement lead time (S23). The cheap "no" still arrives first, and the expensive half turned out not to be expensive.
