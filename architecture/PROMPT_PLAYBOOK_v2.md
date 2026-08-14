# PROMPT PLAYBOOK v2

Session-by-session prompts for the agent. Ordered to match BUILD_PLAN_v2.md.

**How to use:** one prompt per session. Prepend the Standard Preamble to every one. Do not batch prompts — each ends at a verification point where you need to read the output before continuing.

---

## Standard Preamble

Prepend to **every** session prompt:

```
Read MASTER_CONTEXT_v2.md and project_state.md before doing anything else.

State at the start of your response:
  - which Stage and Gate this work belongs to
  - which Invariants (§5) are live for this task
  - what you believe the current repo state is, in one line

Rules for this session:
  - Do not proceed past a Gate. Report gate status and stop.
  - Verify mechanisms; do not assert them (I4). If a result is
    surprising, inspect the cause before reporting it as a finding.
  - Report effective n alongside nominal n for any measurement (I6).
  - If a hypothesis condition is untestable because data is missing,
    say so explicitly in the report (I5). Never omit it silently.
  - Nothing is committed without the test suite green.
  - Log any model configuration you evaluate to trials.jsonl.

End with: what landed, what is deferred and why, and gate status.
Update project_state.md.
```

---

## Behavioural Notes for the Agent

Include this block when a session involves measurement or interpretation:

```
Calibration guidance for this project:

Your previous sessions did something rare and correct: when the Layer 5
ablation returned a byte-identical result, you formed a hypothesis, then
inspected the fitted weights to confirm it rather than reporting it as a
finding. You also stated plainly that the automatic Sharpe-only verdict
had identified the wrong layer. That is the standard.

The failure mode to avoid is the opposite: producing a result that looks
like progress. This project has measured its own predictive core as
worthless. That was the correct output. If your work this session
produces another negative result, report it plainly and do not soften it,
reframe it, or propose tuning to make it go away.

Specifically: do not optimise AUC, do not tune toward Tier 1 numbers, and
do not report Sharpe over a destroyed equity curve.
```

---

# Stage 0 — Teardown

### Prompt 0.1 — Delete the predictive core

```
Stage 0 teardown. Per MASTER_CONTEXT_v2 §8 (Component Register), delete:

  - HistoricalAnalogEngine (Component A / k-NN)
  - Component B direction classifier in its current form
  - EnsembleMetaModel and fit_ensemble_weights
  - class-weighting machinery built for the 65/34 imbalance
  - calibration work premised on that imbalance

These are measured useless (S7, S1) and one of them destroyed the account
in ablation. Remove them; do not deprecate or comment out.

Then: grep the repo and confirm zero remaining references outside git
history. Report anything that referenced them and what you did with the
caller.

Prune tests that only covered deleted components. Report suite count
before and after.
```

### Prompt 0.2 — Stub the risk caps and fix Layer 6

```
Two changes.

1. Remove Phase2Strategy.max_position as a risk mechanism. Per S10 it is
   a model-layer parameter that caps size not risk, and the 6% and 10%
   caps have no implementation at all. Replace with a stub that raises
   NotImplementedError so nothing can size without real caps present.
   Add a test asserting that sizing without caps fails loudly.

2. Update the Layer 6 compression threshold from 0.30 to 0.20 per S6.
   Single-source the constant — report if it was duplicated anywhere.

Gate 0 status when done.
```

---

# Stage A — Measurement Apparatus

### Prompt A.1 — Label metadata

```
Stage A.1. The triple-barrier labeller must emit per sample:
  t0, t1 (barrier-touch or vertical-barrier time), barrier_touched
  ∈ {tp, sl, vertical}, realised_return.

t1 is the critical addition — purging is impossible without it, and its
absence is why the current temporal partitioning leaks (it partitions on
feature timestamps, not label-resolution timestamps).

Backfill t1 for the existing labelled dataset. Report the distribution of
label durations (t1 − t0) in bars: median, p90, max. I want to see how
much overlap we actually have before you build the uniqueness code.
```

### Prompt A.2 — Uniqueness and effective n

```
Stage A.2. Implement sample uniqueness per AFML ch. 4:

  indicator   1[t,i] = 1 if label i's window spans bar t
  concurrency c_t    = Σ_i 1[t,i]
  uniqueness  u[t,i] = 1[t,i] / c_t
  avg uniq    ū_i    = Σ_t u[t,i] / Σ_t 1[t,i]
  effective n n_eff  = Σ_i ū_i

Wire ū_i as training sample weight. Add n_eff to every result object so
it reports alongside nominal n (I6).

Then report, for the full 78,378-row dataset and for the compressed
(ATR pct ≤ 0.20) subset: nominal n, n_eff, and the ratio.

I expect n_eff to be a small fraction of nominal. Report the actual
number without editorialising about what it implies — the implications
come at A.6.
```

### Prompt A.3 — Purged K-Fold with embargo

```
Stage A.3. Implement purged K-fold with embargo, replacing the current
walk-forward splitter. Keep the per-fold reporting structure.

  Purge:   drop training samples whose [t0, t1] overlaps the test fold
  Embargo: additionally drop training samples with t0 in
           [T1, T1 + h], h = 0.01 × total observations

Then verify it with a synthetic leakage test: construct a feature that
deliberately leaks the label, and confirm the OLD splitter shows
inflated AUC while the NEW one does not. Report both numbers.

Do not tell me it works. Show me the leak test failing on the old
splitter and passing on the new one.
```

### Prompt A.4 — Harness defects

```
Stage A.4. Four fixes, all with known causes from the prior audit:

1. equity_curve_from_trade_pnls — floor equity at 0, set equity_destroyed.
   Currently init_cash + cumsum(), unbounded below (S11).

2. Two accounting books (S12). vectorbt sizes off an equity that never
   sees gap costs — 20× divergence measured on fold 2. Reconcile to a
   single equity source of truth. All costs, including gap, applied
   inside the book that sizing reads (I7).

3. Gap cost is n_bars × freq × mag × notional — per-bar, linear,
   uncapped, and it can exceed position value. It was 96% of fold 2's
   loss. Cap at position notional. Then re-derive the per-bar linearity
   assumption and document whatever replaces it.

4. Ablation flagging is still Sharpe-only. Gate on maxDD and PF before
   Sharpe; refuse to rank on Sharpe when any_equity_destroyed (I3).

Verify by reproducing fold 2 of all_layers end-to-end and confirming the
sizing-book and reported maxDD now agree. Show both numbers.
```

### Prompt A.5 — Trial registry and DSR

```
Stage A.5. Two pieces of the falsification discipline.

1. trials.jsonl — one line per evaluated model configuration:
   {config_hash, timestamp, stage, params, n_eff, result_metrics}.
   Wire Optuna to auto-log. Manual runs log too.

2. Implement deflated_sharpe_ratio(returns, n_trials, skew, kurtosis)
   per Bailey & López de Prado.

Then backfill trials.jsonl as far as the existing logs allow, and report
how many distinct configurations this project has already evaluated. If
the true count is unrecoverable, say so and give a documented lower
bound — an underestimate that is labelled as an underestimate is fine,
a guess presented as a count is not.
```

### Prompt A.6 — Recompute Stage 5

```
Stage A.6. Re-run the Stage 5 ceiling measurements under the corrected
apparatus: purged CV, uniqueness weighting, corrected CIs.

Reproduce the full Stage 5 table — unconditioned, compressed <0.30,
compressed ≤0.20, expanding, extended — with columns: nominal n, n_eff,
AUC, corrected 95% CI, logloss vs baseline, boost rounds.

Stated in advance so it is not read as failure: I expect the CIs to widen
substantially and the compressed-regime CI may now straddle 0.5. That is
the honest number and it is a likely outcome, not a bug. Report it
exactly as measured.

This is Gate A. Gate A tests apparatus correctness, NOT whether the
compressed finding survives. Report gate status on the apparatus
criteria, and report the S5 survival result separately as an input to
Stage C. Do not terminate the project on this result.
```

---

# Stage B — Data Acquisition

### Prompt B.1 — Free sources

```
Stage B.1. Ingest, using the existing Layer 1 quality pipeline (gap
classification, duplicate/clock-skew removal, outlier detection):

  - Funding rates: Binance /fapi/v1/fundingRate (full history, 8h)
  - Macro: FRED — DXY, US10Y, VIX (unblocks hypothesis condition 3)
  - 1D and 1W bars via the existing OHLCV pipeline (unblocks condition 4)
  - Perp–spot basis: computed from spot + mark price

For each: report coverage window, gap count by classification, and row
count. Flag any source whose coverage is shorter than the BTC 1h window.
```

### Prompt B.2 — Vendor evaluation

```
Stage B.2. Liquidations, OI history, and long/short ratios beyond ~30
days need a paid vendor. This is the longest lead-time item in the
project.

Evaluate Tardis, Coinglass, Amberdata, and Kaiko on:
  - history depth for BTC perps (note: Binance BTC perps start ~Sep 2019,
    so ~6 years is the ceiling regardless of vendor)
  - liquidation granularity: per-event or aggregated, and at what interval
  - OI interval available historically
  - long/short ratio history
  - cost per month at our volume

Produce a comparison table and a recommendation with reasoning. Do not
sign anything up — give me the table.
```

### Prompt B.3 — Flow features

```
Stage B.3. Build flow features, percentile-ranked against trailing
252-bar history per the Layer 2 convention:

  - liquidation volume, long and short separately, z-scored vs trailing 20
  - liquidation cluster proximity: distance from price to nearest recent
    cluster
  - OI change over 1, 4, 24 bars; OI-to-volume ratio
  - funding rate level and RoC; funding extremity percentile
  - basis level and z-score
  - long/short account ratio and RoC

Rationale to keep in mind: H1 is about forced-flow cascades. We have been
detecting them from the price wick they leave behind. These features
measure the cause directly.

Report each feature's coverage window and the joint window where all are
available — that joint window is the real dataset size for Stage C.
```

### Prompt B.4 — Sweep detector

```
Stage B.4. Re-specify the sweep detector. Current state: 44.1%
confirmation rate vs 2.04% for the earlier proxy; ensemble weighted it
exactly 0.000000 (S8).

Three named suspects:
  - DEFAULT_LEVEL_TOLERANCE_ATR = 0.25 is likely far too loose; a
    quarter-ATR tolerance matches most wicks
  - no compression precondition is applied
  - the "ATR ≤ 20th percentile" definition from the source doc was
    never wired in

Sweep tolerance over {0.05, 0.10, 0.15}. Add ATR pct ≤ 0.20 as a
precondition on sweep detection.

IMPORTANT: select on confirmation rate landing in low single digits.
That is a definitional target set in advance. Do NOT select on
downstream performance — that would be tuning the event definition to
the outcome, which is the overfitting path.

Report confirmation rate and event count for each tolerance value.
```

### Prompt B.5 — Event set and Gate B

```
Stage B.5. Build the fully-conditioned event set:

  atr_percentile ≤ 0.20
  AND confirmed_sweep (tolerance from B.4)
  AND mtf_alignment ≥ 3/4
  AND macro regime ∈ {risk-on, transitional}

This is the first time all four hypothesis conditions have ever been
applied together (S13).

Report independent event count — enforcing minimum temporal separation
so two events in the same continuous directional move count as one —
separately for:
  - the 9-year OHLCV window
  - the ~6-year flow-data window

Gate B kill: fully-conditioned independent event count < 200 over the
flow window means H1 fails on sample grounds. If that happens, report it
plainly and stop. Do not loosen conditions to reach 200 — that inverts
the gate into a target.
```

---

# Stage C — The Decisive Test

### Prompt C.1 — Three-arm ceiling test

```
Stage C. This is the decisive experiment of the project. Run it exactly
as specified and do not extend it.

On the fully-conditioned event set, under purged CV with uniqueness
weighting, measure the multivariate ceiling three ways:

  C-1: 39 OHLCV features only     (baseline on events)
  C-2: flow features only          (is flow independently informative?)
  C-3: OHLCV + flow                (the number that decides the project)

Report per arm: AUC, corrected 95% CI, logloss vs baseline, n_eff, and
boost rounds before early stopping.

Read boost rounds as a PRIMARY diagnostic, not a footnote. S4 established
that 2–32 rounds means there is nothing to learn. If C-3 halts under 40
rounds that is a negative result regardless of AUC.

Gate C:
  - C-3 CI entirely above 0.5 AND materially exceeds C-1 AND rounds > 40
    → pass, proceed to Stage D
  - C-3 CI above 0.5 but no lift over C-1
    → flow adds nothing; H1 falsification condition 5 met
  - C-3 CI includes 0.5
    → decisive negative

A MARGINAL result at Gate C means stop, not proceed. Marginal is exactly
the ambiguity that consumes falsification budget. Report the gate
outcome and stop — do not begin Stage D in the same session regardless
of the result.
```

---

# Stage D — Meta-Label Model

### Prompt D.1 — Freeze the primary rule

```
Stage D.1. Implement the primary rule as a deterministic function with
NO learned parameters:

  IF atr_percentile ≤ 0.20
  AND confirmed_sweep(tolerance frozen from B.4)
  AND mtf_alignment ≥ 3/4
  THEN emit {side, entry, stop, target}

Freeze it. Commit the hash. It must not change during D.2 — if the event
definition moves while the model trains, nothing measured afterwards is
interpretable.

Report the frozen commit hash in project_state.md.
```

### Prompt D.2 — Secondary model

```
Stage D.2. Train the meta-label classifier — the only learned component
in the system.

  Input:  feature vector at event time + event metadata + flow features
  Label:  binary — did this event reach target before stop?
  Train:  purged CV, uniqueness-weighted, every config logged to
          trials.jsonl
  Output: P(success) → size m = 2Φ(z) − 1, z = (p − ½)/√(p(1−p))

Report, in this order of priority:
  1. top-decile realised EV after 75th-percentile costs (PRIMARY)
  2. lift over base rate at operating threshold
  3. calibration on the event set
  4. n_eff
  5. AUC — reported, NOT optimised (I1)

Note the base rate here reflects setup quality, not barrier arithmetic —
that is the whole point of the meta-label reformulation. Report what it
actually is and compare it to the sl/(tp+sl) value for the same geometry.

Gate D: top-decile realised EV > 0 after costs, out-of-sample, ≥ 200
independent events.
```

---

# Stage E — Execution, Risk, Exits

### Prompt E.1 — Real risk caps

```
Stage E.1. Build the execution-layer risk caps. Per S10 these have never
existed — the 6% and 10% constraints have zero implementation.

  2%  max single-trade RISK (not size — requires stops from E.2)
  6%  max correlated-direction exposure, portfolio level
  10% max aggregate open risk

Enforced BELOW the model layer, on the order path, where no model output
can raise them.

Test: assert a deliberately malicious model output demanding 50% size is
clamped to the cap. Test: assert the 6% cap triggers on two correlated
positions. These tests must fail if the caps are ever removed.
```

### Prompt E.2 — Exit engine

```
Stage E.2. Build Layer 13. There are currently no stops at all, which is
why the size cap was never a risk cap.

Five exit types: target achieved, forecast deterioration (15pp drop from
entry, fixed at entry and non-adjustable), risk increase, opportunity
cost (new setup EV lower-CI ≥ 50% higher than current recalculated EV),
time decay (p50 expected duration elapsed).

Log full feature state at every exit decision — this becomes the dataset
for backtesting exit logic independently of entry logic.
```

### Prompt E.3 — Full-system backtest

```
Stage E.3. Full-system backtest: complete system, purged CV, realistic
costs at 75th-percentile spread, fixed-fractional sizing only. No Kelly
at any stage of this test.

Report v4 Tier 1 all six, plus DSR adjusted for the full trials.jsonl
count.

Gate E: all six Tier 1 gates pass on untouched out-of-sample data AND
DSR > 0.5.

These are gates, not targets (I2). If they fail, report the failure. Do
not tune toward them — any result obtained that way is void and would
have to be re-earned on data we no longer have.
```

---

## Prompts to Never Send

Recorded because the pull toward them will be strongest exactly when results are worst:

- *"Try to get AUC above X"* — violates I1. The number is not the objective.
- *"Tune until Sharpe clears 1.0"* — violates I2 and voids the result.
- *"Loosen the conditions so we get enough events"* — inverts Gate B into a target.
- *"Try a few more model architectures and see what works"* — unlogged trial inflation; destroys DSR validity.
- *"Ignore the drawdown for now and look at Sharpe"* — violates I3.
- *"That result seems wrong, run it again differently"* — this is how a decisive negative becomes a marginal one.

If a session produces a negative result and the instinct is to send one of these, the correct action is to open the revision pre-registration template in BUILD_PLAN_v2.md instead.
