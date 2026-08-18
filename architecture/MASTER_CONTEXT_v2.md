# MASTER_CONTEXT v2

**Status:** Active. Supersedes MASTER_CONTEXT v1 in full.
**Reset point:** End of Phase 2, following Stage 5 ceiling measurement.
**Load this document at the start of every session before any other file.**

---

## Changelog

| Date | Trigger | Change |
|---|---|---|
| 2026-08-18 (b) | **Gate B evaluated — H1 fails on sample grounds** | Conditions 3 and 4 pre-registered and committed (`a5a8fbd8`) before any count; QQQ ingested; both condition modules given production callers. **Gate B: 100.0 effective vs 200 required.** O3 resolved NO; S29–S32 added. Falsification budget 2 → 1 (condition 3 restatement). **I11 added** — a module with no caller outside its own test file is not a built condition. Frozen dataset **v3 `913f80f490520e7f…`**. |
| 2026-08-18 | Stage B.0 hardening complete | Dataset frozen (**v2, `a5f81b5b8585d982…`**) — §10 item 4 is now executable. I9 and I10 given implementations (`point_in_time.py`, `acquisition.py`) rather than conventions. FRED publication lag modelled at a one-business-day floor (B.0.4), which **changed macro alignment**: a value for date D is no longer visible on D's own bars. **Base currency is GBP** (B.0.6, DEXUSUK) — §7 gains an FX row, and Tier 1 gates now read the GBP curve. UUP deleted at the route, not just the config. S26–S28 added. |
| 2026-08-15 (b) | Binance Data Vision source review | **No paid data is required (S23).** OI, long/short ratios and taker volume ratio are free via Data Vision `metrics` bulk dumps — the REST 30-day limit binds only the API path. §9 vendor rows replaced with six free sources; hard constraint rewritten around the `metrics` start date (measured at B.2.4) rather than a purchase. Event-level liquidation history is now the single genuine gap, handled by an OI-cascade proxy plus forward-only collection. S24 records that the live `!forceOrder@arr` stream is *sampled*, not complete. O5 retargeted to `create_time` semantics at B.2.2. **The outlier-flag finding was renumbered S23 → S25** to free the operator-assigned numbers. |
| 2026-08-15 | A.6 corrected ceiling measurement; B.1 free-source ingestion; operator rulings 1–7 | Amendments applied inline. S5/S6 retired to the Retired Findings table (**not deleted** — the 0.5390 figure exists in the logs and a session rediscovering it must find the retirement note). S14–S22 added. §3 Open Questions, §9 Data Inventory replaced. §4 H1 restored to **four** conditions — `macro_regime` was lost in a drafting error, not narrowed deliberately; restoring it is free and consumes no budget. I9, I10 added. Independent-instance definition replaced with the structural + effective-count rule. FRED publication lag promoted to **blocking for B.5**, since macro now gates event formation. DTWEXBGS confirmed as the dollar measure; UUP path to be deleted. |

---

## 1. Project Identity

A single-hypothesis research project testing whether a specific, conditional, event-driven edge exists in crypto perpetual futures. It is **not** a trading bot project until a gate says it is. The deliverable of every stage before Stage F is a **measurement**, not a feature.

The project's most valuable asset is its measurement apparatus, not its models. Protect that ordering in every decision.

---

## 2. Settled Findings — Do Not Re-Litigate

These are measured, verified, and closed. Any session that contradicts one of these is wrong and must stop and re-read this section. Any session that proposes re-deriving one of these is wasting budget.

| # | Finding | Evidence |
|---|---|---|
| S1 | The 65/34 class imbalance is barrier arithmetic (`sl/(tp+sl)`), not market structure | Four geometries, nine years, observed within 0.4–1.1pp of driftless-walk prediction at every one |
| S2 | Directional signal on OHLCV-derived features alone is ~0.509 AUC — noise | Symmetric-barrier test; early stopping at 3 rounds |
| S3 | The multivariate ceiling on 39 OHLCV features is 0.5251 [0.514, 0.536] and barely exceeds the best single feature (0.5172) | There is nothing to combine; feature engineering on OHLCV is exhausted |
| S4 | Early stopping halts at 2–32 rounds in every configuration tested | A model finding real structure trains for hundreds |
| S7 | Component B has no measurable skill (0.004–0.016) yet carried 0.942 ensemble weight | Fitted-weight inspection on real data |
| S8 | Layer 5 sweep detector fires at 44.1% confirmation vs 2.04% for the earlier proxy; ensemble weighted it exactly 0.000000 | The optimizer correctly ignored noise |
| S9 | Layer 6 gate blocks 61 of 35,069 bars (0.174%) — it is untested, not degrading | A layer touching 0.17% of bars cannot move Sharpe by 3.4 |
| S10 | There are no execution-layer risk caps in the codebase. The 2%/6%/10% constraints appear only in scoring logic and comments | Grep-verified. The 6% and 10% caps have no implementation whatsoever |
| S11 | Backtest equity has no bankruptcy floor; gap cost is charged per-bar, linearly, uncapped, and can exceed position notional | Reproduced: gap cost was 96% of fold-2 loss |
| S12 | The harness keeps two accounting books — vectorbt sizes off an equity that never sees gap costs (20× divergence measured) | Fold-2 reproduction: sizing book maxDD 0.013 vs reported 0.270 |
| S13 | The hypothesis has **never been tested.** Layers 3 and 4 require data that is not ingested; Layer 5 is unproven. Only 1 of 4 conditions has ever been applied | Measurement was run on all 78,378 bars |
| S14 | No regime gradient exists. All five subsets fall in AUC 0.5095–0.5244 with CIs spanning ~0.035 — statistically indistinguishable. The unconditioned set has the only clean CI purely because its n_eff is 2–3× larger | A.6 corrected table |
| S15 | Purged CV closed a leakage channel worth 0.016 AUC (0.5184 → 0.5021 on the synthetic leak test) | A.3 verification |
| S16 | n_eff / nominal ratio is 0.145–0.218 across subsets. Nominal n overstates information content by 5–7× | A.6 |
| S17 | Uniqueness weighting shifts the class prior: weighted base rate 0.3114 vs unweighted 0.3369. Cause is duration/class correlation — `sl` resolves in a median 4 bars vs `tp` at 7, so down-labels overlap less and take more weight. AUC is rank-based and unaffected; logloss is not | A.6 weighting toggle |
| S18 | Boost rounds halt at 6.8–20.8 across every corrected subset — inside S4's "nothing to fit" band | A.6 |
| S19 | ICE DXY is proprietary and not on FRED. DTWEXBGS (broad, annually reweighted, heavier EM) is the selected substitute — a different index, not a proxy | B.1 |
| S20 | The outlier flag is not timeframe-aware: 0.1% / 8.3% / 58.4% on 1h / 1d / 1w. The 1d and 1w outlier columns are unusable until the threshold is timeframe-scaled | B.1 |
| S21 | Binance treats `startTime=0` as unset, silently returning 500 recent rows instead of full history with no error. Pagination additionally breaks on short pages returned mid-history. Both fixed; 7,592 funding rows recovered vs 500 | B.1 |
| S22 | Derivatives-flow window is bounded by perp–spot basis at 2019-12-23 — 58,198 of 78,666 bars (74%). Funding starts earlier (2019-09-10); all three FRED series cover the full reference window | B.1 |
| S23 | **No paid data is required.** Binance publishes OI, long/short ratios, and taker volume ratio free via Data Vision `metrics`; the REST 30-day limit applies only to the API path, not the bulk dumps | Verified 2026-08-15 |
| S24 | The `!forceOrder@arr` liquidation stream is sampled, not complete — Binance pushes only the largest order per 1000ms per symbol. Any live-collected liquidation series systematically undercounts | Binance API docs |
| S26 | **The account is GBP-denominated (B.0.6).** Percentage risk caps are currency-invariant — the FX rate cancels, so 2/6/10% are unchanged. Gate metrics are **not**: Sharpe, maxDD and PF all move because FX variance enters the return series. A flat USD account under a strengthening pound carries a real GBP drawdown | `backtesting/currency.py`, measured in test |
| S27 | FRED DEXUSUK is **USD per one GBP**, so USD→GBP is a DIVISION. Reversing it is wrong by ~1.6² and still looks plausible on a chart | B.0.6 |
| S28 | The pre-B.0.4 macro join treated a FRED observation as knowable on the date it describes. Correcting it to a one-business-day floor **removes macro from the first bars of every value date** — a real reduction in usable macro coverage, not a cosmetic shift | B.0.4 |
| S29 | **GATE B FAILS.** The fully-conditioned event set contains **100.0 effective independent events** over the flow window against a threshold of 200 — half the requirement, not a marginal miss. Raw 156 → structural 100 → effective 100.0. Falsification condition 2 is met | B.5, frozen v3, conditions pre-registered at commit `a5a8fbd8` |
| S30 | **MTF alignment ≥ 3/4 is the binding filter, removing 69.1%** of compression+sweep events. Macro removes 1.8%. The alignment distribution over scored sweep bars is 0:321, 1:551, 2:534, 3:506, 4:235 — three-of-four agreement is genuinely uncommon, as forecast before counting | B.5 |
| S31 | Risk-Off is rare — 2,088 of 78,328 bars (2.7%). It requires an AND of three conditions including `btc_qqq_corr_20d >= 0.7`. Condition 3 therefore admits 96.79% of bars and was never going to be a material filter. Predicted in `prereg/condition_3_macro.md` **before** the count | B.5 |
| S32 | Event label windows do not overlap at this sparsity: eff/raw = **1.000** on the flow window, **above** the 0.7–0.95 band recorded in advance. Median label horizon is 5 bars against a mean event spacing of 582. The structural collapse does all the reduction; temporal concurrency contributes almost nothing | B.5 |
| S25 | The outlier flag is **written and never read.** Exactly two references exist: one write (`crypto_fetcher.py:333`) and one docstring noting it is ignored (`dataset.py:75`). No persisted parquet carries the column | B.0.5a audit, 2026-08-15 |

### Retired Findings — Leakage Artifacts

These were Settled Findings in v1/v2 and are now falsified by corrected measurement. They are kept visible so that a future session encountering the old numbers in a log does not treat them as live.

| Retired | Original claim | Corrected | Cause |
|---|---|---|---|
| **S5** | Signal present under compression (AUC 0.5390, CI [0.519, 0.559]), absent under expansion | AUC 0.5183, CI [0.4986, 0.5374] — CI includes 0.5 | Leakage. Purged CV closed a channel measured at 0.016 AUC (0.5184 → 0.5021); the compressed point estimate fell 0.021 |
| **S6** | Threshold 0.20 beats 0.30 (0.5390 vs 0.5238) | 0.5183 vs 0.5139 — difference well inside noise | Same leakage channel |

**Do not re-derive, re-test, or cite either.** The 0.20 threshold survives in H1 on a priori grounds only — see §4 note.

---

## 3. Open Questions — Not Yet Settled

| # | Question | Status | Resolves at |
|---|---|---|---|
| O1 | Does the compressed-regime signal survive correction? | **RESOLVED — NO.** See Retired Findings | Closed at A.6 |
| O2 | Does derivatives-flow data lift the ceiling above OHLCV-only? | Open. **Now the sole remaining support for H1** | Stage C |
| O3 | Does the fully-conditioned event set contain ≥ 200 independent instances over the flow window? | **RESOLVED — NO. 100.0 effective, against a threshold of 200.** Gate B fails on sample grounds (S29) | Closed at B.5, 2026-08-18 |
| O4 | Can a defensible sweep definition reach low-single-digit confirmation rate without starving event count? | Open | Stage B.4 |
| O5 | Do the `metrics` series have observation-timestamp semantics compatible with I9? Specifically: is `create_time` a point-in-time snapshot or a window aggregate? A wrong answer produces a Stage C false positive indistinguishable from success | Stage B.2.2 |

---

## 4. The Hypothesis (H1)

> Among BTC and top-liquidity USDT perpetual futures on the 1H timeframe, entries generated by a **deterministic** post-compression liquidity-sweep rule (ATR percentile ≤ 0.20 · confirmed sweep · `macro_regime ∈ {risk-on, transitional}` · MTF alignment ≥ 3/4) can be filtered by a **meta-label classifier** using derivatives-flow features (liquidations, open interest, funding, basis) such that the filtered subset carries positive expected value after 75th-percentile costs, with **top-decile realised EV > 0** and **Deflated Sharpe Ratio > 0.5** over **≥ 200 independent out-of-sample events**.

**Falsification conditions.** H1 fails if any of:

1. Corrected compressed-regime CI includes 0.5 **and** flow features fail to produce a CI entirely above 0.5 (O1 + O2 both negative)
2. Fully-conditioned event count < 200 independent instances
3. Top-decile realised EV ≤ 0 after 75th-percentile costs
4. Deflated Sharpe Ratio ≤ 0.5 out-of-sample
5. Flow features add no measurable lift over OHLCV-only on the same event set

**Note on the macro condition (restored 2026-08-15).** The source hypothesis has **four** conditions and S13 counts four. The macro-regime term was dropped from this section's v2 rewrite by drafting error, not by deliberate narrowing — it survived intact in S13, O3, and the playbook's B.5 prompt. Restoring it is a correction, not a revision, and **consumes no falsification budget**. Ruling the other way would have been a hypothesis revision and would have cost budget.

**Note on the 0.20 threshold (added 2026-08-14).** S6 is retired; the measured justification for 0.20 over 0.30 no longer exists. The threshold remains in H1 on *a priori* grounds — it is the value stated in the original hypothesis text, not a value selected by fitting. It is therefore legitimate as an untested condition and **prohibited from re-tuning**. Any threshold chosen by inspecting the A.6 table would be fitted to noise.

**H1 is not revised by the A.6 or B.1 results.** It has lost supporting evidence, not been falsified. Budget remains 2 of 3.

---

## 5. Non-Negotiable Invariants

**I1 — EV is the objective. AUC is a diagnostic.**
AUC may be reported. It may never be optimised, targeted, or used as a gate. No decision is justified by an AUC improvement alone.

**I2 — Tier 1 metrics are gates, never objectives.**
Sharpe > 1.0, PF > 1.5, maxDD < 20% grant permission to deploy capital. They are not forecasts and not optimisation targets. Any result achieved by tuning toward them is void and must be re-earned on untouched data.

**I3 — No Sharpe over a non-positive equity curve.**
`SegmentResult.equity_destroyed` forces Sharpe to NaN. Ranking, flagging, and reporting must all respect this. Ablation flagging gates on maxDD and PF **before** Sharpe.

**I4 — Verify mechanisms; do not assert them.**
An anomaly is reported as a finding only after its cause has been inspected directly. The prior session's handling of the zero-weight Layer 5 result — hypothesising, then inspecting fitted weights to confirm — is the required standard.

**I5 — Untestable is stated, never omitted.**
If a hypothesis condition cannot be evaluated because its data is not ingested, every report says so explicitly. Silent omission of an untestable condition is how S13 went unnoticed for two phases.

**I6 — Effective n, always.**
Every result reports effective sample size alongside nominal. Nominal n is fictional under overlapping labels.

**I7 — One equity, one book.**
Position sizing reads the same equity object the results are computed from. No cost may be applied outside the book that sizing sees.

**I8 — Delete before adding.**
When a component is measured as useless, it is removed, not retained "in case." Dead components consume attention and reappear in ensembles.

**I9 — Point-in-time discipline.**
A feature at bar *t* may use only information observable at or before bar *t*'s close. Back-filling a settled value into the interval it settles over is lookahead that **no splitter can catch**, because it is baked into the feature rather than the split. Pre-observation bars are NaN, never 0. Every series declares its observation timestamp separately from its value timestamp. Publication lag is modelled, not assumed away.

**I11 — A module with no caller outside its own test file is not a built condition.**
Build plans verify **wiring**, not existence. `classify_macro_regime` and `score_mtf_alignment` passed their own tests for two phases while being unreachable from any production path, and B.5 discovered at the kill test that two of four hypothesis conditions had never been applied. A green test suite over an orphaned module is evidence about the module, not about the system. Every condition module carries a test that fails if its production caller disappears.

**I10 — Acquisition is verified, not trusted.**
Every ingest run asserts returned row count and date range against what was requested, and hard-fails on mismatch. `startTime=0` returned 500 rows for 7,592 with no error (S21); assume every new API has its own version of that bug.

---

## 6. Falsification Budget

**Remaining: 1 of 3.**

| # | Revision | Consumed? | Reason |
|---|---|---|---|
| 1 | The v5 restatement | yes | — |
| 2 | **Condition 3 restated: BTC-dominance leg removed (2026-08-18)** | **yes** | A leg of the Risk-On definition had no free historical source. Unavailability is the *marginal/ambiguous* case under the rules below, not a decisive negative. Pre-registered at `prereg/condition_3_macro.md` before any count |

**Recorded against revision 2:** `dominance_expanding` appears only in `risk_on`, never in `risk_off`, and H1 admits `{risk-on, transitional}` = **not Risk-Off**. Removing the leg can only move bars from Transitional to Risk-On, both of which pass, so **the admitted event set is provably identical** — verified by test, not asserted. The decrement is applied as ruled; this note exists so the ruling can be reversed on full information. Spending a revision one may not owe is the conservative error.

**Rules:**

- A revision is **free** if it follows a *decisive negative* — a result whose CI excludes the hypothesis. Responding to clean falsification is science.
- A revision **consumes budget** if it follows a *marginal or ambiguous* result. Iterating on ambiguity is how hypothesis space gets searched until noise looks like signal.
- Every revision is **pre-registered before any data is examined under it**, using the template in the Build Plan.
- At zero remaining budget, the project terminates. No exceptions, no "one more idea."

**Correcting a drafting error is not a revision.** Restoring a condition that the source hypothesis, S13, and the playbook all carry costs nothing (see §4). Removing one would have cost budget. The asymmetry is deliberate: it must never be cheaper to quietly narrow a hypothesis than to test it.

**Trial registry.** Every distinct model configuration evaluated is logged to `trials.jsonl` with config hash, date, and result. This count feeds the Deflated Sharpe Ratio. Optuna trials count. Manual runs count. You cannot compute an honest DSR without an honest trial count, and DSR is a gate.

---

## 7. Canonical Metric Definitions

Use these exact definitions everywhere. Discrepancies between modules are silent bugs.

| Metric | Definition |
|---|---|
| **EV** | `P(win) × E[win magnitude] − P(loss) × E[loss magnitude] − costs`, costs at 75th-percentile historical spread + taker fees + size-dependent slippage |
| **Top-decile realised EV** | Mean realised P&L of events in the top 10% of meta-model score, after costs, out-of-sample |
| **Effective n** | `Σ ū_i` where `ū_i` is average uniqueness of label i from the concurrency indicator matrix |
| **IC** | Approximated as `2 × (AUC − 0.5)` for reporting; measure directly as Spearman(score, forward return) where possible |
| **DSR** | Deflated Sharpe Ratio (Bailey & López de Prado), adjusted for trial count from `trials.jsonl`, skew, kurtosis, sample length |
| **maxDD** | `(equity − running_max) / running_max`, on a floored equity curve. Values **> 1.0** are impossible by construction post-fix; if seen, it is a bug. **Corrected 2026-08-18:** this previously said "≥ 1.0", which is wrong — total ruin lands on exactly 1.0 and is attainable, not a defect. Measured directly under the mutation-test fixture |
| **Independent instance** | **Two rules, both pre-specified (2026-08-15), neither introducing a new constant.** *Structural:* two events in the same compression episode (an unbroken run of ATR pct ≤ 0.20) in the same direction count as one — this operationalises "same continuous directional move" directly. *Temporal:* concurrency is computed over event label windows `[t0, t1]` with the A.2 machinery, and **effective event count = `Σ ū_i`**. Gate B is evaluated on the effective count; raw is reported alongside. A fixed bar-separation constant was rejected: it would be arbitrary and chosen in the knowledge that it determines the gate |
| **Observation timestamp** | The time at which a value first became knowable, distinct from the time it describes. Funding settled 08:00 has value-timestamp 00:00–08:00 and observation-timestamp 08:00 |
| **Flow window** | 2019-12-23 → present, bounded by perp–spot basis start. 58,198 of 78,666 1h bars (74%). All Stage C/D sample counts are reported over this window |
| **Base currency** | **GBP (B.0.6).** Equity, P&L, drawdown and the risk-cap base are GBP, converted from USD-quoted instruments at the point-in-time DEXUSUK rate (`gbp = usd / gbpusd`). Weekend bars carry Friday's rate — the last observable one, and a real unhedgeable exposure rather than a modelling artifact. **Tier 1 gates read the GBP figures**; USD is reported alongside as reference so the FX contribution is visible |

**Expected effective/raw ratio for events.** 0.7–0.95, not the 0.145–0.218 seen on all bars (S16) — events are sparse by construction, so their label windows overlap far less. Recorded in advance so a ratio in the expected band is not read as confirmation of anything, and a ratio far below it is read as a defect in the event sampler.

---

## 8. Component Register

**KEEP — validated infrastructure**

- Layer 1 Data Engine (OHLCV ingest, gap classification, outlier detection)
- Layer 2 Feature Engineering (39 features — retained as *inputs*, no longer sufficient alone)
- Layer 14 Backtest harness — **after** the S11/S12 fixes
- Ablation harness — **after** the I3 flagging fix
- Walk-forward scaffolding — **after** purging is added
- Test suite (374 passing, 2026-08-15)
- Optuna infrastructure — **with** trial registry wired in
- `SegmentResult.equity_destroyed`, `min_oos_trades_per_fold`, `under_powered_folds`, `n_folds_in_aggregate_sharpe` (landed)

**DELETE — measured useless, remove entirely**

- `HistoricalAnalogEngine` (Component A / k-NN) — distance metric was weighted by Component B importances, inherits its uselessness
- Component B direction classifier in current form (S7)
- `EnsembleMetaModel`, `fit_ensemble_weights` (S7, and it destroyed the account)
- `Phase2Strategy.max_position` as a pseudo-cap (S10 — it is a model-layer parameter, caps size not risk, and there are no stops)
- Class-weighting machinery built for the 65/34 imbalance (S1 — it was compensating for arithmetic)
- Calibration work premised on that imbalance
- **The UUP dollar path** (yfinance ETF in `config.yaml`) — a third dollar measure reachable in one repo. DTWEXBGS is the choice; delete the alternative rather than leaving it to whichever fetcher a caller reaches for

**FIX — specific defects with known causes**

| Target | Defect | Fix |
|---|---|---|
| `equity_curve_from_trade_pnls` | Unbounded below (S11) | Floor at 0; mark `equity_destroyed` |
| Harness accounting | Two books, 20× divergence (S12) | Single equity source of truth; all costs inside the sizing book |
| Gap cost model | `n_bars × freq × mag × notional`, uncapped (S11) | Cap at position notional; reconsider per-bar linearity |
| Layer 6 threshold | 0.30 (S6, retired) | 0.20, a priori and prohibited from re-tuning |
| `DEFAULT_LEVEL_TOLERANCE_ATR` | 0.25 → 44.1% confirmation (S8) | Retune per Stage B.4; target low single digits, set in advance |
| Ablation flagging | Sharpe-only (I3) | Gate on maxDD and PF first; refuse ranking when `any_equity_destroyed` |
| Outlier threshold | Not timeframe-aware; 58.4% flag rate on 1w (S20) | **Non-blocking** — the flag is written and never read (S25), so no series is destroyed and MTF alignment is unaffected. Fix when convenient: rolling MAD with a fixed multiplier, per timeframe, specified in advance rather than tuned to produce a pleasant rate |
| Dataset versioning | Row counts drifted 78,378 → 78,381 → 78,226 → 78,666 across stages | **FIXED at B.0.1.** `data/frozen/v2`, `dataset_sha256 a5f81b5b8585d982…`. `load_frozen()` verifies the hash on every read; versions are immutable and re-pulls take the next number |
| FRED publication lag | Not modelled | **FIXED at B.0.4** — one-business-day floor. ALFRED vintages remain the real fix and raise rather than silently approximating; needs `FRED_API_KEY` |

**BUILD — new, in Build Plan order**

Purged K-Fold + embargo · uniqueness weighting · event sampler · flow-data ingest · meta-label pipeline · execution-layer risk caps · exit engine · trial registry + DSR

---

## 9. Data Inventory and Constraints

| Dataset | Status | Rows | Window | Note |
|---|---|---|---|---|
| BTC/USDT 1h OHLCV | Ingested | 78,666 | 2017-08-17 → 2026-08-13 | Reference. 25 data_error + 3 illiquid gaps (carried from the B.1 report — `gap_class` is not persisted, so it is not re-measurable from `data/raw/`) |
| BTC/USDT 1d | Ingested | 3,284 | 2017-08-17 → 2026-08-13 | Outlier column unusable (S20), and unread (S25) |
| BTC/USDT 1w | Ingested | 469 | 2017-08-20 → 2026-08-09 | Outlier column unusable (S20), and unread (S25) |
| 39 engineered features | Built | — | Same | Ceiling measured, exhausted |
| Funding rate 8h | Ingested | 7,592 | 2019-09-10 → 2026-08-14 | Forward-only alignment per I9 |
| Perp–spot basis 1h | Ingested | 58,198 | 2019-12-23 → 2026-08-14 | **Binds the flow window** |
| FRED DTWEXBGS | Ingested | 5,169 | 2006-01-02 → 2026-08-14 | Selected dollar measure (S19). Publication lag modelled at B.0.4 |
| FRED DGS10 | Ingested | 16,140 | 1962-01-02 → 2026-08-14 | Own gap classifier. Lag modelled |
| FRED VIXCLS | Ingested | 9,252 | 1990-01-02 → 2026-08-14 | Own gap classifier. Lag modelled |
| FRED DEXUSUK | Ingested | 13,947 | 1971-01-04 → 2026-08-14 | **Base-currency rate (B.0.6).** USD per one GBP — required series; its absence is a hard failure, not a degraded run |
| ETH/USDT 1h OHLCV | Ingested | 35,361 | 2022-08-01 → 2026-08-13 | **Out of scope for Gate B** — see below. Note the start: 4 years, not 9, and it postdates the flow window by 2.6 years |
| Open interest (5m) | Free, not yet ingested | — | — | Binance Data Vision `futures/um/daily/metrics/` |
| Long/short ratios (5m) | Free, not yet ingested | — | — | Same file as OI |
| Taker buy/sell volume ratio | Free, not yet ingested | — | — | Same file as OI |
| Liquidations (event-level) | Free, partial window | — | — | `liquidationSnapshot` — publishing reportedly stalled ~2024-03; verify bucket |
| Liquidations (live) | Free, forward-only, sampled | — | — | `!forceOrder@arr` — largest order per 1000ms per symbol only (S24) |
| aggTrades (tick) | Free, deferred | — | — | Large; Tier 3 pending Stage C |

**Hard constraint, revised 2026-08-15.** No paid data is required. The flow window is bounded by whichever of perp–spot basis (2019-12-23) or the `metrics` start date is later — measured at B.2.4. Complete event-level liquidation history is the one genuine gap; it is handled by the OI-cascade proxy (validated at B.2.6 against whatever `liquidationSnapshot` window exists) plus forward-only collection from 2026-08.

All Gate B, C, and D counts are reported over the flow window, and separately over the full OHLCV window for reference only.

**BTC only for Gate B (ruled 2026-08-15).** Three reasons, the third binding:

1. The flow window is defined by *BTC* basis. ETH has no funding or basis ingested, so ETH events sit in a window where Stages C and D cannot operate. Counting events that cannot be modelled inflates the denominator.
2. If BTC alone clears 200, ETH is upside — breadth, per the Fundamental Law, and exactly what Stage E wants.
3. If BTC alone fails, **adding ETH to rescue the count is a decision made after seeing the number.** That is loosening a threshold in a different dimension, and the gate prohibits it.

ETH OHLCV-only event count is reported alongside, for information. ETH funding and basis are a **B.2 line item conditional on Gate B passing** — now free rather than purchased (S23), but still deliberately deferred: reason 3 is about *when the decision is made*, not what it costs. A free rescue after seeing the number is the same inversion of the gate as a paid one.

Note also that ETH 1h begins 2022-08-01 — 4 years, and 2.6 years inside the flow window. It could not have carried a Gate B count on its own terms even if it were in scope.

---

## 10. Session Protocol

Every session begins by:

1. Reading this file, then `project_state.md`
2. Stating which Stage and which Gate the work belongs to
3. Stating which Invariants (§5) are live for the task
4. Confirming the frozen dataset hash matches `project_state.md` before any measurement. A mismatch means stop and reconcile — cross-stage comparisons against a drifting dataset are meaningless

Every session ends by:

1. Updating `project_state.md` with what landed, what was deferred, and why
2. Logging any model configurations evaluated to `trials.jsonl`
3. Explicitly stating whether a Gate was reached, and **never** proceeding past a Gate without operator sign-off

**Nothing is committed without tests passing. Gates are not self-certified by the agent.**
