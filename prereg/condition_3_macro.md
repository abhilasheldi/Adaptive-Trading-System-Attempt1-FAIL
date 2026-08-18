# PRE-REGISTRATION — Condition 3: Macro Regime (restated)

**Registered:** 2026-08-18
**Stage:** B.5, before Gate B
**Status:** COMMITTED BEFORE ANY COUNT IS COMPUTED. Frozen on commit.
**Falsification budget:** consumes one revision — **2 of 3 → 1 of 3**.

---

## Ruling being registered

Restate, do not procure. The BTC-dominance leg is **removed** — no free
historical source exists (CoinGecko `/global` returns current dominance only;
historical dominance needs historical total market cap, which is paid). The
BTC/QQQ correlation leg is **retained**; QQQ daily is free via yfinance.

This consumes falsification budget under §6: a leg was unavailable, which is
the *marginal/ambiguous* case, not a decisive negative.

---

## FINDING RECORDED BEFORE COUNTING: dropping the dominance leg cannot change the event set

Stated here so it is on record *before* any number is produced, and provable
mechanically rather than asserted (I4).

The existing classifier is exactly:

```python
risk_on  = dxy_not_rising AND vix_calm AND dominance_expanding AND correlation_positive
risk_off = dxy_rising    AND vix_elevated_or_spiking AND correlation_compressing_to_one
```

`dominance_expanding` appears in `risk_on` **only**. It appears nowhere in
`risk_off`.

H1's condition 3 admits `macro_regime ∈ {risk-on, transitional}`. Since
Transitional is the residual state and Risk-On/Risk-Off are mutually exclusive
by construction (`dxy_not_rising` and `dxy_rising` partition), that set is
exactly **NOT Risk-Off** (and not Macro-Event Window).

Removing a leg from `risk_on` can only move bars **from Transitional to
Risk-On**. Both pass the filter. **The admitted event set is identical with or
without the dominance leg.**

A test asserting this equivalence over the real series is required before the
count is accepted.

**The budget decrement is applied as ruled regardless.** The ruling is the
operator's and is followed; this note exists so it can be reversed on full
information, not so it can be ignored. Spending a revision one may not owe is
the conservative error; not spending one that is owed is not.

---

## Legs, and the justification of each from MASTER_CONTEXT's own table

MASTER_CONTEXT's regime table, verbatim:

> **Risk-On:** DXY falling/flat, VIX < 20 and falling, BTC dominance expanding,
> BTC/QQQ correlation positive
> **Risk-Off:** DXY rising, VIX > 25 or spiking, cross-asset correlations
> compressing toward 1.0
> **Transitional:** Mixed signals, no clear directional consensus

### Leg 1 — DXY trend

```
dxy_not_rising = dxy_20d_momentum_pct <= 0.0      # "falling/flat"
dxy_rising     = dxy_20d_momentum_pct >  0.0      # "rising"
```

**Justification:** the table says "falling/flat" versus "rising". That is a
sign test; the threshold is zero and is not a chosen number. 20-day window per
the existing field definition and the project's N = 20 convention.
Source series: `dxy_proxy` (DTWEXBGS — a *different index* from ICE DXY, not a
proxy for it, S19; this must be stated wherever a Layer 3 result is reported).

### Leg 2 — VIX

```
vix_calm               = vix_level < 20.0 AND vix_5d_roc_pct <  0.0
vix_elevated_or_spiking = vix_level > 25.0 OR  vix_5d_roc_pct >= 30.0
```

**Justification:** 20.0 and 25.0 are **exact MASTER_CONTEXT numbers**
(`VIX_RISK_ON_MAX`, `VIX_RISK_OFF_MIN`). "and falling" is an AND in the table;
"or spiking" is an OR. Both are preserved as written.

"Spiking" has no numeric definition anywhere in the source docs. The existing
`DEFAULT_VIX_SPIKE_5D_ROC_PCT = 30.0` is **inherited unchanged** — an
acknowledged judgment call, adopted rather than re-chosen, precisely so that no
new number enters at the moment of the kill test.

### Leg 3 — BTC/QQQ correlation

```
correlation_positive            = btc_qqq_corr_20d >  0.0
correlation_compressing_to_one  = btc_qqq_corr_20d >= 0.7
```

**Justification:** "correlation positive" is a sign test — threshold zero, not
chosen. "Compressing toward 1.0" has no stated number; the existing
`DEFAULT_RISK_OFF_CORRELATION_THRESHOLD = 0.7` is **inherited unchanged**, same
reasoning as above.

Computed as the rolling 20-day Pearson correlation of daily BTC and QQQ
**returns** (not levels).

### Leg 4 — BTC dominance — **REMOVED**

No free historical source. Removed from `risk_on`. See the finding above: this
provably cannot change the admitted set.

---

## US10Y is NOT a classifying leg — deviation from the brief, with reason

The brief lists US10Y among the inputs determining risk-on/transitional. **It is
not used in classification here**, and this is a deliberate departure that the
operator should overrule if they disagree.

The instruction for this document is to *justify each threshold from the
existing MASTER_CONTEXT Risk-On definition*. **MASTER_CONTEXT's regime table
contains no US10Y term in any of its four states.** There is therefore no
threshold for US10Y that can meet the stated justification standard — any value
would be invented at the moment of the kill test and then frozen by this very
registration, which is the worst possible place to introduce an unjustified
parameter.

This also matches a documented prior decision in `macro_regime.py`: US10Y is
accepted as an input because the Prompt Playbook lists it, but is used in
"NEITHER classification NOR confidence" and is echoed on the result for
diagnostic visibility only.

**US10Y is therefore ingested, carried, and reported, but does not gate.**

---

## State mapping

```
Macro-Event Window : next_high_impact_event within 24h   -> NOT COMPUTABLE
Risk-On            : dxy_not_rising AND vix_calm AND correlation_positive
Risk-Off           : dxy_rising AND vix_elevated_or_spiking AND correlation_compressing_to_one
Transitional       : everything else
```

**Condition 3 holds when the state is Risk-On or Transitional** — equivalently,
when it is not Risk-Off.

### Macro-Event Window is not computable (I5)

No economic-calendar source is wired into this project. `next_high_impact_event`
is passed as `None`, so this state **never fires**. The consequence is stated
rather than hidden: bars within 24h of an FOMC/NFP/CPI release are classified
by their market signals alone and are **admitted** by condition 3 where the full
definition would have excluded them. **Condition 3 as implemented is therefore
LOOSER than H1's intent**, and the event count is correspondingly an
over-estimate on this axis.

### Structural note, recorded before counting

Risk-Off is an **AND of three** conditions, one of which requires
`btc_qqq_corr_20d >= 0.7`. Simultaneous satisfaction should be uncommon, so
condition 3 is expected to remove **few** events. This is a property of
MASTER_CONTEXT's own definition, not a tuning choice, and it is written down
now so the measured filtering rate is not later mistaken for a bug or for a
result.

---

## I9 and I10

- **QQQ** declares its own observation policy: one business day, matching
  DEXUSUK. A market close on date D is treated as observable from D+1 business
  day. Conservative in the safe direction.
- **All FRED legs** already carry the B.0.4 one-business-day publication floor.
- **QQQ ingest** is verified under I10 — row count and date range asserted
  against request, hard-failing on mismatch.
- **Single-sourcing:** QQQ must be reachable from exactly one provider. Any
  second route hard-fails, per the DTWEXBGS ruling.

---

## What is deliberately NOT in this definition

- No new numeric threshold anywhere. Every number is either an exact
  MASTER_CONTEXT value (20.0, 25.0), a sign test (0.0), or an inherited
  judgment call adopted unchanged (30.0, 0.7).
- No US10Y gate (see above).
- No re-tuning after a count. If the count disappoints, this document does not
  move.
