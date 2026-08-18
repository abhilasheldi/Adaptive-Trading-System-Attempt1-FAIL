# PRE-REGISTRATION — Condition 4: MTF Alignment

**Registered:** 2026-08-18
**Stage:** B.5, before Gate B
**Status:** COMMITTED BEFORE ANY COUNT IS COMPUTED. Frozen on commit.

---

## Why this document exists

The two-condition upper bound of **298.9 effective events** on the flow window is
already known. Defining condition 4 with that number visible is exactly the
situation in which a definition drifts toward the answer it needs. Registration
comes first, and nothing below is swept, tuned, or revisited after a count.

Gate B threshold is 200 effective. This definition was written knowing the
bound is 298.9 and knowing the remaining budget is 33%. It was not chosen to
fit inside that budget.

---

## Data

| timeframe | source | status |
|---|---|---|
| 1d | `BTC_USDT_1d.parquet` (frozen v2, 3,284 rows) | ingested |
| 1w | `BTC_USDT_1w.parquet` (frozen v2, 469 rows) | ingested |

**1h and 4h are NOT used.** H1 names "MTF alignment ≥ 3/4" and the canonical
set in `mtf_alignment.py` is (1H, 4H, 1D, 1W), but only 1d and 1w are ingested
as independent series. 4h would have to be resampled from 1h, and a resampled
series is not an independent structural reading — it is the same information
re-bucketed. Using two ingested timeframes × two reading types gives four
genuinely distinct readings, which is what the 4-point scale requires.

**This is a deviation from the canonical timeframe set and is recorded as
such.** It is not a weakening chosen to raise the count: two reading types on
two timeframes is, if anything, more correlated within timeframe than four
readings across four timeframes would be, so it is at least as likely to
depress the count as to raise it.

---

## The four readings

All four are signed, in {−1, 0, +1}, with **N = 20 periods. Fixed. Not swept.**

### (a) Trend direction — one per timeframe

```
trend_direction = sign(close − SMA(close, 20))
```

`sign(0) = 0`. Computed independently on the 1d series and the 1w series.

### (b) Structure direction — one per timeframe

"Sign of the most recent HH/HL vs LH/LL sequence", made deterministic over the
same N = 20:

```
Let W_now  = the trailing 20 bars ending at the current bar (inclusive)
Let W_prev = the 20 bars immediately before W_now

hh = max(high[W_now]) >  max(high[W_prev])       # higher high
hl = min(low[W_now])  >  min(low[W_prev])        # higher low
lh = max(high[W_now]) <  max(high[W_prev])       # lower high
ll = min(low[W_now])  <  min(low[W_prev])        # lower low

structure_direction = +1  if (hh AND hl)         # HH/HL sequence
                      −1  if (lh AND ll)         # LH/LL sequence
                       0  otherwise              # mixed / no sequence
```

Requiring BOTH legs (not either) is what makes it a *sequence* rather than a
single extreme. Mixed structure reads 0 and contributes to neither side, which
is the honest reading of "no clear sequence".

**No new constant is introduced.** N = 20 is the registered period and is used
for the SMA and for both structure windows.

---

## Alignment and threshold

The sweep's **implied reversal direction**:

| `sweep_direction` | meaning | implied direction |
|---|---|---|
| `"bearish"` | swept resistance, closed back below → reversal down | **−1** |
| `"bullish"` | swept support, closed back above → reversal up | **+1** |

```
alignment_score = count of the 4 readings whose sign EQUALS the implied direction
```

A reading of 0 agrees with nothing and is never counted. Readings of the
opposite sign are not counted.

**Condition 4 holds when `alignment_score ≥ 3`.** Per H1 as written. Not swept.

---

## I9 — point-in-time, and why this is the funding defect in another shape

A 1d or 1w reading is observable **only at that bar's close**, never during it.
A 1d bar indexed at date D describes D 00:00–24:00 and is not knowable until
D+1 00:00. Using it on an intraday 1h bar inside D is lookahead baked into the
feature — the same failure as back-filling a funding rate settled at 08:00 into
the 04:00 bar, and equally invisible to any splitter.

Observation policies, declared explicitly per series (I9):

| series | value timestamp | observation timestamp | policy |
|---|---|---|---|
| BTC 1d | bar open (date D) | D + 1 day | `BAR_CLOSE_1D`, calendar_lag = 1 day |
| BTC 1w | bar open (week W) | W + 7 days | `BAR_CLOSE_1W`, calendar_lag = 7 days |

Alignment onto 1h bars goes through `point_in_time.align_point_in_time`, which
has no policy-free code path. Bars before the first observable reading are
**NaN, never 0** — and a NaN reading fails the condition rather than counting
as neutral agreement.

**Verification required before any count:** `assert_no_lookahead` must pass on
both aligned reading sets, and the naive value-timestamp join must be shown to
fail it. A guard that is never exercised is indistinguishable from one that is
absent (mutation-testing finding, 2026-08-18).

---

## What is deliberately NOT in this definition

- **No 4h reading.** Resampled from 1h, therefore not independent.
- **No neutral-band parameter.** `score_mtf_alignment`'s `DEFAULT_NEUTRAL_BAND`
  applies to continuous readings in [−1, 1]; these readings are already
  discrete {−1, 0, +1}, so a band would either do nothing or silently zero out
  every reading. This definition computes the alignment count directly.
- **No swept alternatives.** N = 20, threshold ≥ 3, and both reading
  definitions are fixed here and are not revisited after a count.

---

## Falsifiability of this registration

If the built implementation disagrees with any line above, the implementation
is wrong, not this document. The count is invalid and must be recomputed.
