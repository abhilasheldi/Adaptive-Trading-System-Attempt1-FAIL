"""
Stage A.1 -- per-sample label metadata (t0, t1, barrier_touched,
realised_return) emitted by `apply_triple_barrier`.

t1 is the load-bearing one. Purged K-Fold (Stage A.3) drops training
samples whose label window [t0, t1] overlaps the test fold; without a real
t1 that predicate cannot be written, which is why the current splitter
partitions on t0 alone and leaks. These tests pin the contract t1 must
satisfy before anything is built on top of it.
"""

import numpy as np
import pandas as pd
import pytest

from src.predictive_core.labeling import apply_triple_barrier


def _frame(closes, highs=None, lows=None, atr=1.0, start="2022-01-01"):
    n = len(closes)
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes if highs is None else np.asarray(highs, dtype=float),
            "low": closes if lows is None else np.asarray(lows, dtype=float),
            "close": closes,
            "volume": np.ones(n),
            "vol_atr_14": np.full(n, atr, dtype=float),
        },
        index=idx,
    )


# ---------------------------------------------------------------------- #
# The four required columns exist and are well-formed
# ---------------------------------------------------------------------- #


def test_emits_all_four_stage_a1_columns():
    df = _frame(np.linspace(100.0, 110.0, 200))
    out = apply_triple_barrier(df, max_candles=24)
    for col in ("t0", "t1", "barrier_touched", "realised_return"):
        assert col in out.columns, f"missing Stage A.1 column: {col}"
    assert out["t0"].notna().all()
    assert out["t1"].notna().all()
    assert out["realised_return"].notna().all()


def test_barrier_touched_is_only_the_three_permitted_values():
    df = _frame(np.linspace(100.0, 130.0, 400))
    out = apply_triple_barrier(df, max_candles=24)
    assert set(out["barrier_touched"].unique()) <= {"tp", "sl", "vertical"}


def test_t0_matches_the_index():
    df = _frame(np.linspace(100.0, 110.0, 200))
    out = apply_triple_barrier(df, max_candles=24)
    assert (out["t0"].to_numpy() == out.index.to_numpy()).all()


# ---------------------------------------------------------------------- #
# t1 -- the purging-critical contract
# ---------------------------------------------------------------------- #


def test_t1_is_strictly_after_t0_for_every_sample():
    """A label that resolves at or before its own event time would make the
    purge predicate vacuous."""
    df = _frame(np.linspace(100.0, 130.0, 400))
    out = apply_triple_barrier(df, max_candles=24)
    assert (out["t1"] > out["t0"]).all()


def test_t1_equals_t0_plus_duration_bars():
    """`duration` and `t1` must describe the same resolution bar. They are
    computed on different code paths for the tp/sl and vertical branches, so
    agreement is asserted, not assumed."""
    df = _frame(np.linspace(100.0, 130.0, 400))
    out = apply_triple_barrier(df, max_candles=24)
    expected = out["t0"] + pd.to_timedelta(out["duration"], unit="h")
    assert (out["t1"] == expected).all()


def test_t1_never_exceeds_the_vertical_barrier_horizon():
    df = _frame(np.linspace(100.0, 130.0, 400))
    horizon = 24
    out = apply_triple_barrier(df, max_candles=horizon)
    assert out["duration"].max() <= horizon
    assert ((out["t1"] - out["t0"]) <= pd.Timedelta(hours=horizon)).all()


def test_t1_is_always_an_observed_bar_timestamp():
    """t1 must be a real bar in the source index, never an extrapolated
    wall-clock offset -- exchange outages leave genuine gaps, and a purge
    boundary landing between bars would silently include or exclude the
    wrong rows."""
    df = _frame(np.linspace(100.0, 130.0, 400))
    out = apply_triple_barrier(df, max_candles=24)
    assert out["t1"].isin(df.index).all()


def test_t1_survives_a_gapped_irregular_index():
    """Resolution is by bar POSITION, not wall-clock arithmetic. With a
    missing hour in the middle, t1 must still land on a real bar."""
    df = _frame(np.linspace(100.0, 130.0, 300))
    df = df.drop(df.index[150:160])  # simulate an exchange outage
    out = apply_triple_barrier(df, max_candles=24)
    assert out["t1"].isin(df.index).all()
    assert (out["t1"] > out["t0"]).all()


# ---------------------------------------------------------------------- #
# barrier_touched and realised_return agree with each other
# ---------------------------------------------------------------------- #


def test_tp_rows_price_at_the_take_profit_barrier():
    """Every 'tp' row must carry exactly +tp_mult*ATR/entry -- a resting
    order fills where it sits."""
    df = _frame(np.linspace(100.0, 160.0, 400))
    tp_mult, atr = 2.0, 1.0
    out = apply_triple_barrier(df, atr_col="vol_atr_14", tp_mult=tp_mult, sl_mult=1.0, max_candles=24)
    tp_rows = out[out["barrier_touched"] == "tp"]
    assert len(tp_rows) > 0, "fixture produced no tp touches"
    expected = tp_mult * atr / tp_rows["close"].to_numpy()
    assert np.allclose(tp_rows["realised_return"].to_numpy(), expected)
    assert (tp_rows["realised_return"] > 0).all()
    assert (tp_rows["direction"] == 1.0).all()


def test_sl_rows_price_at_the_stop_barrier_and_are_negative():
    df = _frame(np.linspace(160.0, 100.0, 400))
    sl_mult, atr = 1.0, 1.0
    out = apply_triple_barrier(df, atr_col="vol_atr_14", tp_mult=2.0, sl_mult=sl_mult, max_candles=24)
    sl_rows = out[out["barrier_touched"] == "sl"]
    assert len(sl_rows) > 0, "fixture produced no sl touches"
    expected = -sl_mult * atr / sl_rows["close"].to_numpy()
    assert np.allclose(sl_rows["realised_return"].to_numpy(), expected)
    assert (sl_rows["realised_return"] < 0).all()
    assert (sl_rows["direction"] == -1.0).all()


def test_vertical_rows_price_at_the_horizon_close():
    """Flat series: nothing touches a barrier, so everything resolves
    vertical at ~zero return."""
    df = _frame(np.full(200, 100.0), atr=5.0)
    out = apply_triple_barrier(df, max_candles=24, neutral_zone_atr=0.25)
    assert (out["barrier_touched"] == "vertical").all()
    assert np.allclose(out["realised_return"].to_numpy(), 0.0)
    assert (out["direction"] == 0.0).all()


def test_realised_return_is_distinct_from_magnitude():
    """`magnitude` is MFE (best the trade ever looked); `realised_return` is
    what an exit at t1 actually captured. Conflating them would overstate
    every stopped-out trade. On an sl row the two must disagree in sign."""
    df = _frame(np.linspace(160.0, 100.0, 400))
    out = apply_triple_barrier(df, max_candles=24)
    sl_rows = out[out["barrier_touched"] == "sl"]
    assert len(sl_rows) > 0
    assert (sl_rows["magnitude"] >= 0).all()          # MFE is unsigned
    assert (sl_rows["realised_return"] < 0).all()     # realised P&L is not


# ---------------------------------------------------------------------- #
# Metadata survives the operations Stage A.3 will perform on it
# ---------------------------------------------------------------------- #


def test_metadata_survives_slicing_and_concat():
    """Purged K-Fold slices and recombines these frames constantly. The
    metadata columns must ride along -- unlike `.attrs`, which does not
    survive a disagreeing concat."""
    df = _frame(np.linspace(100.0, 130.0, 400))
    out = apply_triple_barrier(df, max_candles=24)
    recombined = pd.concat([out.iloc[:100], out.iloc[100:]])
    for col in ("t0", "t1", "barrier_touched", "realised_return"):
        assert col in recombined.columns
    assert (recombined["t1"] == out["t1"]).all()


def test_label_windows_actually_overlap_which_is_why_purging_is_needed():
    """The premise of Stage A.2/A.3 in one assertion: consecutive labels
    share bars, so nominal n overstates independent information."""
    df = _frame(np.linspace(100.0, 130.0, 400))
    out = apply_triple_barrier(df, max_candles=24)
    # Sample i's window [t0_i, t1_i] extends past sample i+1's t0.
    overlapping = (out["t1"].to_numpy()[:-1] > out["t0"].to_numpy()[1:]).sum()
    assert overlapping > 0.9 * (len(out) - 1)
