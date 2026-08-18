"""
Stage B.0 hardening -- I9 point-in-time, I10 acquisition, dataset freeze,
FRED publication lag, the dollar-path deletion, and GBP base currency.

Each test names the defect it prevents. B.0's whole purpose is that these
failures are silent in production: a back-filled feature, a truncated
fetch, a drifting dataset and an unlagged macro release all produce
well-formed output that is simply wrong.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtesting.currency import (
    CurrencyError,
    RiskCapBasis,
    convert_equity_curve,
    dual_currency_report,
    gbpusd_at_bars,
    usd_to_gbp,
)
from src.data_engine.acquisition import (
    AcquisitionError,
    AcquisitionRequest,
    assert_acquisition,
)
from src.data_engine.freeze import (
    FrozenDatasetError,
    freeze,
    load_frozen,
    next_version,
    verify,
)
from src.data_engine.macro_fetcher import _reject_dollar_tickers
from src.data_engine.point_in_time import (
    FRED_DAILY,
    FUNDING_8H,
    ObservationPolicy,
    align_point_in_time,
    assert_no_lookahead,
    observation_coverage,
)

UTC = "UTC"


# ---------------------------------------------------------------------- #
# B.0.2 -- I9 point-in-time
# ---------------------------------------------------------------------- #


def test_policy_refuses_to_default_its_lag():
    """An unstated zero lag is the assumption I9 exists to prevent. Zero is
    allowed; zero-by-omission is not."""
    with pytest.raises(ValueError, match="must state a lag"):
        ObservationPolicy(name="x", kind="published_after_value_date")

    ObservationPolicy(  # explicit zero is fine
        name="x", kind="settles_at_period_end", calendar_lag=pd.Timedelta(0)
    )


def test_negative_lag_is_rejected_as_lookahead_by_construction():
    with pytest.raises(ValueError, match="lookahead"):
        ObservationPolicy(
            name="x", kind="published_after_value_date", business_day_lag=-1
        )


def test_value_is_not_visible_before_its_observation_time():
    """The funding defect in one assertion. A rate settled 08:00 describes
    00:00-08:00, and must not appear on the 04:00 bar."""
    settle = pd.Timestamp("2024-01-01T08:00Z")
    funding = pd.Series([0.001], index=pd.DatetimeIndex([settle]), name="funding_rate")
    bars = pd.date_range("2024-01-01T00:00Z", periods=12, freq="1h")

    out = align_point_in_time(funding, bars, FUNDING_8H)

    assert out.loc[pd.Timestamp("2024-01-01T04:00Z")] != out.loc[pd.Timestamp("2024-01-01T04:00Z")]  # NaN
    assert np.isnan(out.loc[pd.Timestamp("2024-01-01T07:00Z")])
    assert out.loc[settle] == 0.001            # visible AT the settlement bar
    assert out.loc[pd.Timestamp("2024-01-01T11:00Z")] == 0.001  # and carried forward


def test_pre_observation_bars_are_nan_never_zero():
    """A zero funding rate is a real market state. Filling absence with zero
    makes the two indistinguishable to every model downstream."""
    s = pd.Series([0.002], index=pd.DatetimeIndex([pd.Timestamp("2024-01-05T08:00Z")]))
    bars = pd.date_range("2024-01-01T00:00Z", periods=48, freq="1h")
    out = align_point_in_time(s, bars, FUNDING_8H)

    pre = out.loc[out.index < pd.Timestamp("2024-01-05T08:00Z")]
    assert pre.isna().all()
    assert (pre == 0).sum() == 0


def test_alignment_cannot_be_called_without_a_policy():
    """The structural half of I9: there is no policy-free code path."""
    s = pd.Series([1.0], index=pd.DatetimeIndex([pd.Timestamp("2024-01-01T00:00Z")]))
    bars = pd.date_range("2024-01-01T00:00Z", periods=3, freq="1h")
    with pytest.raises(TypeError):
        align_point_in_time(s, bars)  # type: ignore[call-arg]


def test_assert_no_lookahead_catches_the_naive_alignment_it_exists_to_audit():
    """The guard must FAIL on the defect, not merely pass on clean data.

    The defect is the obvious implementation: reindex onto the bars and
    forward-fill on the VALUE timestamp, ignoring publication lag. That is
    what `align_to_bars` did before B.0.4, and it produces a frame that
    looks entirely normal."""
    s = pd.Series([5.0], index=pd.DatetimeIndex([pd.Timestamp("2024-01-02T00:00Z")]))
    bars = pd.date_range("2024-01-01T00:00Z", periods=96, freq="1h")

    # Compliant alignment passes.
    assert_no_lookahead(s, align_point_in_time(s, bars, FRED_DAILY), FRED_DAILY)

    # The naive one -- value-timestamp ffill, no lag -- does not.
    naive = s.reindex(s.index.union(bars)).sort_index().ffill().reindex(bars)
    assert naive.notna().sum() > 0
    # Either branch is a correct catch: the value appears on its own value
    # date, which under FRED_DAILY has no observation at or before it.
    with pytest.raises(AssertionError, match="back-filled|lookahead"):
        assert_no_lookahead(s, naive, FRED_DAILY)


def test_assert_no_lookahead_catches_an_outright_back_fill():
    """Back-filling places a value on bars that PRECEDE its observation --
    the failure no splitter can catch, because it is inside the feature."""
    s = pd.Series([5.0], index=pd.DatetimeIndex([pd.Timestamp("2024-01-03T00:00Z")]))
    bars = pd.date_range("2024-01-01T00:00Z", periods=96, freq="1h")

    backfilled = pd.Series(5.0, index=bars)  # value everywhere, including before
    with pytest.raises(AssertionError, match="back-filled|lookahead"):
        assert_no_lookahead(s, backfilled, FRED_DAILY)


def test_observation_coverage_reports_the_pre_observation_window():
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-10", "2024-01-11"], utc=True))
    bars = pd.date_range("2024-01-01T00:00Z", periods=24 * 20, freq="1h")
    cov = observation_coverage(s, bars, FRED_DAILY)
    assert cov["n_pre_observation_bars"] > 0
    assert 0 < cov["pre_observation_fraction"] < 1


# ---------------------------------------------------------------------- #
# B.0.4 -- FRED publication lag
# ---------------------------------------------------------------------- #


def test_fred_observation_is_shifted_by_at_least_one_business_day():
    """Unlagged macro would let a Friday release gate a Friday event. Macro
    now gates event FORMATION, so this contaminates the event set itself."""
    friday = pd.Timestamp("2024-01-05T00:00Z")
    s = pd.Series([100.0], index=pd.DatetimeIndex([friday]), name="vix")
    bars = pd.date_range("2024-01-05T00:00Z", periods=24 * 5, freq="1h")

    out = align_point_in_time(s, bars, FRED_DAILY)

    assert out.loc[friday] != out.loc[friday]                       # NaN on the value date
    assert np.isnan(out.loc[pd.Timestamp("2024-01-05T23:00Z")])
    assert out.loc[pd.Timestamp("2024-01-08T00:00Z")] == 100.0      # Monday


def test_friday_release_lands_on_monday_not_saturday():
    """Business-day, not calendar. A calendar shift puts Friday's release on
    Saturday -- a day no statistical agency publishes on -- understating the
    delay exactly when the weekend makes it matter."""
    friday = pd.Timestamp("2024-01-05T00:00Z")
    observed = FRED_DAILY.observation_timestamps(pd.DatetimeIndex([friday]))[0]
    assert observed == pd.Timestamp("2024-01-08T00:00Z")
    assert observed.dayofweek == 0


def test_alfred_hook_raises_rather_than_silently_approximating():
    """A silent fallback would make a vintage-accurate run indistinguishable
    from an approximate one in the output."""
    from src.data_engine.fred_fetcher import fetch_alfred_vintage

    with pytest.raises(NotImplementedError, match="API key"):
        fetch_alfred_vintage("DGS10")


# ---------------------------------------------------------------------- #
# B.0.3 -- I10 acquisition assertion
# ---------------------------------------------------------------------- #


def _frame(n: int, start: str = "2019-09-10", freq: str = "8h") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq=freq, tz=UTC)
    return pd.DataFrame({"funding_rate": np.zeros(n)}, index=idx)


def test_acquisition_catches_the_s21_truncation():
    """500 rows returned for 7,592, HTTP 200, no error. The exact shape."""
    truncated = _frame(500, start="2026-03-01")
    with pytest.raises(AcquisitionError, match="S21 signature"):
        assert_acquisition(truncated, AcquisitionRequest(
            source="funding:BTCUSDT", expected_min_rows=7_000,
        ))


def test_acquisition_catches_a_start_date_the_endpoint_ignored():
    late = _frame(7_600, start="2026-03-01")
    with pytest.raises(AcquisitionError, match="history starts"):
        assert_acquisition(late, AcquisitionRequest(
            source="funding:BTCUSDT",
            requested_start=pd.Timestamp("2019-09-01", tz=UTC),
            allowed_start_slack=pd.Timedelta(days=30),
        ))


def test_acquisition_allows_stated_slack_for_instrument_inception():
    """BTC perps begin 2019-09 whatever you ask for. That is a fact about the
    instrument, and it must be STATED per call rather than assumed globally."""
    real = _frame(7_592, start="2019-09-10")
    assert_acquisition(real, AcquisitionRequest(
        source="funding:BTCUSDT",
        requested_start=pd.Timestamp("2017-07-14", tz=UTC),
        allowed_start_slack=pd.Timedelta(days=900),
        expected_min_rows=7_000,
    ))


def test_acquisition_rejects_empty_unsorted_and_duplicated():
    with pytest.raises(AcquisitionError, match="no rows"):
        assert_acquisition(_frame(0), AcquisitionRequest(source="x"))

    dupe = _frame(4)
    dupe = pd.concat([dupe, dupe.iloc[[0]]]).sort_index()
    with pytest.raises(AcquisitionError, match="duplicate"):
        assert_acquisition(dupe, AcquisitionRequest(source="x"))

    unsorted = _frame(4).iloc[::-1]
    with pytest.raises(AcquisitionError, match="not sorted"):
        assert_acquisition(unsorted, AcquisitionRequest(source="x"))


def test_acquisition_catches_pagination_ending_early():
    short = _frame(100, start="2019-09-10")
    with pytest.raises(AcquisitionError, match="short of the requested"):
        assert_acquisition(short, AcquisitionRequest(
            source="funding:BTCUSDT",
            requested_end=pd.Timestamp("2026-08-14", tz=UTC),
        ))


# ---------------------------------------------------------------------- #
# B.0.1 -- dataset freeze
# ---------------------------------------------------------------------- #


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    d = tmp_path / "raw"
    d.mkdir()
    idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz=UTC)
    pd.DataFrame({"close": np.arange(10.0)}, index=idx).to_parquet(d / "a.parquet")
    pd.DataFrame({"x": np.arange(10.0)}, index=idx).to_parquet(d / "b.parquet")
    return d


def test_freeze_then_verify_round_trips(raw_dir: Path, tmp_path: Path):
    frozen = tmp_path / "frozen"
    m = freeze(raw_dir, frozen, note="test")
    assert m["version"] == 1
    assert len(m["files"]) == 2
    assert verify(1, frozen)["dataset_sha256"] == m["dataset_sha256"]


def test_verify_detects_a_modified_frozen_file(raw_dir: Path, tmp_path: Path):
    """The drift this exists for: A.6 measured 78,226 rows against a file
    that no longer contains 78,226 rows."""
    frozen = tmp_path / "frozen"
    freeze(raw_dir, frozen)

    target = frozen / "v1" / "a.parquet"
    idx = pd.date_range("2024-01-01", periods=11, freq="1h", tz=UTC)
    pd.DataFrame({"close": np.arange(11.0)}, index=idx).to_parquet(target)

    with pytest.raises(FrozenDatasetError, match="CHANGED"):
        verify(1, frozen)
    with pytest.raises(FrozenDatasetError, match="does not match its manifest hash"):
        load_frozen("a.parquet", 1, frozen)


def test_freeze_refuses_to_overwrite_an_existing_version(raw_dir: Path, tmp_path: Path):
    """Immutability is the entire guarantee. A hash in project_state.md must
    keep meaning what it meant."""
    frozen = tmp_path / "frozen"
    freeze(raw_dir, frozen)
    with pytest.raises(FrozenDatasetError, match="immutable"):
        freeze(raw_dir, frozen, version=1)


def test_repulls_take_the_next_version(raw_dir: Path, tmp_path: Path):
    frozen = tmp_path / "frozen"
    freeze(raw_dir, frozen)
    assert next_version(frozen) == 2
    assert freeze(raw_dir, frozen)["version"] == 2


def test_load_frozen_returns_the_data(raw_dir: Path, tmp_path: Path):
    frozen = tmp_path / "frozen"
    freeze(raw_dir, frozen)
    assert len(load_frozen("a.parquet", 1, frozen)) == 10


# ---------------------------------------------------------------------- #
# B.0.5 -- the dollar path is closed, not just unconfigured
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("name,symbol", [
    ("dxy_proxy", "UUP"),
    ("dollar", "QQQ"),        # caught on the KEY, not the symbol
    ("usd_index", "SPY"),
    ("anything", "DX-Y.NYB"),  # caught on the SYMBOL, not the key
])
def test_dollar_tickers_are_rejected_at_the_route_not_just_the_config(name, symbol):
    """Deleting the config entry alone leaves the route open -- the next
    person who wants a dollar column adds a ticker and gets a fourth answer."""
    with pytest.raises(ValueError, match="DTWEXBGS"):
        _reject_dollar_tickers({name: symbol})


def test_non_dollar_tickers_still_pass():
    _reject_dollar_tickers({"vix": "^VIX", "spx": "^GSPC", "qqq": "QQQ"})


def test_config_no_longer_carries_a_dollar_ticker():
    import yaml
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    tickers = cfg["macro"]["tickers"]
    assert "dxy_proxy" not in tickers
    assert "UUP" not in tickers.values()


def test_substitution_note_says_different_index_not_proxy():
    """B.0.5 requires this phrasing in the REPORTS. 'DXY proxy' is precisely
    the wording that would let a reader assume interchangeability."""
    from src.data_engine.fred_fetcher import dollar_substitution_note

    note = dollar_substitution_note()
    assert "DIFFERENT INDEX" in note
    assert "not a proxy" in note
    assert "DTWEXBGS" in note


# ---------------------------------------------------------------------- #
# B.0.6 -- GBP base currency
# ---------------------------------------------------------------------- #


def _fx(n: int = 240, rate: float = 1.25) -> pd.Series:
    idx = pd.date_range("2024-01-01T00:00Z", periods=n, freq="1h")
    return pd.Series(np.full(n, rate), index=idx, name="gbpusd")


def test_usd_to_gbp_divides_because_dexusuk_is_usd_per_gbp():
    """Getting this backwards is wrong by ~1.6-squared and still looks
    plausible on a chart."""
    usd = pd.Series([1250.0], index=pd.DatetimeIndex([pd.Timestamp("2024-01-01T00:00Z")]))
    out = usd_to_gbp(usd, _fx(1, rate=1.25))
    assert out.iloc[0] == pytest.approx(1000.0)   # not 1562.5


def test_gbp_equity_is_lower_than_usd_when_sterling_is_strong():
    usd_equity = pd.Series(np.full(24, 10_000.0),
                           index=pd.date_range("2024-01-01T00:00Z", periods=24, freq="1h"))
    gbp = convert_equity_curve(usd_equity, _fx(24, rate=1.25))
    assert np.allclose(gbp.to_numpy(), 8_000.0)


def test_weekend_uses_fridays_rate_because_it_is_the_last_observable_one():
    """Crypto trades through the weekend; the FX fix does not. Friday's rate
    is I9-compliant -- it is the last rate anyone could have observed."""
    macro = pd.DataFrame(
        {"gbpusd": [1.20, 1.30]},
        index=pd.to_datetime(["2024-01-04", "2024-01-08"], utc=True),  # Thu, Mon
    )
    bars = pd.date_range("2024-01-05T00:00Z", periods=24 * 6, freq="1h")  # Fri -> Wed
    rate = gbpusd_at_bars(macro, bars)

    saturday = pd.Timestamp("2024-01-06T12:00Z")
    sunday = pd.Timestamp("2024-01-07T12:00Z")
    assert rate.loc[saturday] == 1.20
    assert rate.loc[sunday] == 1.20
    # Monday's value carries a 1-business-day publication lag -> visible Tuesday.
    assert rate.loc[pd.Timestamp("2024-01-09T00:00Z")] == 1.30


def test_missing_rate_is_nan_not_silently_one():
    """Carrying 1.0 would report USD numbers as though they were GBP."""
    macro = pd.DataFrame({"gbpusd": [1.25]},
                         index=pd.to_datetime(["2024-01-10"], utc=True))
    bars = pd.date_range("2024-01-01T00:00Z", periods=48, freq="1h")
    rate = gbpusd_at_bars(macro, bars)
    assert rate.isna().all()
    assert (rate == 1.0).sum() == 0


def test_conversion_refuses_when_no_rate_is_observable():
    usd = pd.Series([100.0, 101.0],
                    index=pd.date_range("2024-01-01T00:00Z", periods=2, freq="1h"))
    empty = pd.Series([np.nan, np.nan], index=usd.index, name="gbpusd")
    with pytest.raises(CurrencyError, match="no observable|No observable"):
        usd_to_gbp(usd, empty)


def test_missing_gbpusd_column_is_a_hard_failure():
    macro = pd.DataFrame({"vix": [15.0]}, index=pd.to_datetime(["2024-01-01"], utc=True))
    with pytest.raises(CurrencyError, match="DEXUSUK"):
        gbpusd_at_bars(macro, pd.date_range("2024-01-01T00:00Z", periods=4, freq="1h"))


def test_fx_moves_change_the_gate_metrics_not_just_the_units():
    """The substantive consequence of B.0.6. A flat USD account with a
    falling pound has a real GBP drawdown -- and the GBP number is the one
    the Tier 1 gate reads."""
    n = 24 * 30
    idx = pd.date_range("2024-01-01T00:00Z", periods=n, freq="1h")
    usd_equity = pd.Series(np.full(n, 10_000.0), index=idx)      # flat in USD
    strengthening = pd.Series(np.linspace(1.20, 1.45, n), index=idx, name="gbpusd")

    r = dual_currency_report(usd_equity, strengthening)

    assert r.usd["max_drawdown"] == pytest.approx(0.0, abs=1e-9)
    assert r.gbp["max_drawdown"] > 0.15
    assert r.gbp["total_return"] < -0.15
    assert "GBP (gate)" in r.table()


def test_risk_cap_basis_pins_the_currency_the_percentages_apply_to():
    """Ratios are currency-invariant; the base is not. This exists so E.1
    cannot quietly implement the caps against the USD book."""
    b = RiskCapBasis()
    assert b.currency == "GBP"
    assert (b.max_single_trade_risk_pct,
            b.max_correlated_direction_pct,
            b.max_aggregate_open_risk_pct) == (2.0, 6.0, 10.0)
    assert "GBP equity" in b.describe()
    assert "S10" in b.describe()  # states that enforcement does not exist yet
