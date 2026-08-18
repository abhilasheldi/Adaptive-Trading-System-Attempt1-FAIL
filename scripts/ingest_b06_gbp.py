"""
B.0.6 -- re-ingest the FRED macro frame including DEXUSUK (GBP base currency).

Rebuilds `data/raw/macro_fred_daily.parquet` with all four series. Every
fetch is verified against its request (I10, B.0.3), and the resulting frame
is checked for the point-in-time property it will be consumed under (I9,
B.0.2) before it is written.

    python scripts/ingest_b06_gbp.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_engine.acquisition import (  # noqa: E402
    AcquisitionRequest,
    assert_acquisition,
)
from src.data_engine.fred_fetcher import (  # noqa: E402
    FRED_SERIES,
    REQUIRED_SERIES,
    classify_macro_gaps,
    dollar_substitution_note,
    fetch_fred_series,
    save,
)
from src.data_engine.point_in_time import (  # noqa: E402
    FRED_DAILY,
    align_point_in_time,
    assert_no_lookahead,
    observation_coverage,
)

OUT = Path("data/raw/macro_fred_daily.parquet")

# Minimum rows expected per series, from the B.1 measurement. Set BELOW the
# known counts, not at them -- these are floors that catch an S21-class
# truncation, not equality checks that would fail every time FRED publishes
# a new day.
EXPECTED_MIN = {
    "dxy_proxy": 5_000,
    "us10y": 16_000,
    "vix": 9_000,
    "gbpusd": 13_000,
}


def main() -> int:
    cols = {}
    print(f"{'series':<12}{'fred id':<12}{'rows':>9}  window")
    print("-" * 62)

    for name, (sid, _exact, _note) in FRED_SERIES.items():
        s = fetch_fred_series(sid)
        assert_acquisition(
            s.to_frame(name),
            AcquisitionRequest(
                source=f"fred:{sid}",
                expected_min_rows=EXPECTED_MIN.get(name),
                note=f"B.0.6 re-ingest; {name}",
            ),
        )
        cols[name] = s
        print(f"{name:<12}{sid:<12}{len(s):>9,}  {s.index.min().date()} -> {s.index.max().date()}")

    macro = pd.DataFrame(cols).sort_index()

    missing = [c for c in REQUIRED_SERIES if c not in macro.columns or macro[c].isna().all()]
    if missing:
        raise SystemExit(
            f"!! required series missing: {missing}. Without a base-currency rate every "
            f"reported number is in the wrong currency (B.0.6)."
        )

    # I9 check on the frame we are about to persist, at the 1h cadence it
    # will actually be consumed at. Verified, not assumed.
    bars = pd.date_range(macro.index.min(), macro.index.max(), freq="1h", tz="UTC")
    assert_no_lookahead(macro, align_point_in_time(macro, bars, FRED_DAILY), FRED_DAILY)

    print("\ngap classification (own calendar -- weekends are market_closed, not errors)")
    for c in macro.columns:
        counts = classify_macro_gaps(macro, c).value_counts().to_dict()
        print(f"  {c:<12}{ {k: int(v) for k, v in counts.items()} }")

    print(f"\npoint-in-time coverage at 1h, publication lag = {FRED_DAILY.business_day_lag} business day")
    cov = observation_coverage(macro, bars, FRED_DAILY)
    print(f"  first observable: {cov['first_observable']}")
    print(f"  pre-observation bars: {cov['n_pre_observation_bars']:,} of {cov['n_bars']:,}")

    save(macro, OUT)
    print(f"\nsaved {len(macro):,} rows x {len(macro.columns)} cols -> {OUT}")
    print(f"\n{dollar_substitution_note()}")
    print("\nRe-freeze now: python scripts/freeze_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
