"""
Unit tests for macro data ingestion. Network calls (yfinance, CoinGecko) are
mocked -- these test the routing/scaling logic, not live data.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from src.data_engine.macro_fetcher import fetch_macro_tickers, ingest_macro


def test_ingest_macro_routes_btc_dominance_key_to_coingecko(tmp_path):
    """
    Regression test: `ingest_macro` pops a `"btc_dominance"` key (any value)
    out of `config["macro"]["tickers"]` and routes it to
    `fetch_btc_dominance()` (CoinGecko) instead of yfinance. The key was
    previously named `"btc_dom"` in config.yaml, which didn't match --
    `btc_dom_requested` was always False, so CoinGecko was never called, and
    the associated value ("BTC.D", not a real yfinance symbol) was sent to
    yfinance instead, producing all-NaN columns. This test locks in the
    correct routing by key name, independent of the config file's contents.
    """
    config = {
        "data_dir": str(tmp_path),
        "macro": {
            "tickers": {
                "vix": "^VIX",
                "btc_dominance": "coingecko",  # value is irrelevant; only the key matters
            }
        },
    }

    with patch("src.data_engine.macro_fetcher.fetch_macro_tickers") as mock_fetch_tickers, \
         patch("src.data_engine.macro_fetcher.fetch_btc_dominance") as mock_fetch_dom:
        mock_fetch_tickers.return_value = pd.DataFrame(
            {"vix_Close": [15.0]}, index=[pd.Timestamp.now(tz="UTC")]
        )
        mock_fetch_dom.return_value = pd.DataFrame(
            {"btc_dominance_pct": [55.0]}, index=[pd.Timestamp.now(tz="UTC")]
        )

        ingest_macro(config)

        # btc_dominance must NOT be forwarded to the yfinance ticker fetch.
        called_tickers = mock_fetch_tickers.call_args[0][0]
        assert "btc_dominance" not in called_tickers
        assert called_tickers == {"vix": "^VIX"}
        # CoinGecko must have been called since the key was present.
        mock_fetch_dom.assert_called_once()


def test_ingest_macro_skips_coingecko_when_key_absent(tmp_path):
    """Sibling case: with no 'btc_dominance' key, CoinGecko must not be
    called at all -- confirms the routing is driven by key presence, not
    called unconditionally."""
    config = {
        "data_dir": str(tmp_path),
        "macro": {"tickers": {"vix": "^VIX"}},
    }

    with patch("src.data_engine.macro_fetcher.fetch_macro_tickers") as mock_fetch_tickers, \
         patch("src.data_engine.macro_fetcher.fetch_btc_dominance") as mock_fetch_dom:
        mock_fetch_tickers.return_value = pd.DataFrame(
            {"vix_Close": [15.0]}, index=[pd.Timestamp.now(tz="UTC")]
        )
        ingest_macro(config)
        mock_fetch_dom.assert_not_called()


def test_fetch_macro_tickers_applies_no_scale_correction():
    """
    Regression test: this module used to multiply ^TNX/^TYX/^FVX columns by
    0.1 on the (incorrect) assumption that Yahoo reports yield * 10. Verified
    against live data that yfinance's ^TNX/^TYX/^FVX already report the
    actual percentage yield directly (^TNX closed at 0.917 on 2021-01-04,
    matching the real ~0.92% 10Y yield that day; 4.670 on 2026-08-06, matching
    the real ~4.67% yield then) -- the old 0.1 factor silently divided real
    yield data by 10. `fetch_macro_tickers` must pass ticker values through
    unmodified.
    """
    idx = pd.DatetimeIndex(["2024-01-02"], name="Date")
    raw = pd.DataFrame({"Close": [4.67], "Open": [4.6], "High": [4.7], "Low": [4.55], "Volume": [0]}, index=idx)

    with patch("src.data_engine.macro_fetcher.yf.download") as mock_download:
        mock_download.return_value = raw
        result = fetch_macro_tickers({"us10y": "^TNX"}, start="2024-01-01")

    assert result["us10y_Close"].iloc[0] == pytest.approx(4.67), (
        "us10y_Close was scaled -- expected the raw yfinance value passed through unmodified."
    )
