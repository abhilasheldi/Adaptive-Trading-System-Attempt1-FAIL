"""
Transaction cost model for the walk-forward backtesting harness.

Costs are fit and applied BEFORE any EV, Sharpe, profit factor, or drawdown
is computed (see metrics.py) -- MASTER_CONTEXT's central metric is Expected
Value per trade, and EV is meaningless if it's computed gross of costs.

Four components, all fit exclusively on a fold's TRAIN partition (no
lookahead into val/test when deriving cost parameters -- the same
discipline `predictive_core` applies to model fitting applies here to cost
assumptions):

1. Spread cost: the 75th percentile (MASTER_CONTEXT is explicit: NOT the
   median -- the median understates the tail-heavy cost of trading through
   a wide-spread regime) of the historical per-bar spread distribution.
   Real bid/ask spread data isn't available from the OHLCV feed this system
   ingests (see `data_engine/crypto_fetcher.py`), so spread is proxied from
   the candle's high-low range scaled by `spread_proxy_fraction_of_range`.
   This is a documented stand-in -- replace it the moment L2/quote data is
   available.
2. Taker fee: a fixed fraction of notional, charged on both entry and exit.
3. Slippage: scales with both the size of the position being entered/exited
   and the candle's range at that bar (bigger orders in choppier/thinner
   bars move the execution price further against you).
4. Gap cost: an expected-value cost added to trades whose holding period
   spans a historically gap-prone window (weekend or low-liquidity
   overnight UTC hours), sized from the empirical frequency and magnitude
   of abnormal candle-to-candle jumps observed in the TRAIN partition
   during those specific windows.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostParameters:
    """Fitted, fold-specific cost parameters. Immutable and fully reported
    so every backtest run can show exactly what cost assumptions produced
    its numbers."""

    spread_p75_frac: float
    taker_fee_frac: float
    slippage_coefficient: float
    gap_frequency_weekend: float
    gap_magnitude_weekend_frac: float
    gap_frequency_overnight: float
    gap_magnitude_overnight_frac: float
    overnight_start_utc_hour: int
    overnight_end_utc_hour: int
    gap_threshold_mult: float
    spread_proxy_fraction_of_range: float
    n_train_bars: int


class TransactionCostModel:
    """
    Fit on a fold's TRAIN partition, then applied to that same fold's
    in-sample (train+val) and out-of-sample (test) backtests. Cost
    parameters are deliberately derived only from what would have been
    knowable at decision time (the training window), never re-fit on the
    period being evaluated -- fitting spread/slippage/gap assumptions on
    the test window itself would leak realized-period cost conditions into
    a number that's supposed to represent an ex-ante cost estimate.
    """

    def __init__(
        self,
        taker_fee_frac: float = 0.0005,
        spread_percentile: float = 75.0,
        spread_proxy_fraction_of_range: float = 0.25,
        slippage_coefficient: float = 0.5,
        overnight_start_utc_hour: int = 0,
        overnight_end_utc_hour: int = 6,
        gap_threshold_mult: float = 3.0,
    ) -> None:
        if taker_fee_frac < 0:
            raise ValueError("taker_fee_frac must be non-negative.")
        if not (0.0 < spread_percentile <= 100.0):
            raise ValueError("spread_percentile must be in (0, 100].")
        if spread_percentile < 75.0:
            raise ValueError(
                "MASTER_CONTEXT requires the 75th-percentile historical spread, "
                "not a lower (more optimistic) percentile such as the median. "
                f"Got spread_percentile={spread_percentile}."
            )
        if spread_proxy_fraction_of_range <= 0:
            raise ValueError("spread_proxy_fraction_of_range must be positive.")
        if slippage_coefficient < 0:
            raise ValueError("slippage_coefficient must be non-negative.")
        if not (0 <= overnight_start_utc_hour < 24) or not (0 <= overnight_end_utc_hour < 24):
            raise ValueError("overnight hours must be in [0, 24).")
        if gap_threshold_mult <= 0:
            raise ValueError("gap_threshold_mult must be positive.")

        self.taker_fee_frac = taker_fee_frac
        self.spread_percentile = spread_percentile
        self.spread_proxy_fraction_of_range = spread_proxy_fraction_of_range
        self.slippage_coefficient = slippage_coefficient
        self.overnight_start_utc_hour = overnight_start_utc_hour
        self.overnight_end_utc_hour = overnight_end_utc_hour
        self.gap_threshold_mult = gap_threshold_mult

        self._params: Optional[CostParameters] = None

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #

    def fit(self, train_df: pd.DataFrame) -> "TransactionCostModel":
        """
        train_df must contain 'close', 'high', 'low' and a DatetimeIndex.
        Must be called with the TRAIN partition of a fold only.
        """
        required = {"close", "high", "low"}
        missing = required - set(train_df.columns)
        if missing:
            raise ValueError(f"train_df missing required columns: {sorted(missing)}")
        if not isinstance(train_df.index, pd.DatetimeIndex):
            raise TypeError("train_df index must be a DatetimeIndex.")
        if len(train_df) < 2:
            raise ValueError("train_df has too few rows to fit a cost model.")

        spread_frac = (
            (train_df["high"] - train_df["low"]) * self.spread_proxy_fraction_of_range
        ) / train_df["close"]
        spread_frac = spread_frac.replace([np.inf, -np.inf], np.nan).dropna()
        if spread_frac.empty:
            raise ValueError("no valid rows to fit the spread proxy from.")
        spread_p75_frac = float(np.percentile(spread_frac, self.spread_percentile))

        is_weekend = self._is_weekend(train_df.index)
        is_overnight = self._is_overnight(train_df.index)
        gap_w_freq, gap_w_mag = self._fit_gap_stats(train_df, is_weekend)
        gap_o_freq, gap_o_mag = self._fit_gap_stats(train_df, is_overnight)

        self._params = CostParameters(
            spread_p75_frac=spread_p75_frac,
            taker_fee_frac=self.taker_fee_frac,
            slippage_coefficient=self.slippage_coefficient,
            gap_frequency_weekend=gap_w_freq,
            gap_magnitude_weekend_frac=gap_w_mag,
            gap_frequency_overnight=gap_o_freq,
            gap_magnitude_overnight_frac=gap_o_mag,
            overnight_start_utc_hour=self.overnight_start_utc_hour,
            overnight_end_utc_hour=self.overnight_end_utc_hour,
            gap_threshold_mult=self.gap_threshold_mult,
            spread_proxy_fraction_of_range=self.spread_proxy_fraction_of_range,
            n_train_bars=len(train_df),
        )
        return self

    @staticmethod
    def _is_weekend(index: pd.DatetimeIndex) -> np.ndarray:
        return np.isin(np.asarray(index.dayofweek), [5, 6])

    def _is_overnight(self, index: pd.DatetimeIndex) -> np.ndarray:
        hour = np.asarray(index.hour)
        start, end = self.overnight_start_utc_hour, self.overnight_end_utc_hour
        if start <= end:
            return (hour >= start) & (hour < end)
        return (hour >= start) | (hour < end)  # wraps past midnight

    def _fit_gap_stats(self, train_df: pd.DataFrame, window_mask: np.ndarray) -> Tuple[float, float]:
        """
        Empirical frequency and mean magnitude of "gap events" -- candle-to-
        candle jumps larger than `gap_threshold_mult` times the recent
        typical candle range -- restricted to bars in `window_mask`
        (weekend or overnight). Both computed on TRAIN only.
        """
        close = train_df["close"].to_numpy(dtype=float)
        rng_frac = ((train_df["high"] - train_df["low"]) / train_df["close"]).to_numpy(dtype=float)
        typical_range_frac = (
            pd.Series(rng_frac, index=train_df.index).rolling(20, min_periods=5).median().to_numpy()
        )

        prev_close = np.empty_like(close)
        prev_close[0] = np.nan
        prev_close[1:] = close[:-1]
        jump_frac = np.abs(close - prev_close) / prev_close

        valid = window_mask & ~np.isnan(jump_frac) & ~np.isnan(typical_range_frac) & (typical_range_frac > 0)
        if not np.any(valid):
            return 0.0, 0.0

        is_gap = jump_frac[valid] > (self.gap_threshold_mult * typical_range_frac[valid])
        freq = float(np.mean(is_gap))
        if not np.any(is_gap):
            return freq, 0.0
        magnitude = float(np.mean(jump_frac[valid][is_gap]))
        return freq, magnitude

    # ------------------------------------------------------------------ #
    # Applying costs
    # ------------------------------------------------------------------ #

    def per_bar_fees_and_slippage(
        self, df: pd.DataFrame, position_size_frac: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (fees_array, slippage_array) aligned to `df.index`, suitable
        for vectorbt's `fees=`/`slippage=` order parameters. Both are
        one-sided cost fractions; vectorbt applies each independently to
        every order (entry and exit), so an entry+exit pair correctly nets
        out to a full round-trip cost.

        `fees` = taker fee + half the 75th-percentile spread (crossing the
        spread costs half of it on each side of a round trip).
        `slippage` = slippage_coefficient * |order size at that bar| *
        (candle range / close) -- MASTER_CONTEXT requires slippage to scale
        with both position size and candle range; this does both.

        "Order size" is the CHANGE in target position at that bar (the
        first bar's implicit prior position is flat/0), not the resulting
        post-order target level. This distinction matters: an order that
        closes a position out to flat has a target of 0 afterward, but the
        trade itself is not free -- you still have to sell the full prior
        position, which has real market impact. Scaling slippage by the
        post-order target instead of the traded delta would silently price
        every exit-to-flat order at zero slippage.
        """
        self._require_fitted()
        p = self._params
        required = {"close", "high", "low"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"df missing required columns: {sorted(missing)}")

        n = len(df)
        fees = np.full(n, p.taker_fee_frac + p.spread_p75_frac / 2.0, dtype=float)

        candle_range_frac = ((df["high"] - df["low"]) / df["close"]).to_numpy(dtype=float)
        aligned = position_size_frac.reindex(df.index).fillna(0.0)
        order_delta = aligned.diff()
        if len(order_delta) > 0:
            order_delta.iloc[0] = aligned.iloc[0]  # first bar's prior position is implicitly flat
        size = order_delta.abs().to_numpy(dtype=float)
        slippage = p.slippage_coefficient * size * candle_range_frac
        slippage = np.nan_to_num(slippage, nan=0.0, posinf=0.0, neginf=0.0)
        return fees, slippage

    def apply_gap_cost_to_trades(self, trades: pd.DataFrame, bar_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Adjusts each realized trade's PnL/return downward by the expected
        gap-holding cost for every weekend/overnight bar its
        [Entry Timestamp, Exit Timestamp] holding period spans. Must run
        BEFORE Sharpe/PF/EV are computed from these trades -- see the
        module docstring. `trades` must be vectorbt's
        `Portfolio.trades.records_readable` (or any frame with the same
        'Entry Timestamp'/'Exit Timestamp'/'Size'/'Avg Entry Price'/'PnL'
        columns) for a single backtest run over `bar_index`.
        """
        self._require_fitted()
        p = self._params
        trades = trades.copy()
        if trades.empty:
            trades["gap_cost"] = pd.Series(dtype=float)
            trades["net_pnl"] = pd.Series(dtype=float)
            trades["net_return"] = pd.Series(dtype=float)
            return trades

        is_weekend = self._is_weekend(bar_index)
        is_overnight = self._is_overnight(bar_index)
        weekend_cum = np.concatenate([[0], np.cumsum(is_weekend)])
        overnight_cum = np.concatenate([[0], np.cumsum(is_overnight)])

        entry_pos = bar_index.searchsorted(trades["Entry Timestamp"].to_numpy(), side="left")
        exit_pos = bar_index.searchsorted(trades["Exit Timestamp"].to_numpy(), side="right") - 1
        exit_pos = np.clip(exit_pos, 0, len(bar_index) - 1)

        n_weekend_held = weekend_cum[exit_pos + 1] - weekend_cum[entry_pos]
        n_overnight_held = overnight_cum[exit_pos + 1] - overnight_cum[entry_pos]

        expected_gap_cost_frac = (
            n_weekend_held * p.gap_frequency_weekend * p.gap_magnitude_weekend_frac
            + n_overnight_held * p.gap_frequency_overnight * p.gap_magnitude_overnight_frac
        )

        notional = (trades["Size"].abs() * trades["Avg Entry Price"]).to_numpy(dtype=float)
        notional_safe = np.where(notional > 0, notional, np.nan)
        gap_cost_value = expected_gap_cost_frac * notional

        trades["n_weekend_bars_held"] = n_weekend_held
        trades["n_overnight_bars_held"] = n_overnight_held
        trades["gap_cost"] = gap_cost_value
        trades["net_pnl"] = trades["PnL"].to_numpy(dtype=float) - gap_cost_value
        trades["net_return"] = np.nan_to_num(trades["net_pnl"].to_numpy() / notional_safe, nan=0.0)
        return trades

    def _require_fitted(self) -> None:
        if self._params is None:
            raise RuntimeError("TransactionCostModel.fit() must be called before use.")

    @property
    def params(self) -> CostParameters:
        self._require_fitted()
        return self._params
