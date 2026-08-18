"""
Base currency GBP -- BUILD_PLAN_v2 B.0.6.

The account is denominated in GBP. Instruments quote and settle in USDT,
so every equity curve, P&L figure, drawdown and risk cap is a GBP quantity
that happens to be computed from USD-quoted prices.

=========================================================================
THE CONVERSION
=========================================================================
FRED `DEXUSUK` is **U.S. Dollars per ONE British Pound**, so

    gbp = usd / gbpusd

and NOT a multiplication. Getting this backwards produces a curve that is
wrong by a factor of ~1.6-squared and still looks entirely plausible on a
chart, which is why the direction is asserted in the tests rather than
trusted to the reader.

=========================================================================
POINT-IN-TIME, AND THE WEEKEND
=========================================================================
The rate applied at bar `t` is the last rate OBSERVABLE at `t`, via the
same `point_in_time` machinery as every other series (I9). DEXUSUK is a
daily FRED series and carries the same one-business-day publication floor
(B.0.4).

Crypto trades through the weekend; the FX fix does not. Friday's rate is
therefore forward-filled across Saturday and Sunday, which is I9-compliant
because it is the last rate anyone could actually have observed. It is not
the rate a Sunday conversion would really have transacted at -- there is no
such rate, the market is shut -- and that gap is a real, unhedgeable
exposure rather than a modelling artifact. Recorded here so a weekend
drawdown measured in GBP is read as the exposure it is.

=========================================================================
WHAT CHANGES AND WHAT DOES NOT
=========================================================================
**Percentage risk caps are currency-invariant.** 2% of GBP equity buys the
same position as 2% of the equivalent USD equity: the FX rate cancels. So
the 2/6/10% caps do not need GBP-specific numbers -- what matters is that
they are percentages OF THE GBP EQUITY, which is what `RiskCapBasis` below
pins down. (The caps themselves remain unimplemented, S10; E.1 builds them.
This module fixes the base they will apply to.)

**Gate metrics are NOT currency-invariant.** Sharpe, max drawdown and
profit factor all change, because FX variance enters the return series. A
strategy at 18% USD drawdown can breach the 20% GBP gate on a bad month for
sterling without a single trade going differently. That is not noise to be
corrected away -- it is the drawdown the account actually experiences, and
the GBP figure is therefore the one the gate reads.

Both are reported side by side so the FX contribution is visible rather
than absorbed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.backtesting import metrics
from src.data_engine.point_in_time import (
    FRED_DAILY,
    ObservationPolicy,
    align_point_in_time,
)

GBPUSD_COLUMN = "gbpusd"


class CurrencyError(RuntimeError):
    """Raised when a conversion cannot be performed honestly."""


@dataclass(frozen=True)
class RiskCapBasis:
    """Which equity the percentage risk caps are percentages of.

    Exists so that E.1 cannot quietly implement the caps against the USD
    book. The caps are currency-invariant as ratios; the BASE is not.
    """

    currency: str = "GBP"
    max_single_trade_risk_pct: float = 2.0
    max_correlated_direction_pct: float = 6.0
    max_aggregate_open_risk_pct: float = 10.0

    def describe(self) -> str:
        return (
            f"Risk caps are percentages of {self.currency} equity: "
            f"{self.max_single_trade_risk_pct}% single trade / "
            f"{self.max_correlated_direction_pct}% correlated direction / "
            f"{self.max_aggregate_open_risk_pct}% aggregate open risk. "
            f"Ratios are currency-invariant; the base is not (B.0.6). "
            f"Enforcement is Stage E.1 -- these caps have no implementation yet (S10)."
        )


def gbpusd_at_bars(
    macro: pd.DataFrame,
    bar_index: pd.DatetimeIndex,
    policy: ObservationPolicy = FRED_DAILY,
    column: str = GBPUSD_COLUMN,
) -> pd.Series:
    """Point-in-time USD-per-GBP rate on `bar_index`.

    Weekend forward-fill falls out of the point-in-time alignment: Friday's
    rate is simply the most recent observable one until Monday's publishes.
    """
    if column not in macro.columns:
        raise CurrencyError(
            f"{column!r} not in the macro frame (have: {list(macro.columns)}). "
            f"Ingest FRED DEXUSUK -- B.0.6 makes it a required series, because "
            f"without it the account has no base currency and every reported "
            f"number is in the wrong one."
        )
    rate = align_point_in_time(macro[[column]], bar_index, policy)[column]
    return rate.rename(column)


def usd_to_gbp(usd: pd.Series, gbpusd: pd.Series) -> pd.Series:
    """Convert a USD-denominated series to GBP at the point-in-time rate.

    DEXUSUK is USD per GBP, so this DIVIDES. Bars with no observable rate
    stay NaN -- never silently carried at 1.0, which would report USD
    numbers as though they were GBP.
    """
    rate = gbpusd.reindex(usd.index)
    if rate.isna().all():
        raise CurrencyError(
            "No observable GBP/USD rate on any bar in this segment. Converting "
            "would mean inventing one."
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        out = usd / rate.replace(0.0, np.nan)
    return out.rename(usd.name)


def convert_equity_curve(usd_equity: pd.Series, gbpusd: pd.Series) -> pd.Series:
    """GBP equity curve, floored at zero to preserve S11.

    The floor is reapplied after conversion: a curve floored in USD is
    floored in GBP too (the rate is positive), but the floor is asserted
    rather than inferred so a negative rate in bad data cannot smuggle a
    negative equity through.
    """
    gbp = usd_to_gbp(usd_equity, gbpusd)
    if (gbp.dropna() < 0).any():
        raise CurrencyError(
            "Negative GBP equity after conversion -- implies a non-positive FX "
            "rate. Bad data, not a bad strategy."
        )
    return gbp


@dataclass(frozen=True)
class DualCurrencyResult:
    """Metrics in both currencies. GBP is the gate; USD is the reference."""

    gbp_equity: pd.Series
    usd_equity: pd.Series
    gbp: dict
    usd: dict
    fx_coverage: float

    def table(self) -> str:
        rows = [
            ("final equity", "gbp_final", "usd_final", "{:,.2f}"),
            ("total return", "total_return", "total_return", "{:.2%}"),
            ("Sharpe", "sharpe", "sharpe", "{:.3f}"),
            ("max drawdown", "max_drawdown", "max_drawdown", "{:.2%}"),
        ]
        out = [
            f"{'metric':<18}{'GBP (gate)':>16}{'USD (reference)':>18}",
            "-" * 52,
        ]
        for label, gk, uk, fmt in rows:
            g = self.gbp.get("final" if gk.endswith("final") else gk)
            u = self.usd.get("final" if uk.endswith("final") else uk)
            out.append(
                f"{label:<18}{_fmt(g, fmt):>16}{_fmt(u, fmt):>18}"
            )
        out.append("")
        out.append(f"FX coverage: {self.fx_coverage:.1%} of bars carry an observable rate.")
        out.append(
            "GBP is the reported base currency and the one the Tier 1 gates read. "
            "Differences are FX, not strategy (B.0.6)."
        )
        return "\n".join(out)


def _fmt(v, fmt: str) -> str:
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt.format(v)


def dual_currency_report(
    usd_equity: pd.Series,
    gbpusd: pd.Series,
    bars_per_year: Optional[float] = None,
) -> DualCurrencyResult:
    """Compute the gate metrics in both currencies, side by side.

    GBP first and labelled as the gate, so a reader cannot mistake which
    column the Tier 1 thresholds are applied to.
    """
    gbp_equity = convert_equity_curve(usd_equity, gbpusd)
    if bars_per_year is None:
        bars_per_year = metrics.infer_bars_per_year(usd_equity.index)

    def summarise(curve: pd.Series) -> dict:
        clean = curve.dropna()
        if len(clean) < 2:
            return {"final": float("nan"), "total_return": float("nan"),
                    "sharpe": float("nan"), "max_drawdown": float("nan")}
        rets = clean.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        return {
            "final": float(clean.iloc[-1]),
            "total_return": float(clean.iloc[-1] / clean.iloc[0] - 1.0),
            "sharpe": metrics.sharpe_ratio_from_returns(rets, bars_per_year),
            "max_drawdown": metrics.max_drawdown(clean),
        }

    coverage = float(gbpusd.reindex(usd_equity.index).notna().mean())
    return DualCurrencyResult(
        gbp_equity=gbp_equity,
        usd_equity=usd_equity,
        gbp=summarise(gbp_equity),
        usd=summarise(usd_equity),
        fx_coverage=coverage,
    )


__all__ = [
    "CurrencyError",
    "RiskCapBasis",
    "DualCurrencyResult",
    "GBPUSD_COLUMN",
    "gbpusd_at_bars",
    "usd_to_gbp",
    "convert_equity_curve",
    "dual_currency_report",
]
