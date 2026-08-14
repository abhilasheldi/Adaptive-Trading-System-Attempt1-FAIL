"""
Ablation ranking and flagging, gated on solvency before performance.

Stage A.4 fix 4, Invariant I3.

=========================================================================
WHY THIS IS A MODULE AND NOT SCRIPT-LOCAL LOGIC
=========================================================================
The previous implementation lived inside `scripts/run_layer_ablation.py`,
which was deleted at Stage 0 along with the strategy it drove. Rebuilding
it in the next runner would rebuild the defect with it, so the rule lives
here, tested, and whatever Stage E ablation runner appears must call it.

=========================================================================
THE DEFECT THIS PREVENTS
=========================================================================
The old flagging ranked variants by OOS Sharpe and flagged whichever layer
lowered it. On the 2026-08-14 ablation that produced a directly wrong
verdict: it flagged Layer 6 (the volatility gate) as the degrading layer,
because removing Layer 6 RAISED Sharpe from -8.284 to -4.846.

Both of those runs had maxDD above 1.0 -- 1.023 and 1.235. Equity had gone
through zero. Every return past that crossing was computed against a
zero-or-negative denominator, so both Sharpes were arithmetic on a
destroyed account, and the "improvement" from removing Layer 6 was noise in
a meaningless quantity. The layer actually destroying the account was 7C,
which the Sharpe-only rule reported as HELPING (+4.075 Sharpe).

Invariant I3: "Ablation flagging gates on maxDD and PF BEFORE Sharpe.
Refuse ranking when any_equity_destroyed."

The ordering is not a preference. A Sharpe computed over a non-positive
equity curve is not a worse measurement of the same thing -- it is not a
measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

# maxDD >= this means equity reached zero. Post-S11 the curve is floored, so
# a value of exactly 1.0 is the signature of ruin rather than an overshoot.
RUIN_MAX_DD = 1.0


@dataclass(frozen=True)
class VariantOutcome:
    """One ablation variant's headline numbers."""

    name: str
    oos_sharpe: float
    oos_profit_factor: float
    oos_max_drawdown: float
    oos_trades: int
    equity_destroyed: bool = False

    @property
    def is_solvent(self) -> bool:
        """Solvency gate. Checked BEFORE any performance comparison."""
        if self.equity_destroyed:
            return False
        dd = self.oos_max_drawdown
        return not (np.isnan(dd) or dd >= RUIN_MAX_DD)


@dataclass
class AblationVerdict:
    rankable: bool
    reason: str
    ranked: List[VariantOutcome] = field(default_factory=list)
    insolvent: List[VariantOutcome] = field(default_factory=list)
    flagged: Dict[str, List[str]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"rankable={self.rankable}: {self.reason}"]
        if self.insolvent:
            lines.append("INSOLVENT (excluded from ranking): "
                         + ", ".join(v.name for v in self.insolvent))
        for name, reasons in self.flagged.items():
            lines.append(f"FLAGGED {name}: " + "; ".join(reasons))
        return "\n".join(lines)


def rank_variants(variants: Sequence[VariantOutcome]) -> AblationVerdict:
    """Rank ablation variants, solvency first.

    Refuses to rank at all when ANY variant destroyed equity. That is
    deliberate and stronger than merely excluding the insolvent ones: an
    ablation compares variants against each other, and a comparison set
    containing a destroyed account has no common scale. Reporting "variant B
    beat variant A" when A blew up is not a ranking, it is a category error.
    """
    variants = list(variants)
    if not variants:
        return AblationVerdict(False, "no variants supplied")

    insolvent = [v for v in variants if not v.is_solvent]
    solvent = [v for v in variants if v.is_solvent]

    if insolvent:
        return AblationVerdict(
            rankable=False,
            reason=(
                f"{len(insolvent)} of {len(variants)} variants destroyed equity "
                f"(maxDD >= {RUIN_MAX_DD} or equity_destroyed). Sharpe over a "
                f"non-positive equity curve is not a measurement, so no ranking "
                f"is published (I3). Read maxDD and profit factor instead."
            ),
            ranked=[], insolvent=insolvent,
            flagged={v.name: ["equity destroyed"] for v in insolvent},
        )

    # Solvent set: rank on maxDD, then PF, then Sharpe -- in that order.
    ranked = sorted(
        solvent,
        key=lambda v: (
            v.oos_max_drawdown if not np.isnan(v.oos_max_drawdown) else np.inf,
            -(v.oos_profit_factor if not np.isnan(v.oos_profit_factor) else -np.inf),
            -(v.oos_sharpe if not np.isnan(v.oos_sharpe) else -np.inf),
        ),
    )
    return AblationVerdict(
        rankable=True,
        reason="all variants solvent; ranked on maxDD, then profit factor, then Sharpe",
        ranked=ranked,
    )


def flag_degrading_layers(
    full_stack: VariantOutcome,
    leave_one_out: Dict[str, VariantOutcome],
) -> Dict[str, List[str]]:
    """Flag layers whose REMOVAL improves the run on solvency/PF/maxDD/Sharpe.

    Evaluated in that order. A layer is flagged if removing it improves any
    higher-priority criterion, and Sharpe is consulted ONLY when both runs
    are solvent -- which is precisely the check the old Sharpe-only rule
    skipped when it exonerated Layer 7C.
    """
    flagged: Dict[str, List[str]] = {}
    for layer, without in leave_one_out.items():
        reasons: List[str] = []

        if not full_stack.is_solvent and without.is_solvent:
            reasons.append("removing it restores solvency")

        if not (np.isnan(without.oos_max_drawdown) or np.isnan(full_stack.oos_max_drawdown)):
            if without.oos_max_drawdown < full_stack.oos_max_drawdown:
                reasons.append(
                    f"max drawdown {full_stack.oos_max_drawdown:.3f} -> "
                    f"{without.oos_max_drawdown:.3f} without it"
                )

        if not (np.isnan(without.oos_profit_factor) or np.isnan(full_stack.oos_profit_factor)):
            if without.oos_profit_factor > full_stack.oos_profit_factor:
                reasons.append(
                    f"profit factor {full_stack.oos_profit_factor:.3f} -> "
                    f"{without.oos_profit_factor:.3f} without it"
                )

        if full_stack.is_solvent and without.is_solvent:
            if not (np.isnan(without.oos_sharpe) or np.isnan(full_stack.oos_sharpe)):
                if without.oos_sharpe > full_stack.oos_sharpe:
                    reasons.append(
                        f"Sharpe {full_stack.oos_sharpe:.3f} -> "
                        f"{without.oos_sharpe:.3f} without it"
                    )

        if reasons:
            flagged[layer] = reasons
    return flagged


__all__ = [
    "RUIN_MAX_DD",
    "VariantOutcome",
    "AblationVerdict",
    "rank_variants",
    "flag_degrading_layers",
]
