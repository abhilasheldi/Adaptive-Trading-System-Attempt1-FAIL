"""
Layer 10 -- Position Sizing and Execution-Layer Risk Caps. NOT IMPLEMENTED.

This module exists to make the ABSENCE of risk caps loud instead of silent.
Every function here raises `ExecutionRiskCapsNotImplemented`. That is the
entire current behaviour and it is deliberate.

=========================================================================
WHY THIS IS A STUB AND NOT AN IMPLEMENTATION
=========================================================================
Finding S10 (MASTER_CONTEXT_v2 section 2): "There are no execution-layer
risk caps in the codebase. The 2%/6%/10% constraints appear only in scoring
logic and comments." Grep-verified at the time, and re-verified at Stage 0:

  - 2%  single-trade risk        -- no implementation
  - 6%  correlated-direction     -- no implementation. `risk_engine.py`'s
                                    `CORRELATED_DIRECTION_CAP_PCT = 6.0` is
                                    an input to a 0-100 SCORE. It flags. It
                                    does not clamp, reject, or resize
                                    anything
  - 10% aggregate open risk      -- no implementation whatsoever

The thing that previously looked like a cap was `Phase2Strategy.max_position`
(default 0.02), deleted at Stage 0. It was never a risk cap, for two
independent reasons:

  1. It capped SIZE, not RISK. Risk = size x stop distance. With no stop,
     the risk of a 2%-of-capital position is not 2% -- it is unbounded.
     MASTER_CONTEXT's 2% figure is a cap on risk.
  2. It sat in the MODEL layer, where a model output could route around it.
     A real cap is enforced BELOW the model, on the order path, where no
     model output can raise it.

There are still no stops anywhere in this codebase (Layer 13, Stage E.2),
which is why (1) is not a hypothetical objection.

=========================================================================
WHAT STAGE E.1 MUST BUILD HERE
=========================================================================
Per BUILD_PLAN_v2 E.1, enforced below the model layer, unbypassable:

  - 2% max single-trade risk    (risk, not size -- REQUIRES stops from E.2)
  - 6% max correlated-direction exposure   (portfolio-level)
  - 10% max aggregate open risk            (portfolio-level)

The acceptance test named in E.1: a deliberately malicious model output
demanding 50% size must be clamped. Until then, calling into this module
raises rather than returning a number that would be unenforced.

Sizing is also gated on Stage D, which produces `P(success)` and the
`m = 2*Phi(z) - 1` sizing input. Stage E.3 specifies fixed-fractional
sizing for Stage 1 -- explicitly NO Kelly -- so this module must not grow a
Kelly implementation before that ordering is respected.

DO NOT soften these raises into a permissive default. A default that
returns a size is exactly the failure S10 describes: a number that looks
like a risk control and enforces nothing.
"""

# Declared here so the numbers live in the module that will enforce them,
# rather than only inside a scoring formula that cannot. These are MASTER_
# CONTEXT's exact figures. They are currently DECLARATIONS, not controls --
# nothing reads them yet, by construction.
SINGLE_TRADE_RISK_CAP_PCT = 2.0
CORRELATED_DIRECTION_RISK_CAP_PCT = 6.0
AGGREGATE_OPEN_RISK_CAP_PCT = 10.0

_MESSAGE = (
    "Execution-layer risk caps are not implemented (MASTER_CONTEXT_v2 finding "
    "S10). Position sizing is blocked until Stage E.1 builds the 2% single-trade, "
    "6% correlated-direction, and 10% aggregate open-risk caps below the model "
    "layer, and Stage E.2 builds the stops that make a RISK cap distinguishable "
    "from a SIZE cap. Refusing to return a size rather than returning an "
    "unenforced one."
)


class ExecutionRiskCapsNotImplemented(NotImplementedError):
    """Raised by every sizing entry point in this module.

    Subclasses `NotImplementedError` so a bare `except NotImplementedError`
    still catches it, but is named so a traceback says why sizing refused
    rather than merely that something was unfinished.
    """


def calculate_position_size(*args, **kwargs):
    """Would return a position size. Raises instead.

    This is the ONLY sanctioned entry point for turning a model signal into
    a position size. Anything that sizes without routing through here is
    sizing without caps, which is the S10 defect.
    """
    raise ExecutionRiskCapsNotImplemented(_MESSAGE)


def apply_execution_risk_caps(*args, **kwargs):
    """Would clamp a requested size to the 2%/6%/10% caps. Raises instead.

    Stage E.1's acceptance criterion attaches here: once implemented, a
    model output demanding 50% size must come back clamped, not honoured.
    """
    raise ExecutionRiskCapsNotImplemented(_MESSAGE)


__all__ = [
    "SINGLE_TRADE_RISK_CAP_PCT",
    "CORRELATED_DIRECTION_RISK_CAP_PCT",
    "AGGREGATE_OPEN_RISK_CAP_PCT",
    "ExecutionRiskCapsNotImplemented",
    "calculate_position_size",
    "apply_execution_risk_caps",
]
