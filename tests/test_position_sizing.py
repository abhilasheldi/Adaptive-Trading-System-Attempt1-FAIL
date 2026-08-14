"""
Stage 0 task 0.3 acceptance: "Tests fail loudly if anything sizes without
caps."

These tests assert a REFUSAL, not a behaviour. They are the guard that
stops the S10 defect from reappearing: a sizing path that returns a number
while the 2%/6%/10% execution-layer caps have no implementation behind it.

When Stage E.1 implements the caps, these tests are expected to fail. That
is the point -- they should be replaced at that moment with the E.1
acceptance test (a malicious model output demanding 50% size comes back
clamped), not deleted quietly to make a suite go green.
"""

import pytest

from src.risk_sizing_selection.position_sizing import (
    AGGREGATE_OPEN_RISK_CAP_PCT,
    CORRELATED_DIRECTION_RISK_CAP_PCT,
    SINGLE_TRADE_RISK_CAP_PCT,
    ExecutionRiskCapsNotImplemented,
    apply_execution_risk_caps,
    calculate_position_size,
)


def test_calculate_position_size_refuses_to_size_without_caps():
    with pytest.raises(ExecutionRiskCapsNotImplemented):
        calculate_position_size()


def test_apply_execution_risk_caps_refuses_without_an_implementation():
    with pytest.raises(ExecutionRiskCapsNotImplemented):
        apply_execution_risk_caps()


def test_sizing_refusal_survives_plausible_call_signatures():
    """A caller guessing at the eventual signature must still be refused --
    the raise cannot be routed around by passing arguments that look like
    the real ones."""
    with pytest.raises(ExecutionRiskCapsNotImplemented):
        calculate_position_size(p_success=0.9, equity=100_000.0, stop_distance_pct=1.0)
    with pytest.raises(ExecutionRiskCapsNotImplemented):
        apply_execution_risk_caps(requested_size_pct=50.0)


def test_refusal_is_catchable_as_not_implemented_error():
    """`ExecutionRiskCapsNotImplemented` must remain a `NotImplementedError`
    subclass, so generic handlers still catch it."""
    assert issubclass(ExecutionRiskCapsNotImplemented, NotImplementedError)
    with pytest.raises(NotImplementedError):
        calculate_position_size()


def test_refusal_message_names_the_reason():
    """A traceback must say WHY sizing refused, not merely that something is
    unfinished -- otherwise this stub reads as an unfinished feature rather
    than an enforced safety property."""
    with pytest.raises(ExecutionRiskCapsNotImplemented) as excinfo:
        calculate_position_size()
    message = str(excinfo.value)
    assert "S10" in message
    assert "2%" in message and "6%" in message and "10%" in message


def test_cap_values_are_the_exact_master_context_figures():
    assert SINGLE_TRADE_RISK_CAP_PCT == 2.0
    assert CORRELATED_DIRECTION_RISK_CAP_PCT == 6.0
    assert AGGREGATE_OPEN_RISK_CAP_PCT == 10.0


def test_no_module_level_sizing_function_returns_a_number():
    """Nothing in this module may quietly acquire a permissive path that
    returns a size. Enumerates the module's public callables and asserts
    every one refuses, so ADDING a working sizing function without also
    implementing caps fails here."""
    import src.risk_sizing_selection.position_sizing as sizing

    callables = [
        getattr(sizing, name)
        for name in sizing.__all__
        if callable(getattr(sizing, name)) and not isinstance(getattr(sizing, name), type)
    ]
    assert callables, "expected at least one sizing entry point to guard"
    for fn in callables:
        with pytest.raises(ExecutionRiskCapsNotImplemented):
            fn()
