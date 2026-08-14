"""
Trial registry -- `trials.jsonl`.

BUILD_PLAN_v2 A.5, MASTER_CONTEXT_v2 section 6:

    "Every distinct model configuration evaluated is logged to trials.jsonl
     with config hash, date, and result. This count feeds the Deflated
     Sharpe Ratio. Optuna trials count. Manual runs count. You cannot
     compute an honest DSR without an honest trial count, and DSR is a
     gate."

One JSON object per line:

    {config_hash, timestamp, stage, params, n_eff, result_metrics}

WHY APPEND-ONLY JSONL AND NOT A DATABASE
The registry's only job is to be hard to quietly shrink. A line-oriented
append-only file is auditable with `wc -l`, diffable in git, survives a
crash mid-write with at most one corrupt trailing line, and cannot be
"cleaned up" by an ORM migration. A trial count that can be silently
reduced is worse than no trial count, because DSR computed from it looks
rigorous while overstating significance.

NOTHING IN THIS MODULE DELETES OR REWRITES. There is no update function and
no delete function, deliberately. Correcting a mistaken entry is done by
appending a superseding record, never by editing history.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

DEFAULT_REGISTRY_PATH = Path("trials.jsonl")


def _canonical(obj: Any) -> Any:
    """Recursively canonicalise for hashing: sorted keys, floats normalised.

    Two configurations that differ only in dict ordering, or in int-vs-float
    representation of the same value, must hash identically -- otherwise the
    trial count inflates on cosmetic differences and DSR becomes too
    conservative rather than too permissive.
    """
    if isinstance(obj, dict):
        return {k: _canonical(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)):
        f = float(obj)
        return round(f, 12) + 0.0     # collapse -0.0 and float noise
    return obj


def config_hash(params: Dict[str, Any], stage: Optional[str] = None) -> str:
    """Stable SHA-256 over the canonicalised configuration.

    `stage` participates in the hash when given: the same hyperparameters
    evaluated at Stage C and at Stage D are two distinct evaluations of the
    hypothesis space, and DSR must count them as two.
    """
    payload = {"stage": stage, "params": _canonical(params)}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class TrialRecord:
    config_hash: str
    timestamp: str
    stage: str
    params: Dict[str, Any]
    n_eff: Optional[float]                       # I6: never omitted, may be null
    result_metrics: Dict[str, Any] = field(default_factory=dict)
    nominal_n: Optional[int] = None
    source: str = "manual"                       # 'optuna' | 'manual' | 'backfill'
    note: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=str)


def log_trial(
    stage: str,
    params: Dict[str, Any],
    result_metrics: Optional[Dict[str, Any]] = None,
    n_eff: Optional[float] = None,
    nominal_n: Optional[int] = None,
    source: str = "manual",
    note: Optional[str] = None,
    path: Path | str = DEFAULT_REGISTRY_PATH,
) -> TrialRecord:
    """Append one evaluated configuration. Returns the record written.

    `n_eff` is Optional because it is genuinely not computable for some
    evaluations (no label metadata on the frame). It is written as null in
    that case rather than omitted or defaulted to nominal n -- an absent
    measurement is not a measurement of no overlap (I5).
    """
    rec = TrialRecord(
        config_hash=config_hash(params, stage),
        timestamp=datetime.now(timezone.utc).isoformat(),
        stage=stage,
        params=params,
        n_eff=None if n_eff is None else float(n_eff),
        result_metrics=result_metrics or {},
        nominal_n=nominal_n,
        source=source,
        note=note,
    )
    p = Path(path)
    if p.parent and str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)
    # Append + flush + fsync: a trial that ran but was not recorded makes the
    # registry undercount, which is the one direction of error that flatters
    # the DSR.
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(rec.to_json() + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return rec


def read_trials(path: Path | str = DEFAULT_REGISTRY_PATH) -> List[TrialRecord]:
    """All records. A malformed trailing line (crash mid-append) is skipped
    with the count of skips available via `registry_summary`."""
    p = Path(path)
    if not p.exists():
        return []
    out: List[TrialRecord] = []
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(TrialRecord(**json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue
    return out


def distinct_configuration_count(path: Path | str = DEFAULT_REGISTRY_PATH) -> int:
    """Number of DISTINCT configurations -- the N that feeds DSR.

    Distinct, not total lines: re-evaluating an identical configuration does
    not widen the searched hypothesis space, so it must not inflate the
    deflation term.
    """
    return len({r.config_hash for r in read_trials(path)})


def registry_summary(path: Path | str = DEFAULT_REGISTRY_PATH) -> Dict[str, Any]:
    trials = read_trials(path)
    by_stage: Dict[str, int] = {}
    by_source: Dict[str, int] = {}
    for r in trials:
        by_stage[r.stage] = by_stage.get(r.stage, 0) + 1
        by_source[r.source] = by_source.get(r.source, 0) + 1
    return {
        "total_records": len(trials),
        "distinct_configurations": len({r.config_hash for r in trials}),
        "by_stage": dict(sorted(by_stage.items())),
        "by_source": dict(sorted(by_source.items())),
        "with_n_eff": sum(1 for r in trials if r.n_eff is not None),
    }


# ---------------------------------------------------------------------- #
# Optuna integration
# ---------------------------------------------------------------------- #


def optuna_logging_callback(
    stage: str,
    extra_params: Optional[Dict[str, Any]] = None,
    n_eff: Optional[float] = None,
    nominal_n: Optional[int] = None,
    path: Path | str = DEFAULT_REGISTRY_PATH,
):
    """Callback for `study.optimize(..., callbacks=[...])`.

    Logs EVERY trial, not just the best one. That is the point: the trial
    count DSR needs is the size of the hypothesis space actually searched,
    and a search that tried 15 configurations and kept 1 has still searched
    15. Logging only the winner is precisely how DSR gets flattered.
    """

    def _cb(study, trial):
        params = dict(trial.params)
        if extra_params:
            params.update(extra_params)
        log_trial(
            stage=stage,
            params=params,
            result_metrics={
                "objective_value": None if trial.value is None else float(trial.value),
                "optuna_trial_number": trial.number,
                "state": str(trial.state),
            },
            n_eff=n_eff,
            nominal_n=nominal_n,
            source="optuna",
            path=path,
        )

    return _cb


__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "TrialRecord",
    "config_hash",
    "log_trial",
    "read_trials",
    "distinct_configuration_count",
    "registry_summary",
    "optuna_logging_callback",
]
