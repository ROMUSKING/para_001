"""Fairness contract for the rival world-model benchmark (plan §3), enforced in code.

Each arm contributes one record describing how it was trained and scored. The benchmark is
reported only if every check passes; otherwise the notebook stops and the failing check is
written to ``fairness_contract.json`` (a failed contract is itself a reportable outcome).
"""

from __future__ import annotations

import hashlib
from typing import Mapping, Sequence

# Fields that must be identical for every arm in one comparison.
EQUAL_FIELDS = (
    "data_manifest_hash",
    "split_hash",
    "window_spec",
    "test_windows_hash",
    "input_encoder",
    "target_encoder",
    "steps",
    "batch_size",
    "seeds",
    "lr_grid",
    "tuning_steps",
    "gpu",
)


def windows_hash(episode_ids: Sequence[str], window_starts: Sequence[int]) -> str:
    """Order-independent hash of the set of scored windows."""
    keys = sorted(f"{e}:{int(s)}" for e, s in zip(episode_ids, window_starts))
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def split_hash(assignment: Mapping[str, str]) -> str:
    keys = sorted(f"{e}:{s}" for e, s in assignment.items())
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def _canonical(value):
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, dict):
        return {k: _canonical(v) for k, v in sorted(value.items())}
    return value


def check_fairness(records: Mapping[str, Mapping], reference: str, tolerance: float = 0.10) -> dict:
    """Check the contract across arms; returns ``{'passed', 'checks', ...}`` (never raises)."""
    if reference not in records:
        return {"passed": False, "checks": [{"check": "reference_present", "passed": False, "detail": reference}]}
    checks = []
    for field in EQUAL_FIELDS:
        values = {arm: _canonical(rec.get(field)) for arm, rec in records.items()}
        missing = sorted(arm for arm, v in values.items() if v is None)
        distinct = {repr(v) for v in values.values()}
        passed = not missing and len(distinct) == 1
        checks.append({"check": f"equal:{field}", "passed": passed,
                       "detail": {"missing": missing, "values": values} if not passed else values[reference]})
    target = records[reference].get("prediction_parameters")
    for arm, rec in records.items():
        params = rec.get("prediction_parameters")
        if not target or params is None:
            checks.append({"check": f"parameters:{arm}", "passed": False, "detail": "parameter count missing"})
            continue
        gap = (params - target) / target
        checks.append({"check": f"parameters:{arm}", "passed": abs(gap) <= tolerance,
                       "detail": {"parameters": int(params), "reference": int(target), "relative_gap": gap}})
    for arm, rec in records.items():
        # Every planned seed must end DONE or FAILED: a failed seed is reported, never
        # replaced or silently dropped (comprehensive plan, "Prohibited shortcuts").
        finished, failed = rec.get("seeds_finished"), rec.get("seeds_failed", [])
        planned = rec.get("seeds")
        passed = (
            finished is not None and planned is not None
            and sorted([*finished, *failed]) == sorted(planned) and not set(finished) & set(failed)
        )
        checks.append({"check": f"every_seed_accounted:{arm}", "passed": passed,
                       "detail": {"planned": planned, "finished": finished, "failed": failed}})
    return {
        "passed": all(c["passed"] for c in checks),
        "reference": reference,
        "tolerance": tolerance,
        "checks": checks,
    }
