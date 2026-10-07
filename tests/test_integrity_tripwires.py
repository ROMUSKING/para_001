"""N0.2 prohibited-shortcut tripwires (governing comprehensive plan).

Each test reintroduces one listed shortcut and pins the corresponding helper's
failure in src/adjointrwm/analysis/allocation.py. Positive cases (clean inputs
pass) are asserted alongside so the tripwires cannot be satisfied vacuously.
"""

import numpy as np
import pytest

from adjointrwm.analysis.allocation import (
    assert_distinct_replicates,
    assert_predictions_ignore_futures,
    assert_single_hardware_stratum,
)


def test_hardware_stratum_passes_when_uniform():
    assert assert_single_hardware_stratum(
        [{"hardware": "L4", "ms": 1.0}, {"hardware": "L4", "ms": 2.0}]) == "L4"


def test_hardware_stratum_refuses_pooled_claim():
    """T4/L4/A100 timings must never be pooled into one systems claim."""
    with pytest.raises(ValueError, match="never pooled"):
        assert_single_hardware_stratum(
            [{"hardware": "L4", "ms": 1.0}, {"hardware": "A100", "ms": 0.4}])


def test_hardware_stratum_refuses_untagged_records():
    """An untagged timing is not a licence to pool."""
    with pytest.raises(ValueError, match="no 'hardware' tag"):
        assert_single_hardware_stratum([{"hardware": "L4"}, {"ms": 1.0}])


def test_distinct_replicates_passes_when_unique():
    assert assert_distinct_replicates(
        ["a" * 64, "b" * 64, "0123456789abcdef" * 4]) == 3


def test_distinct_replicates_refuses_silent_replacement():
    """A failed/aborted seed must never be silently replaced by a duplicate."""
    with pytest.raises(ValueError, match="silently replaced"):
        assert_distinct_replicates(["a" * 64, "b" * 64, "a" * 64])


def test_distinct_replicates_refuses_bare_integers_and_floats():
    """String equality on non-hash ids proves nothing: bare ints, short tokens
    and stringified floats are rejected even when unique."""
    with pytest.raises(ValueError, match="hex digest"):
        assert_distinct_replicates(["0", "1", "2"])
    with pytest.raises(ValueError, match="hex digest"):
        assert_distinct_replicates(["0.123456", "0.123457"])
    with pytest.raises(ValueError, match="hex digest"):
        assert_distinct_replicates(["deadbeef"])


def test_future_closure_passes_for_deployable_inputs():
    """A selector reading only deployment inputs is invariant to future swaps."""
    def predict(record):
        return [record["x"] * 2]

    assert_predictions_ignore_futures(
        predict, {"x": 1.0},
        [{"future_target": 99.0}, {"future_target": -3.0, "oracle_mask": [1]}]) is None


def test_future_closure_catches_privileged_callback():
    """A callback reading realised futures through its inputs is caught."""
    def predict(record):
        return [record["x"] + record.get("future_target", 0.0)]

    with pytest.raises(ValueError, match="never enter selector inputs"):
        assert_predictions_ignore_futures(
            predict, {"x": 1.0, "future_target": 0.0}, [{"future_target": 5.0}])


def test_future_closure_catches_float_score_drift():
    """Coverage extends to float scores, not just discrete choices."""
    def predict(record):
        return np.array([record["x"] + 1e-9 * record.get("realised_gain", 0.0)])

    with pytest.raises(ValueError, match="never enter selector inputs"):
        assert_predictions_ignore_futures(
            predict, {"x": 1.0}, [{"realised_gain": 100.0}])
