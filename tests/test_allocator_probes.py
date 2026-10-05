"""Tests for the §1893 readiness probes in scripts/run_allocator_optimization_benchmark.py.

Covers the pure-numpy adjacent-rung panels and the critic width-scale wiring.
Heavy pieces (teacher, loaders, GPU training) are not unit-tested here.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parent.parent


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "run_allocator_optimization_benchmark",
        ROOT / "scripts/run_allocator_optimization_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_allocator_optimization_benchmark"] = module
    spec.loader.exec_module(module)
    return module


def test_adjacent_rung_panels_split_oracle_strata():
    """Low rung (oracle 0-1) and high rung (oracle 2-3) are disjoint and exhaustive."""
    module = _load_script()
    rng = np.random.default_rng(0)
    net = rng.uniform(-0.02, 0.02, size=(60, 4))
    oracle = net.argmax(axis=-1)
    critic_choices = oracle  # oracle critic: zero regret, gain over mode0 >= 0
    choices = {
        "allocator_critic": critic_choices,
        "uncertainty": np.zeros(60, dtype=int),
        "always_mode0": np.zeros(60, dtype=int),
        "random_expected": None,
    }
    panels = module.adjacent_rung_panels(net, choices)
    assert set(panels) == {"low_rung", "high_rung"}
    assert panels["low_rung"]["n"] + panels["high_rung"]["n"] == 60
    assert panels["low_rung"]["n"] == int((oracle <= 1).sum())
    # Near-oracle critic beats mode0 within each stratum (adaptive gain positive).
    for rung in ("low_rung", "high_rung"):
        if panels[rung]["n"] > 0:
            assert panels[rung]["adaptive_gain_trimmed"] >= 0.0


def test_adjacent_rung_panels_empty_stratum_is_nan_not_crash():
    """A stratum with no windows reports NaN, never a surrogate or zero."""
    module = _load_script()
    net = np.zeros((8, 4))
    net[:, 3] = 1.0  # oracle always mode 3 -> low rung empty
    choices = {
        "allocator_critic": np.full(8, 3),
        "uncertainty": np.zeros(8, dtype=int),
        "always_mode0": np.zeros(8, dtype=int),
        "random_expected": None,
    }
    panels = module.adjacent_rung_panels(net, choices)
    assert panels["low_rung"]["n"] == 0
    assert np.isnan(panels["low_rung"]["critic_trimmed"])
    assert panels["low_rung"]["vs_random"] is None
    assert panels["high_rung"]["n"] == 8


def test_critic_width_scale_changes_params_not_inputs():
    """Width scale changes capacity only: same input contract, monotone params."""
    from adjointrwm.allocators import DirectCritic, matched_critic_hidden

    d = 64
    base = matched_critic_hidden(d)
    counts = {}
    for scale in (0.5, 1.0, 2.0):
        hidden = max(1, round(base * scale))
        head = DirectCritic(d, hidden)
        counts[scale] = sum(p.numel() for p in head.parameters())
        sample = (torch.randn(2, d), torch.randn(2, 4, d), torch.zeros(4),
                  torch.ones(2), torch.ones(2))
        assert head(*sample).shape == (2, 4)
    assert counts[0.5] < counts[1.0] < counts[2.0]
