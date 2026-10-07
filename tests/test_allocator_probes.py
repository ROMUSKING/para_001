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


def test_write_seed_partial_schema_and_last_wins(tmp_path):
    """Partial flush keeps the latest entry per checkpoint plus the seed's ledger slice."""
    import argparse

    module = _load_script()
    args = argparse.Namespace(critic_width_scale=0.5, critic_only=True,
                              eval_checkpoints=[300, 1000], probe_panels=True)
    cr = {300: [{"a": 1}], 1000: [{"a": 2}, {"a": 3}]}
    vr = {300: [{"b": 4}], 1000: []}
    vp = {300: [{"p": 5}], 1000: [{"p": 6}]}
    ledger = [{"seed": 2, "step": 200, "train_critic_loss": 0.5},
              {"seed": 3, "step": 200, "train_critic_loss": 0.6}]
    path = module.write_seed_partial(tmp_path, 2, args, cr, vr, vp, ledger, True)
    import json as _json

    doc = _json.loads(path.read_text())
    assert doc["seed"] == 2 and doc["finite"] is True
    assert doc["critic_width_scale"] == 0.5 and doc["critic_only"] is True
    assert doc["checkpoints"] == {"300": {"a": 1}, "1000": {"a": 3}}
    assert doc["val_checkpoints"] == {"300": {"b": 4}}
    assert doc["val_panels"] == {"300": {"p": 5}, "1000": {"p": 6}}
    assert doc["surrogate_ledger"] == [ledger[0]]
    assert path.name == "partial_seed_2.json"


def test_adjacent_rung_panels_critic_only_subset():
    """Critic-only runs omit costate policies: panels need only critic/uncertainty/mode0."""
    module = _load_script()
    rng = np.random.default_rng(1)
    net = rng.uniform(-0.02, 0.02, size=(24, 4))
    oracle = net.argmax(axis=-1)
    choices = {
        "allocator_critic": oracle,
        "uncertainty": np.zeros(24, dtype=int),
        "always_mode0": np.zeros(24, dtype=int),
        "random_expected": None,
    }
    panels = module.adjacent_rung_panels(net, choices)
    assert panels["low_rung"]["n"] + panels["high_rung"]["n"] == 24
