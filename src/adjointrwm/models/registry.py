"""Arm registry and parameter matching for the rival world-model benchmark."""

from __future__ import annotations

from typing import Callable

import torch

from .adjoint_rwm import build_adjoint_rwm
from .common import ArmDims, WorldModel
from .feature_predictor import build_dino_wm, build_vjepa2_ac
from .hybrid_adjoint import build_hybrid_adjoint_rwm
from .rssm import build_rssm
from .spatial_adjoint import build_spatial_adjoint_rwm
from .tdmpc2 import build_tdmpc2

REFERENCE_ARM = "adjoint_rwm"

BUILDERS: dict[str, Callable[..., WorldModel]] = {
    "adjoint_rwm": build_adjoint_rwm,
    "spatial_adjoint_rwm": build_spatial_adjoint_rwm,
    "hybrid_adjoint_rwm": build_hybrid_adjoint_rwm,
    "dreamerv3_rssm": build_rssm,
    "tdmpc2": build_tdmpc2,
    "dino_wm": build_dino_wm,
    "vjepa2_ac": build_vjepa2_ac,
}

# Width knob per arm: d_model / hidden / latent+mlp / predictor dim. Steps respect each arm's
# divisibility constraints (heads, blocks x classes, SimNorm groups).
WIDTH_GRIDS: dict[str, range] = {
    "adjoint_rwm": range(128, 1025, 64),
    "spatial_adjoint_rwm": range(128, 1025, 64),
    "hybrid_adjoint_rwm": range(128, 1025, 64),
    "dreamerv3_rssm": range(64, 1025, 16),
    "tdmpc2": range(128, 4097, 32),
    "dino_wm": range(64, 1537, 16),
    "vjepa2_ac": range(64, 1025, 32),
}

# Recipe hyperparameters that the official configs fix (recorded in the fairness manifest).
RECIPE_DEFAULTS: dict[str, dict] = {
    "adjoint_rwm": {"grad_clip": 1.0},
    "spatial_adjoint_rwm": {"grad_clip": 1.0},
    "hybrid_adjoint_rwm": {"grad_clip": 1.0},
    "dreamerv3_rssm": {"grad_clip": 1.0},
    "tdmpc2": {"grad_clip": 20.0},
    "dino_wm": {"grad_clip": 1.0},
    "vjepa2_ac": {"grad_clip": 1.0},
}


def build_arm(name: str, dims: ArmDims, width: int | None = None, **overrides) -> WorldModel:
    if name not in BUILDERS:
        raise KeyError(f"unknown arm {name!r}; choose from {sorted(BUILDERS)}")
    return BUILDERS[name](dims, width=width, **overrides)


def count_prediction_parameters(name: str, dims: ArmDims, width: int | None = None, **overrides) -> int:
    """Prediction-path trainable parameters, counted without allocating memory."""
    with torch.device("meta"):
        model = build_arm(name, dims, width=width, **overrides)
    return model.prediction_parameters()


def match_width(
    name: str,
    dims: ArmDims,
    target: int,
    tolerance: float = 0.10,
    grid: range | list | None = None,
    **overrides,
) -> dict:
    """Width whose prediction-path parameter count is closest to ``target``.

    Returns ``{'width', 'parameters', 'relative_gap', 'within_tolerance'}``. Parameter counts
    grow with width, so the scan stops once a count exceeds twice the target.
    """
    best = None
    for width in grid or WIDTH_GRIDS[name]:
        try:
            params = count_prediction_parameters(name, dims, width=width, **overrides)
        except ValueError:
            continue  # width violates an arm constraint
        gap = (params - target) / target
        if best is None or abs(gap) < abs(best["relative_gap"]):
            best = {"width": int(width), "parameters": int(params), "relative_gap": float(gap)}
        if params > 2 * target:
            break
    if best is None:
        raise ValueError(f"no valid width for {name}")
    best["within_tolerance"] = abs(best["relative_gap"]) <= tolerance
    best["target"] = int(target)
    best["tolerance"] = tolerance
    return best
