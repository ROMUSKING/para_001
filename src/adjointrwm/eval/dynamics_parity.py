"""Same-split comparison of dynamics checkpoints against persistence (NumPy only).

The pilot's dynamics gate (``docs/research-notes/2026-09-29-droid100-pilot-findings.md``) and pilot v2's gate compare a model with persistence, the last
observed state repeated over the horizon, by the horizon-mean RMSE. These helpers compute that gate row for any predictions, so two checkpoints can be
scored on the same windows and their relative improvements compared.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .prediction import per_window_mse, persistence_forecast, rmse_by_horizon


def dynamics_gate_row(pred_state: np.ndarray, target_state: np.ndarray, context_state: np.ndarray, horizon: int, margin: float = 0.02) -> dict:
    """One dynamics-gate row: the model's horizon-mean RMSE, persistence's, the relative improvement ``(persistence - model) / persistence`` and whether it
    reaches ``margin``. ``pred_state`` and ``target_state`` are ``[windows, horizon, state_dim]``, ``context_state`` is ``[windows, context, state_dim]``; all in
    the same (normalised) units. The by-horizon lists show where a model beats persistence (persistence usually wins at one step ahead)."""
    pred, target, context = (np.asarray(x, dtype=np.float64) for x in (pred_state, target_state, context_state))
    if pred.shape != target.shape or pred.shape[1] != horizon or context.shape[0] != pred.shape[0]:
        raise ValueError(f"shape mismatch: pred {pred.shape}, target {target.shape}, context {context.shape}, horizon {horizon}")
    model_by_h = rmse_by_horizon(per_window_mse(pred, target))
    persistence_by_h = rmse_by_horizon(per_window_mse(persistence_forecast(context, horizon), target))
    model_rmse, persistence_rmse = float(model_by_h.mean()), float(persistence_by_h.mean())
    improvement = (persistence_rmse - model_rmse) / persistence_rmse
    return {
        "windows": int(pred.shape[0]), "model_rmse": model_rmse, "persistence_rmse": persistence_rmse, "relative_improvement": float(improvement),
        "required_margin": float(margin), "passed": bool(improvement >= margin),
        "model_rmse_by_horizon": [float(x) for x in model_by_h], "persistence_rmse_by_horizon": [float(x) for x in persistence_by_h],
        "improvement_by_horizon": [float((p - m) / p) for m, p in zip(model_by_h, persistence_by_h)],
    }


def compare_normalisers(a: Mapping, b: Mapping) -> dict:
    """How far two input normalisers (``state_mean``, ``state_std``, ... arrays) are apart: keys present in only one, shape mismatches, and the largest absolute
    difference per shared key and overall. Two checkpoints can only be compared in their native units if these agree closely."""
    shared = sorted(set(a) & set(b))
    shape_mismatch = sorted(k for k in shared if np.shape(a[k]) != np.shape(b[k]))
    diffs = {k: float(np.max(np.abs(np.asarray(a[k], dtype=np.float64) - np.asarray(b[k], dtype=np.float64)))) for k in shared if k not in shape_mismatch}
    return {"only_in_a": sorted(set(a) - set(b)), "only_in_b": sorted(set(b) - set(a)), "shape_mismatch": shape_mismatch, "max_abs_diff": diffs,
            "max_abs_diff_overall": max(diffs.values()) if diffs else None}


def anchor_check(recomputed: float, stored: float, atol: float) -> dict:
    """Whether a recomputed number reproduces the one the original run stored (within ``atol``). A diagnostic built on a pipeline that cannot reproduce the
    original number says nothing about it."""
    diff = abs(float(recomputed) - float(stored))
    return {"recomputed": float(recomputed), "stored": float(stored), "abs_diff": diff, "atol": float(atol), "within": bool(diff <= atol)}
