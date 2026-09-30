"""Same-split comparison of dynamics checkpoints against persistence (NumPy only).

The pilot's dynamics gate (``docs/research-notes/2026-09-29-droid100-pilot-findings.md``) and pilot v2's gate compare a model with persistence, the last
observed state repeated over the horizon, by the horizon-mean RMSE. These helpers compute that gate row for any predictions, so two checkpoints can be
scored on the same windows and their relative improvements compared.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..data.windows import split_order_key
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


# ---- episode-level k-fold study (docs/plans/dynamics-kfold-plan.md) -----------------------------------------------------------------------------

CLASSES = ("BEATS", "WORSE_THAN_PERSISTENCE", "FAILS", "INCONCLUSIVE")


def kfold_assignment(episode_ids: Iterable[str], k: int = 5) -> dict[str, int]:
    """Fold of each episode: its rank by ascending SHA-256 of the id (the pilot split's ordering rule), modulo ``k``. Every episode gets exactly one fold."""
    ids = list(episode_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("episode ids must be unique before splitting")
    if k < 2 or k > len(ids):
        raise ValueError(f"k must be between 2 and the number of episodes ({len(ids)}), got {k}")
    return {episode_id: rank % k for rank, episode_id in enumerate(sorted(ids, key=split_order_key))}


def inner_validation_split(pool_ids: Iterable[str], n_validation: int) -> tuple[list[str], list[str]]:
    """``(training ids, inner validation ids)`` of a training pool: in the hash order, the first ``n_validation`` episodes are validation, the rest training."""
    ordered = sorted(pool_ids, key=split_order_key)
    if not 0 < n_validation < len(ordered):
        raise ValueError(f"n_validation must be between 1 and {len(ordered) - 1}, got {n_validation}")
    return ordered[n_validation:], ordered[:n_validation]


def episode_error_table(pred_state: np.ndarray, target_state: np.ndarray, context_state: np.ndarray, episode_ids: Sequence[str], horizon: int) -> pd.DataFrame:
    """Per episode: the window count and, per horizon step, the sum over windows of the state-dimension-mean squared error of the model and of persistence.

    ``episode_ids`` has one entry per window. Summed squared errors (not means) so that episodes pool exactly: pooled MSE is the sum over episodes divided by the
    total window count."""
    pred, target, context = (np.asarray(x, dtype=np.float64) for x in (pred_state, target_state, context_state))
    ids = np.asarray(list(episode_ids))
    if pred.shape != target.shape or pred.shape[1] != horizon or len(ids) != pred.shape[0] or context.shape[0] != pred.shape[0]:
        raise ValueError(f"shape mismatch: pred {pred.shape}, target {target.shape}, context {context.shape}, {len(ids)} episode ids, horizon {horizon}")
    model = per_window_mse(pred, target)                                         # [windows, horizon]
    persistence = per_window_mse(persistence_forecast(context, horizon), target)
    rows = []
    for episode_id in sorted(set(ids.tolist())):
        mask = ids == episode_id
        rows.append({"episode_id": episode_id, "windows": int(mask.sum()),
                     **{f"model_sse_h{h + 1}": float(model[mask, h].sum()) for h in range(horizon)},
                     **{f"persistence_sse_h{h + 1}": float(persistence[mask, h].sum()) for h in range(horizon)}})
    return pd.DataFrame(rows)


def _arrays(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizon = sum(c.startswith("model_sse_h") for c in table.columns)
    n = table["windows"].to_numpy(dtype=np.float64)
    model = table[[f"model_sse_h{h + 1}" for h in range(horizon)]].to_numpy(dtype=np.float64)
    persistence = table[[f"persistence_sse_h{h + 1}" for h in range(horizon)]].to_numpy(dtype=np.float64)
    return n, model, persistence


def _relative_improvement(n: np.ndarray, model: np.ndarray, persistence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Horizon-mean relative improvement ``(P - M) / P`` and the per-horizon ones, for summed squared errors ``[..., episodes, horizon]`` and counts ``[..., episodes]``."""
    total = n.sum(axis=-1)[..., None]
    rmse_model = np.sqrt(model.sum(axis=-2) / total)
    rmse_persistence = np.sqrt(persistence.sum(axis=-2) / total)
    by_horizon = (rmse_persistence - rmse_model) / rmse_persistence
    overall = (rmse_persistence.mean(axis=-1) - rmse_model.mean(axis=-1)) / rmse_persistence.mean(axis=-1)
    return overall, by_horizon


def pooled_relative_improvement(table: pd.DataFrame) -> dict:
    """The pooled estimate of an episode table: horizon-mean RMSE of the model and of persistence, the relative improvement and the per-horizon ones."""
    n, model, persistence = _arrays(table)
    total = n.sum()
    rmse_model, rmse_persistence = np.sqrt(model.sum(axis=0) / total), np.sqrt(persistence.sum(axis=0) / total)
    overall, by_horizon = _relative_improvement(n, model, persistence)
    return {"episodes": int(len(table)), "windows": int(total), "model_rmse": float(rmse_model.mean()), "persistence_rmse": float(rmse_persistence.mean()),
            "relative_improvement": float(overall), "improvement_by_horizon": [float(x) for x in by_horizon],
            "model_rmse_by_horizon": [float(x) for x in rmse_model], "persistence_rmse_by_horizon": [float(x) for x in rmse_persistence]}


def _interval(samples: np.ndarray, level: float) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0 * 100.0
    low, high = np.percentile(samples, [tail, 100.0 - tail], axis=0)
    return low, high


def cluster_bootstrap_relative_improvement(table: pd.DataFrame, resamples: int = 10_000, seed: int = 0, level: float = 0.95) -> dict:
    """Episode-cluster bootstrap of the pooled relative improvement: resample the episodes with replacement, recompute from the resampled summed errors."""
    n, model, persistence = _arrays(table)
    episodes = len(n)
    index = np.random.default_rng(seed).integers(0, episodes, size=(resamples, episodes))
    overall, by_horizon = _relative_improvement(n[index], model[index], persistence[index])
    estimate = pooled_relative_improvement(table)
    low, high = _interval(overall, level)
    h_low, h_high = _interval(by_horizon, level)
    return {"estimate": estimate["relative_improvement"], "ci_low": float(low), "ci_high": float(high), "level": level, "resamples": resamples, "seed": seed, "episodes": episodes,
            "by_horizon": [{"step": h + 1, "estimate": estimate["improvement_by_horizon"][h], "ci_low": float(h_low[h]), "ci_high": float(h_high[h])} for h in range(model.shape[1])]}


def random_subset_gate_rate(table: pd.DataFrame, subset_size: int = 10, draws: int = 10_000, seed: int = 0, margin: float = 0.02) -> dict:
    """How often a random ``subset_size``-episode split would pass the gate: draw subsets without replacement, pool each, and report the fraction with relative improvement >= ``margin``."""
    n, model, persistence = _arrays(table)
    episodes = len(n)
    if not 0 < subset_size <= episodes:
        raise ValueError(f"subset_size must be between 1 and {episodes}, got {subset_size}")
    index = np.argsort(np.random.default_rng(seed).random((draws, episodes)), axis=1)[:, :subset_size]
    overall, _ = _relative_improvement(n[index], model[index], persistence[index])
    q05, q50, q95 = np.percentile(overall, [5, 50, 95])
    return {"subset_size": subset_size, "draws": draws, "seed": seed, "margin": margin, "pass_rate": float((overall >= margin).mean()),
            "quantiles": {"q05": float(q05), "q50": float(q50), "q95": float(q95)}}


def classify_against_margin(ci_low: float, ci_high: float, margin: float = 0.02) -> str:
    """The frozen decision rule: ``BEATS`` if the interval's lower bound is above the margin, ``WORSE_THAN_PERSISTENCE`` if its upper bound is below 0, ``FAILS`` if the
    upper bound is below the margin, else ``INCONCLUSIVE``. All comparisons are strict."""
    if ci_low > margin:
        return "BEATS"
    if ci_high < 0:
        return "WORSE_THAN_PERSISTENCE"
    if ci_high < margin:
        return "FAILS"
    return "INCONCLUSIVE"


def paired_bootstrap_difference(table_a: pd.DataFrame, table_b: pd.DataFrame, resamples: int = 10_000, seed: int = 0, level: float = 0.95) -> dict:
    """Paired episode-cluster bootstrap of ``R(a) - R(b)`` over the same episodes (the same resampled episodes are used for both tables)."""
    a, b = table_a.sort_values("episode_id").reset_index(drop=True), table_b.sort_values("episode_id").reset_index(drop=True)
    if list(a["episode_id"]) != list(b["episode_id"]):
        raise ValueError("the two tables must cover the same episodes")
    (n_a, m_a, p_a), (n_b, m_b, p_b) = _arrays(a), _arrays(b)
    index = np.random.default_rng(seed).integers(0, len(a), size=(resamples, len(a)))
    diff = _relative_improvement(n_a[index], m_a[index], p_a[index])[0] - _relative_improvement(n_b[index], m_b[index], p_b[index])[0]
    low, high = _interval(diff, level)
    estimate = pooled_relative_improvement(a)["relative_improvement"] - pooled_relative_improvement(b)["relative_improvement"]
    label = "A_BETTER" if low > 0 else "B_BETTER" if high < 0 else "NOT_DISTINGUISHED"
    return {"estimate": float(estimate), "ci_low": float(low), "ci_high": float(high), "level": level, "resamples": resamples, "seed": seed, "episodes": len(a), "label": label}
