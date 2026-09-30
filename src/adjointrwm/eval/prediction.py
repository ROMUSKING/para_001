"""Method-blind prediction metrics, classical baselines and paired model comparisons.

Every metric here is a function of predictions and targets only. The arm name travels as
a label column and never enters a computation (comprehensive plan, "Prohibited shortcuts":
identical predictions must receive identical endpoint values whatever the method label).

Conventions:

* States are in the normalised space of :func:`adjointrwm.data.fit_normaliser` unless a
  name says ``native``. Native errors are ``(pred - target) * state_std`` (the mean cancels).
* ``mse[n, h]`` is the squared error of window ``n`` at horizon step ``h``, averaged over
  state dimensions. RMSE per horizon step is ``sqrt(mean_n mse[n, h])``; the primary
  endpoint is the mean of that over horizon steps (the pilot's dynamics-gate metric).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

OUTCOMES = ("reference_better", "rival_better", "equivalent_within_margin", "inconclusive")


# ---------------------------------------------------------------------------
# Per-window errors
# ---------------------------------------------------------------------------

def per_window_mse(pred: np.ndarray, target: np.ndarray, columns: Sequence[int] | None = None) -> np.ndarray:
    pred, target = np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError(f"expected matching [N, H, S] arrays, got {pred.shape} and {target.shape}")
    err = (pred - target) ** 2
    if columns is not None:
        err = err[..., list(columns)]
    return err.mean(axis=-1)


def per_window_native_mse(
    pred_norm: np.ndarray, target_norm: np.ndarray, state_std: np.ndarray, columns: Sequence[int] | None = None
) -> np.ndarray:
    std = np.asarray(state_std, dtype=np.float64)
    return per_window_mse(np.asarray(pred_norm) * std, np.asarray(target_norm) * std, columns)


def per_window_cosine(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred, target = np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64)
    num = (pred * target).sum(-1)
    den = np.linalg.norm(pred, axis=-1) * np.linalg.norm(target, axis=-1)
    return num / np.maximum(den, 1e-12)


def interval_coverage(mean: np.ndarray, logvar: np.ndarray, target: np.ndarray, z: float) -> np.ndarray:
    """Per-window fraction of state dims whose target lies within ``mean +- z * sigma``."""
    sigma = np.exp(0.5 * np.asarray(logvar, dtype=np.float64))
    inside = np.abs(np.asarray(target, dtype=np.float64) - np.asarray(mean, dtype=np.float64)) <= z * sigma
    return inside.mean(axis=-1)


def rmse_by_horizon(mse: np.ndarray) -> np.ndarray:
    return np.sqrt(np.asarray(mse, dtype=np.float64).mean(axis=0))


def horizon_mean_rmse(mse: np.ndarray) -> float:
    return float(rmse_by_horizon(mse).mean())


def effective_rank(latents: np.ndarray) -> float:
    """exp(entropy) of the normalised singular values of centred latents (collapse tripwire)."""
    x = np.asarray(latents, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    if s.sum() <= 0:
        return 0.0
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


# ---------------------------------------------------------------------------
# Long-format prediction tables
# ---------------------------------------------------------------------------

def prediction_frame(
    *,
    arm: str,
    seed: int,
    episode_ids: Sequence[str],
    window_starts: Sequence[int],
    pred_state: np.ndarray,
    target_state: np.ndarray,
    state_std: np.ndarray,
    groups: Mapping[str, Sequence[int]] | None = None,
    pred_visual: np.ndarray | None = None,
    target_visual: np.ndarray | None = None,
    pred_logvar: np.ndarray | None = None,
    split: str = "test",
    condition: str = "nominal",
) -> pd.DataFrame:
    """One row per (window, horizon step) with every per-window metric."""
    n, h = np.asarray(pred_state).shape[:2]
    columns = {
        "arm": np.repeat(arm, n * h),
        "seed": np.repeat(int(seed), n * h),
        "split": np.repeat(split, n * h),
        "condition": np.repeat(condition, n * h),
        "episode_id": np.repeat(np.asarray(episode_ids), h),
        "window_start": np.repeat(np.asarray(window_starts, dtype=int), h),
        "h": np.tile(np.arange(1, h + 1), n),
        "mse_norm": per_window_mse(pred_state, target_state).ravel(),
        "mse_native": per_window_native_mse(pred_state, target_state, state_std).ravel(),
    }
    for name, cols in (groups or {}).items():
        columns[f"mse_native_{name}"] = per_window_native_mse(pred_state, target_state, state_std, cols).ravel()
    if pred_visual is not None and target_visual is not None:
        columns["visual_cos"] = per_window_cosine(pred_visual, target_visual).ravel()
    if pred_logvar is not None:
        columns["cover_1sigma"] = interval_coverage(pred_state, pred_logvar, target_state, 1.0).ravel()
        columns["cover_2sigma"] = interval_coverage(pred_state, pred_logvar, target_state, 2.0).ravel()
    return pd.DataFrame(columns)


def summarize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Per (arm, seed, split, condition): primary endpoint, per-h RMSE and secondary metrics."""
    rows = []
    keys = ["arm", "seed", "split", "condition"]
    for key, group in frame.groupby(keys, sort=True):
        by_h = group.groupby("h")
        row = dict(zip(keys, key))
        rmse_h = np.sqrt(by_h["mse_norm"].mean().to_numpy())
        row["rmse_norm"] = float(rmse_h.mean())
        for i, value in enumerate(rmse_h, start=1):
            row[f"rmse_norm_h{i}"] = float(value)
        for column in group.columns:
            if column.startswith("mse_native"):
                row[column.replace("mse_", "rmse_")] = float(np.sqrt(by_h[column].mean().to_numpy()).mean())
        for column in ("visual_cos", "cover_1sigma", "cover_2sigma"):
            if column in group:
                row[column] = float(group[column].mean())
        row["num_windows"] = int(group[["episode_id", "window_start"]].drop_duplicates().shape[0])
        row["num_episodes"] = int(group["episode_id"].nunique())
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Paired comparison with a two-level (seed x episode) bootstrap
# ---------------------------------------------------------------------------

def _cells(frame: pd.DataFrame, arm: str, metric: str, seeds: Sequence[int], episodes: Sequence[str], horizons):
    sub = frame[frame["arm"] == arm]
    sums = sub.pivot_table(index=["seed", "episode_id"], columns="h", values=metric, aggfunc="sum")
    counts = sub.pivot_table(index=["seed", "episode_id"], columns="h", values=metric, aggfunc="count")
    idx = pd.MultiIndex.from_product([seeds, episodes], names=["seed", "episode_id"])
    sums = sums.reindex(index=idx, columns=horizons).to_numpy().reshape(len(seeds), len(episodes), len(horizons))
    counts = counts.reindex(index=idx, columns=horizons).to_numpy().reshape(len(seeds), len(episodes), len(horizons))
    return np.nan_to_num(sums), np.nan_to_num(counts)


def _rmse_from_weights(sums, counts, w_seed, w_episode):
    num = np.einsum("bs,be,seh->bh", w_seed, w_episode, sums)
    den = np.einsum("bs,be,seh->bh", w_seed, w_episode, counts)
    return np.sqrt(num / np.maximum(den, 1e-300)).mean(axis=1)


def paired_relative_difference(
    frame: pd.DataFrame,
    reference: str,
    rival: str,
    metric: str = "mse_norm",
    num_resamples: int = 5000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """``(RMSE_reference - RMSE_rival) / RMSE_rival`` with a CI; negative = reference better.

    Both arms must have been scored on exactly the same (episode, window, h) cells, which is
    part of the fairness contract. Seeds are paired by value; seeds present for only one arm
    are reported and excluded. The bootstrap resamples seeds and test episodes with
    replacement (episodes are the independent units; windows within one are correlated).
    """
    cols = ["episode_id", "window_start", "h"]
    ref_cells = frame.loc[frame["arm"] == reference, cols].drop_duplicates()
    riv_cells = frame.loc[frame["arm"] == rival, cols].drop_duplicates()
    if ref_cells.empty or riv_cells.empty:
        raise ValueError(f"no rows for {reference!r} or {rival!r}")
    merged = ref_cells.merge(riv_cells, how="outer", indicator=True)
    if (merged["_merge"] != "both").any():
        raise ValueError(f"{reference!r} and {rival!r} were not scored on the same windows")
    ref_seeds = set(frame.loc[frame["arm"] == reference, "seed"])
    riv_seeds = set(frame.loc[frame["arm"] == rival, "seed"])
    seeds = sorted(ref_seeds & riv_seeds)
    if not seeds:
        raise ValueError("no paired seeds")
    episodes = sorted(ref_cells["episode_id"].unique())
    horizons = sorted(ref_cells["h"].unique())

    ref = _cells(frame, reference, metric, seeds, episodes, horizons)
    riv = _cells(frame, rival, metric, seeds, episodes, horizons)
    ones_s, ones_e = np.ones((1, len(seeds))), np.ones((1, len(episodes)))
    r_ref = _rmse_from_weights(*ref, ones_s, ones_e)[0]
    r_riv = _rmse_from_weights(*riv, ones_s, ones_e)[0]

    rng = np.random.default_rng(seed)
    w_s = rng.multinomial(len(seeds), np.full(len(seeds), 1 / len(seeds)), size=num_resamples)
    w_e = rng.multinomial(len(episodes), np.full(len(episodes), 1 / len(episodes)), size=num_resamples)
    b_ref = _rmse_from_weights(*ref, w_s, w_e)
    b_riv = _rmse_from_weights(*riv, w_s, w_e)
    boot = (b_ref - b_riv) / np.maximum(b_riv, 1e-300)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(boot, [alpha, 1 - alpha])

    per_seed = {}
    for i, s in enumerate(seeds):
        w = np.zeros((1, len(seeds)))
        w[0, i] = 1
        a = _rmse_from_weights(*ref, w, ones_e)[0]
        b = _rmse_from_weights(*riv, w, ones_e)[0]
        per_seed[int(s)] = float((a - b) / b)
    return {
        "reference": reference,
        "rival": rival,
        "metric": metric,
        "reference_rmse": float(r_ref),
        "rival_rmse": float(r_riv),
        "relative_difference": float((r_ref - r_riv) / r_riv),
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": confidence,
        "per_seed": per_seed,
        "seeds": [int(s) for s in seeds],
        "unpaired_seeds": sorted(int(s) for s in (ref_seeds ^ riv_seeds)),
        "num_episodes": len(episodes),
        "num_resamples": num_resamples,
    }


def classify_relative_difference(ci_low: float, ci_high: float, margin: float) -> str:
    """Frozen decision rule of the rival-benchmark plan §4 (negative = reference better)."""
    if ci_high < 0:
        return "reference_better"
    if ci_low > 0:
        return "rival_better"
    if -margin < ci_low and ci_high < margin:
        return "equivalent_within_margin"
    return "inconclusive"


# ---------------------------------------------------------------------------
# Classical baselines (no learning, or closed form)
# ---------------------------------------------------------------------------

def persistence_forecast(context_state: np.ndarray, horizon: int) -> np.ndarray:
    last = np.asarray(context_state)[:, -1:, :]
    return np.repeat(last, horizon, axis=1)


class RidgeForecaster:
    """Linear forecaster of future state deltas from recent states and all given actions.

    Features: the last ``history`` context states, all context actions and all future
    actions (flattened), standardised with train statistics. Target: ``target_state`` minus
    the last context state. ``lambda`` is chosen by validation MSE from ``lambdas``.
    """

    def __init__(self, history: int = 2, lambdas: Sequence[float] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)):
        self.history = int(history)
        self.lambdas = tuple(float(x) for x in lambdas)
        self.coef_ = None

    def features(self, context_state, context_action, future_actions) -> np.ndarray:
        n = len(context_state)
        parts = [
            np.asarray(context_state)[:, -self.history :].reshape(n, -1),
            np.asarray(context_action).reshape(n, -1),
            np.asarray(future_actions).reshape(n, -1),
        ]
        return np.concatenate(parts, axis=1).astype(np.float64)

    def _design(self, x):
        z = (x - self.x_mean_) / self.x_std_
        return np.concatenate([z, np.ones((len(z), 1))], axis=1)

    def _solve(self, design, y, lam):
        penalty = lam * np.eye(design.shape[1])
        penalty[-1, -1] = 0.0  # no penalty on the bias
        return np.linalg.solve(design.T @ design + penalty, design.T @ y)

    def fit(self, train: Mapping, validation: Mapping) -> "RidgeForecaster":
        x = self.features(train["context_state"], train["context_action"], train["future_actions"])
        self.x_mean_, self.x_std_ = x.mean(0), np.maximum(x.std(0), 1e-8)
        y = self._delta(train)
        design = self._design(x)
        xv = self._design(self.features(validation["context_state"], validation["context_action"], validation["future_actions"]))
        yv = self._delta(validation)
        scores = {}
        for lam in self.lambdas:
            coef = self._solve(design, y, lam)
            scores[lam] = float(((xv @ coef - yv) ** 2).mean())
        self.lambda_ = min(scores, key=scores.get)
        self.validation_mse_ = scores
        self.coef_ = self._solve(design, y, self.lambda_)
        self.horizon_, self.state_dim_ = np.asarray(train["target_state"]).shape[1:]
        return self

    @staticmethod
    def _delta(split: Mapping) -> np.ndarray:
        target = np.asarray(split["target_state"], dtype=np.float64)
        last = np.asarray(split["context_state"], dtype=np.float64)[:, -1:, :]
        return (target - last).reshape(len(target), -1)

    def predict(self, context_state, context_action, future_actions) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("fit first")
        design = self._design(self.features(context_state, context_action, future_actions))
        delta = (design @ self.coef_).reshape(len(design), self.horizon_, self.state_dim_)
        return np.asarray(context_state, dtype=np.float64)[:, -1:, :] + delta


def derangement(n: int, seed: int) -> np.ndarray:
    """A permutation with no fixed points (every window gets another window's actions)."""
    if n < 2:
        raise ValueError("need at least two windows to shuffle actions")
    order = np.random.default_rng(seed).permutation(n)
    perm = np.empty(n, dtype=int)
    perm[order] = np.roll(order, -1)  # one random n-cycle: no index maps to itself
    return perm


def stack_windows(dataset, indices: Sequence[int] | None = None, keys: Sequence[str] | None = None) -> dict:
    """Materialise windows of a :class:`adjointrwm.data.WindowDataset` into stacked arrays."""
    indices = range(len(dataset)) if indices is None else indices
    items = [dataset[i] for i in indices]
    keys = keys or [k for k, v in items[0].items() if isinstance(v, np.ndarray)]
    out = {k: np.stack([it[k] for it in items]) for k in keys}
    out["episode_id"] = np.array([it["episode_id"] for it in items])
    out["window_start"] = np.array([it["window_start"] for it in items])
    return out
