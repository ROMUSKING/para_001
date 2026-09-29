"""Offline diagnostics for adjoint / critic allocation runs.

Everything here is plain NumPy/pandas so it runs anywhere (laptop, CI, Colab)
without a GPU or the training stack. Two entry points matter:

* :func:`summarize_traces` works on the ``allocation_traces.parquet`` file the
  production pilot already writes (per-window choices and regrets).
* :func:`opportunity_audit` works on a full per-window gain matrix
  ``exact_gain[N, K]``. The pilot does not log that matrix yet; the
  diagnostics notebook in ``notebooks/02-diagnostics`` produces it.

Sign convention (matches the pilot notebook): ``exact_gain[i, k]`` is the
reduction in the declared objective from applying candidate ``k`` to window
``i``, *minus* that candidate's cost. Larger is better; the oracle picks the
arg-max; regret of a choice is ``max_k gain - gain[choice]``.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

CHOICE_COLUMNS = ("oracle_choice", "adjoint_choice", "critic_choice", "hybrid_choice")
REGRET_COLUMNS = ("adjoint_regret", "critic_regret", "hybrid_regret")


# ---------------------------------------------------------------------------
# Trace-level summaries (work on the parquet the pilot already writes)
# ---------------------------------------------------------------------------

def load_traces(path: str) -> pd.DataFrame:
    """Load an ``allocation_traces.parquet`` file and check its schema."""
    frame = pd.read_parquet(path)
    missing = {"episode_id", *CHOICE_COLUMNS[:3], *REGRET_COLUMNS[:2]} - set(frame.columns)
    if missing:
        raise ValueError(f"allocation traces missing columns: {sorted(missing)}")
    return frame


def choice_distribution(frame: pd.DataFrame, num_candidates: int) -> pd.DataFrame:
    """Share of windows on which each policy picks each candidate.

    Rows are policies (oracle/adjoint/critic/hybrid), columns are candidate
    indices. A learned allocator that never picks a candidate the oracle often
    prefers has collapsed, whatever its average regret looks like.
    """
    rows = {}
    for column in CHOICE_COLUMNS:
        if column not in frame:
            continue
        counts = np.bincount(frame[column].to_numpy(dtype=int), minlength=num_candidates)
        rows[column.replace("_choice", "")] = counts / counts.sum()
    return pd.DataFrame(rows, index=[f"c{k}" for k in range(num_candidates)]).T


def top1_accuracy(frame: pd.DataFrame, policy: str) -> float:
    return float((frame[f"{policy}_choice"] == frame["oracle_choice"]).mean())


def episode_bootstrap_ci(
    frame: pd.DataFrame,
    column: str,
    baseline: str | None = None,
    num_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """Cluster (episode-level) bootstrap CI for a mean, or a paired difference.

    Windows from one episode are strongly correlated, so resampling windows
    understates uncertainty. This resamples whole episodes with replacement.
    With ``baseline`` set it bootstraps ``mean(column - baseline)``.
    """
    values = frame[column].to_numpy(dtype=float)
    if baseline is not None:
        values = values - frame[baseline].to_numpy(dtype=float)
    episodes = frame["episode_id"].to_numpy()
    unique, inverse = np.unique(episodes, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique)).astype(float)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(num_resamples, len(unique)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(boot, [alpha, 1.0 - alpha])
    return {
        "statistic": f"mean({column}{' - ' + baseline if baseline else ''})",
        "estimate": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": confidence,
        "num_episodes": int(len(unique)),
        "num_windows": int(len(values)),
    }


def per_episode_means(frame: pd.DataFrame, columns: Sequence[str] = REGRET_COLUMNS[:2]) -> pd.DataFrame:
    present = [c for c in columns if c in frame]
    return frame.groupby("episode_id")[present].mean().sort_values(present[0])


def summarize_traces(frame: pd.DataFrame, num_candidates: int, seed: int = 0) -> dict:
    """One JSON-serialisable summary of an allocation trace file."""
    distribution = choice_distribution(frame, num_candidates)
    oracle_share = distribution.loc["oracle"]
    summary = {
        "num_windows": int(len(frame)),
        "num_episodes": int(frame["episode_id"].nunique()),
        "choice_distribution": distribution.round(4).to_dict(orient="index"),
        "top1": {p: top1_accuracy(frame, p) for p in ("adjoint", "critic") if f"{p}_choice" in frame},
        "chance_top1": 1.0 / num_candidates,
        # Best achievable top-1 by a constant policy = the oracle's modal share.
        "best_constant_top1": float(oracle_share.max()),
        "best_constant_candidate": int(np.argmax(oracle_share.to_numpy())),
        "adjoint_critic_agreement": float((frame["adjoint_choice"] == frame["critic_choice"]).mean()),
        "unused_candidates": {
            p: [k for k in range(num_candidates) if distribution.loc[p, f"c{k}"] == 0.0]
            for p in ("adjoint", "critic") if p in distribution.index
        },
        "regret_ci": {
            c: episode_bootstrap_ci(frame, c, seed=seed) for c in REGRET_COLUMNS[:2]
        },
        "adjoint_minus_critic_ci": episode_bootstrap_ci(
            frame, "adjoint_regret", baseline="critic_regret", seed=seed
        ),
        "regret_quantiles": {
            c: {q: float(frame[c].quantile(q)) for q in (0.5, 0.9, 0.99)} for c in REGRET_COLUMNS[:2]
        },
    }
    if "gate_probability" in frame:
        summary["gate"] = {
            "invocation_rate": float((frame["gate_probability"] >= 0.5).mean()),
            "max_probability": float(frame["gate_probability"].max()),
        }
    return summary


# ---------------------------------------------------------------------------
# Opportunity audit (needs the full per-window gain matrix)
# ---------------------------------------------------------------------------

@dataclass
class OpportunityReport:
    num_windows: int
    num_candidates: int
    oracle_mean_gain: float
    best_fixed_candidate: int
    best_fixed_mean_gain: float
    random_mean_gain: float
    headroom_over_best_fixed: float      # oracle - best fixed: what adaptivity can buy
    headroom_over_random: float          # oracle - random (expected, not one draw)
    fraction_oracle_differs_from_best_fixed: float
    median_gain_spread: float            # per-window max - min across candidates
    cost_gap: float                      # max - min candidate cost (same units as gain)
    spread_exceeds_cost_gap_fraction: float
    fixed_candidate_mean_gain: list
    oracle_share: list
    passes: bool
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def opportunity_audit(
    exact_gain: np.ndarray,
    candidate_costs: Sequence[float] | None = None,
    min_relative_headroom: float = 0.15,
) -> OpportunityReport:
    """Is there anything for an adaptive allocator to win on this benchmark?

    Compares the per-window oracle with the best *constant* policy and with
    uniform random choice. If the oracle barely beats the best constant choice,
    no allocator (adjoint or critic) can show an advantage and the benchmark
    must be redesigned before the H2 comparison means anything.

    ``passes`` requires headroom over the best fixed policy to be at least
    ``min_relative_headroom`` of the headroom over random. This is a per-decision
    proxy for the research plan's opportunity gate §7.2B-1 (oracle beats the best
    fixed allocation by >= 15 % of normalised rate-distortion area); the
    confirmatory gate must still be computed on the RD area with a CI.
    """
    gain = np.asarray(exact_gain, dtype=float)
    if gain.ndim != 2 or gain.shape[1] < 2:
        raise ValueError("exact_gain must have shape [num_windows, num_candidates>=2]")
    n, k = gain.shape
    oracle = gain.max(axis=1)
    fixed = gain.mean(axis=0)
    best_fixed = int(np.argmax(fixed))
    random_mean = float(gain.mean())
    spread = gain.max(axis=1) - gain.min(axis=1)
    costs = np.zeros(k) if candidate_costs is None else np.asarray(candidate_costs, dtype=float)
    cost_gap = float(costs.max() - costs.min())

    head_fixed = float(oracle.mean() - fixed[best_fixed])
    head_random = float(oracle.mean() - random_mean)
    if head_random <= 0:
        passes, reason = False, "all candidates are identical on every window"
    elif head_fixed < min_relative_headroom * head_random:
        passes, reason = False, (
            f"oracle beats best fixed candidate by {head_fixed:.3g}, "
            f"< {min_relative_headroom:.0%} of its margin over random ({head_random:.3g})"
        )
    else:
        passes, reason = True, "per-window choice has measurable headroom over the best constant policy"

    return OpportunityReport(
        num_windows=n,
        num_candidates=k,
        oracle_mean_gain=float(oracle.mean()),
        best_fixed_candidate=best_fixed,
        best_fixed_mean_gain=float(fixed[best_fixed]),
        random_mean_gain=random_mean,
        headroom_over_best_fixed=head_fixed,
        headroom_over_random=head_random,
        fraction_oracle_differs_from_best_fixed=float((gain.argmax(axis=1) != best_fixed).mean()),
        median_gain_spread=float(np.median(spread)),
        cost_gap=cost_gap,
        spread_exceeds_cost_gap_fraction=float((spread > cost_gap).mean()) if cost_gap > 0 else float("nan"),
        fixed_candidate_mean_gain=[float(x) for x in fixed],
        oracle_share=[float(x) for x in np.bincount(gain.argmax(axis=1), minlength=k) / n],
        passes=passes,
        reason=reason,
    )


def policy_regret(exact_gain: np.ndarray, choices: Iterable[int]) -> np.ndarray:
    """Per-window regret of a vector of choices against the oracle."""
    gain = np.asarray(exact_gain, dtype=float)
    idx = np.asarray(list(choices), dtype=int)
    return gain.max(axis=1) - gain[np.arange(len(idx)), idx]


def first_order_scores(costate: np.ndarray, effects: np.ndarray, costs: np.ndarray) -> np.ndarray:
    """Adjoint (first-order) candidate scores: ``-<lambda, delta_k> - cost_k``.

    ``costate`` [N, D], ``effects`` [N, K, D], ``costs`` [K] (already weighted).
    Feeding the *exact* co-state here gives the ceiling for any amortised
    co-state estimator; if this fails to beat random, the linearisation, not
    the estimator, is the problem.
    """
    return -np.einsum("nd,nkd->nk", costate, effects) - np.asarray(costs)[None, :]


def compare_distributions(train_gain: np.ndarray, test_gain: np.ndarray) -> Mapping[str, list]:
    """Oracle-choice shares and mean gains per candidate on two splits."""
    out = {}
    for name, gain in (("train", train_gain), ("test", test_gain)):
        gain = np.asarray(gain, dtype=float)
        k = gain.shape[1]
        out[f"{name}_oracle_share"] = (np.bincount(gain.argmax(axis=1), minlength=k) / len(gain)).tolist()
        out[f"{name}_mean_gain"] = gain.mean(axis=0).tolist()
    return out
