"""Domain-neutral allocation metrics, as defined in the governing plan (§2.3, §2.3.1, §7.2B).

All objectives are lower-is-better. Curves are indexed by budget ``B`` (steps or cost units).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..analysis.allocation import episode_bootstrap_ci


def _curve(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def regret_curve(j_method, j_oracle) -> np.ndarray:
    """``J_method(B) - J_oracle(B)``."""
    return _curve(j_method) - _curve(j_oracle)


def aurc(curve, budgets: Sequence[float] | None = None) -> float:
    """Trapezoidal area under a (regret) curve over the budget range."""
    y = _curve(curve)
    x = np.arange(len(y), dtype=float) if budgets is None else _curve(budgets)
    if len(y) < 2:
        return 0.0
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2.0)


def absolute_adaptive_gain(j_fixed, j_method) -> np.ndarray:
    """``J_fixed(B) - J_method(B)``; positive means the adaptive method is better."""
    return _curve(j_fixed) - _curve(j_method)


def fraction_of_oracle_advantage(j_fixed: float, j_method: float, j_oracle: float, delta_min: float) -> float:
    """``(J_fixed - J_method) / (J_fixed - J_oracle)``; NaN when the oracle advantage <= ``delta_min``."""
    denominator = j_fixed - j_oracle
    if denominator <= delta_min:
        return float("nan")
    return float((j_fixed - j_method) / denominator)


def opportunity_over_budgets(j_fixed, j_oracle, budgets=None, min_relative: float = 0.15) -> dict:
    """Plan §7.2B-1 over a budget range: ``area(J_fixed - J_oracle) >= min_relative * area(J_fixed)``."""
    head = aurc(absolute_adaptive_gain(j_fixed, j_oracle), budgets)
    base = aurc(j_fixed, budgets)
    relative = head / base if base > 0 else float("nan")
    return {"oracle_headroom_area": head, "fixed_area": base, "relative_headroom": relative,
            "min_relative": min_relative, "passes": bool(base > 0 and relative >= min_relative)}


def transfer_summary(deltas: Mapping[str, float], m_domain: float) -> dict:
    """``TransferMacro`` (equal-domain mean) and ``TransferWorst`` (min); positive = better."""
    values = np.array(list(deltas.values()), dtype=float)
    macro, worst = float(values.mean()), float(values.min())
    return {"transfer_macro": macro, "transfer_worst": worst, "m_domain": m_domain,
            "worst_domain_guard_passes": bool(worst >= -m_domain), "per_domain": dict(deltas)}


def policy_curves(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean objective per (instance, policy, budget); random draws are averaged (expected random)."""
    return frame.groupby(["instance", "policy", "budget"], as_index=False)["objective"].mean()


BEST_KNOWN = "best_known"


def with_best_known(wide: pd.DataFrame) -> pd.DataFrame:
    """Add the pointwise minimum over every evaluated policy (privileged ones included).

    A greedy one-step oracle is not a true oracle and can be beaten, which would make regret
    negative. The best-known curve upper-bounds the true oracle ``J*(B)``, so regret against it
    is non-negative and a *lower bound* on true oracle regret. It depends on which policies were
    evaluated, so reports must list them.
    """
    out = wide.copy()
    out[BEST_KNOWN] = wide.min(axis=1)
    return out


def aurc_table(frame: pd.DataFrame, reference: str = BEST_KNOWN, fixed: str | None = None) -> pd.DataFrame:
    """Per (instance, policy): AURC of regret to ``reference`` and final-budget regret.

    ``reference`` is a policy name or :data:`BEST_KNOWN` (default).
    """
    curves = policy_curves(frame)
    wide = curves.pivot_table(index=["instance", "budget"], columns="policy", values="objective").sort_index()
    wide = with_best_known(wide)
    rows = []
    for instance, block in wide.groupby(level="instance"):
        ref = block[reference].to_numpy()
        for policy in [c for c in block.columns if c != BEST_KNOWN]:
            j = block[policy].to_numpy()
            row = {"instance": instance, "policy": policy, "aurc": aurc(regret_curve(j, ref)),
                   "final_regret": float(j[-1] - ref[-1]), "final_objective": float(j[-1])}
            if fixed is not None:
                row["adaptive_gain_area"] = aurc(absolute_adaptive_gain(block[fixed].to_numpy(), j))
            rows.append(row)
    return pd.DataFrame(rows)


def paired_aurc_difference(table: pd.DataFrame, policy_a: str, policy_b: str, num_resamples: int = 10_000, seed: int = 0) -> dict:
    """Bootstrap CI over instances for ``AURC_a - AURC_b`` (negative = ``a`` has lower regret)."""
    wide = table.pivot_table(index="instance", columns="policy", values="aurc")
    frame = pd.DataFrame({"episode_id": wide.index.astype(str), "a": wide[policy_a].to_numpy(), "b": wide[policy_b].to_numpy()})
    out = episode_bootstrap_ci(frame, "a", baseline="b", num_resamples=num_resamples, seed=seed)
    out["statistic"] = f"mean(AURC_{policy_a} - AURC_{policy_b}) over instances"
    return out
