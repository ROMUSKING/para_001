"""Domain-neutral allocation interface, runner and metrics, plus reference domains (NumPy only).

See docs/plans/cross-domain-plan.md. ``linear_ode`` is D4, the exact-adjoint reference domain.
"""

from .base import ACTION_KINDS, AllocationDomain, Candidate, Cost, CostWeights, DomainSpec
from .linear_ode import AdaptiveTimeSteppingDomain, LinearODEInstance, Pulse, d4_policies, sample_instances
from .metrics import (
    BEST_KNOWN,
    absolute_adaptive_gain,
    aurc,
    aurc_table,
    fraction_of_oracle_advantage,
    opportunity_over_budgets,
    paired_aurc_difference,
    policy_curves,
    regret_curve,
    transfer_summary,
    with_best_known,
)
from .runner import Policy, PolicyContext, Trace, evaluate_policies, exhaustive_oracle_curve, one_step_oracle, random_policy, run_policy

__all__ = [
    "ACTION_KINDS",
    "BEST_KNOWN",
    "AdaptiveTimeSteppingDomain",
    "AllocationDomain",
    "Candidate",
    "Cost",
    "CostWeights",
    "DomainSpec",
    "LinearODEInstance",
    "Policy",
    "PolicyContext",
    "Pulse",
    "Trace",
    "absolute_adaptive_gain",
    "aurc",
    "aurc_table",
    "d4_policies",
    "evaluate_policies",
    "exhaustive_oracle_curve",
    "fraction_of_oracle_advantage",
    "one_step_oracle",
    "opportunity_over_budgets",
    "paired_aurc_difference",
    "policy_curves",
    "random_policy",
    "regret_curve",
    "run_policy",
    "sample_instances",
    "transfer_summary",
    "with_best_known",
]
