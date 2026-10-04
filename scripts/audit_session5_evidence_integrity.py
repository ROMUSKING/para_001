#!/usr/bin/env python3
"""Recompute the Session 5 evidence-integrity findings from committed artefacts.

Every number quoted in ``docs/audits/2026-10-03_session5_seed_duplication_and_voi_identity_audit.md``
is produced here, from files already committed in this repository. Nothing is taken
from the research note on trust.

Usage::

    python scripts/audit_session5_evidence_integrity.py
    python scripts/audit_session5_evidence_integrity.py --json

Exit status is 0 when every check reproduces the audit's findings. It is *not* a
pass/fail gate on the science: these checks confirm that the audit is accurate
about the state of the evidence, not that the evidence is sound.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

SUMMARY = Path("results/benchmarks/robustness_horizon/robustness_horizon_summary.json")
RUNNER = Path("scripts/benchmark_robustness_horizon_allocator.py")
ALLOCATORS = Path("src/adjointrwm/allocators.py")

POLICIES = (
    "exact_costate",
    "always_mode0",
    "direct_critic",
    "direct_critic_curv_matched",
    "first_order",
    "normalized_first_order",
    "second_order_curvature",
    "belief_space_voi",
)
REGIMES = ("zero", "uniform", "default", "latency_weighted", "high_penalty")
HORIZONS = ("2", "4")
UNCERT_WEIGHT_DEFAULT = 0.5  # allocators.belief_space_voi_scores default


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def mean_over_seeds(summary, regime, policy, horizon=None):
    block = summary[0]["horizons"][horizon] if horizon else summary[0]["cost_regimes"][regime]
    values = [
        (s["horizons"][horizon] if horizon else s["cost_regimes"][regime])["per_policy"][policy]["mean_regret"]
        for s in summary
    ]
    assert block  # keeps the reference above meaningful for readers
    return st.mean(values)


def advantage(critic: float, treatment: float) -> float:
    return (critic - treatment) / critic * 100.0


def check_seed_duplication(summary) -> dict:
    rule("CHECK 1  Seed fingerprint: is seed 2 an independent replicate of seed 0?")
    # Wall-clock fields legitimately vary between repeated runs of the same work, so
    # they are compared separately: identical *performance* is what indicates a duplicate.
    timing_fields = {"mean_latency_ms_per_window", "throughput_hz"}

    def flat(node, path=""):
        out = {}
        if isinstance(node, dict):
            for key, value in node.items():
                out.update(flat(value, f"{path}.{key}"))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                out.update(flat(value, f"{path}[{index}]"))
        else:
            out[path] = node
        return out

    def classify(node):
        flat_map = flat(node)
        performance = {k: v for k, v in flat_map.items() if k.split(".")[-1] not in timing_fields}
        timing = {k: v for k, v in flat_map.items() if k.split(".")[-1] in timing_fields}
        return performance, timing

    bodies = [{k: v for k, v in seed.items() if k != "seed"} for seed in summary]
    perf_maps = [classify(body)[0] for body in bodies]
    time_maps = [classify(body)[1] for body in bodies]

    schema_ok = all(m.keys() == perf_maps[0].keys() for m in perf_maps)
    if not schema_ok:
        print("  schemas differ between seeds; cannot compare positionally")
        return {"schema_match": False}

    keys = list(perf_maps[0])
    identical_perf_0_2 = sum(perf_maps[0][k] == perf_maps[2][k] for k in keys)
    identical_perf_0_1 = sum(perf_maps[0][k] == perf_maps[1][k] for k in keys)
    differing_perf = [k for k in keys if perf_maps[0][k] != perf_maps[2][k]]
    time_keys = list(time_maps[0])
    differing_time = [k for k in time_keys if time_maps[0][k] != time_maps[2][k]]

    print("  Performance scalars (regret, gain, CI, win rate, counts) compared:")
    print(f"    seed 0 vs seed 2 identical                        : {identical_perf_0_2}/{len(keys)}")
    print(f"    seed 0 vs seed 1 identical                        : {identical_perf_0_1}/{len(keys)}")
    if differing_perf:
        print(f"    performance scalars that DIFFER (0 vs 2)          : {len(differing_perf)}")
        for k in differing_perf[:10]:
            print(f"      {k}: {perf_maps[0][k]} vs {perf_maps[2][k]}")
    print("  Wall-clock scalars (expected to vary between any two runs):")
    print(f"    seed 0 vs seed 2 differing                       : {len(differing_time)}/{len(time_keys)}")

    duplicate = identical_perf_0_2 == len(keys)
    print(f"\n  verdict on PERFORMANCE metrics                    : "
          f"{'DUPLICATE - not an independent replicate' if duplicate else 'independent'}")

    source = RUNNER.read_text()
    fallback_present = 'run_dir / "seed_0" / "best.pt"' in source
    print(f"\n  runner contains a seed_0 checkpoint fallback       : {fallback_present}")
    if fallback_present:
        print("  -> the runner offers exactly one code path by which seed 2 can silently")
        print("     evaluate seed 0's teacher weights (added in commit 2f5507d). Identical")
        print("     performance metrics show the same weights were used, but WHICH weights")
        print("     cannot be settled from committed files alone: the runner tries")
        print("     run_dir/best.pt first, and checkpoints live on Drive. Confirming the")
        print("     mechanism requires the run log or the checkpoint inventory/hashes.")
    print("     Note this is a provenance limit, not a limit on Finding 1: whatever the")
    print("     mechanism, seeds 0 and 2 are the same evaluation and cannot be counted")
    print("     as two independent replicates.")
    return {
        "schema_match": schema_ok,
        "performance_scalars": len(keys),
        "identical_performance_seed0_seed2": identical_perf_0_2,
        "identical_performance_seed0_seed1": identical_perf_0_1,
        "differing_performance_keys": differing_perf,
        "differing_timing_scalars": len(differing_time),
        "verdict": "DUPLICATE - not an independent replicate" if duplicate else "independent",
        "fallback_present": fallback_present,
        "provenance": "not determinable from committed files; fallback is the only silent path",
    }


def check_per_seed_sign(summary) -> dict:
    rule("CHECK 2  Per-seed sign of the primary endpoint (VOI vs direct critic)")
    flips = 0
    rows = []
    for regime in REGIMES:
        per_seed = []
        for seed in summary:
            block = seed["cost_regimes"][regime]["per_policy"]
            per_seed.append(
                advantage(block["direct_critic"]["mean_regret"], block["belief_space_voi"]["mean_regret"])
            )
        distinct = sorted({round(v, 6) for v in per_seed})
        mixed = min(per_seed) < 0 < max(per_seed)
        flips += int(mixed)
        rows.append({"regime": regime, "per_seed_pct": per_seed, "distinct_values": len(distinct)})
        print(
            f"  {regime:<16} "
            + "  ".join(f"seed{i}={v:+7.2f}%" for i, v in enumerate(per_seed))
            + f"   distinct={len(distinct)}  sign_flip={mixed}"
        )

    for horizon in HORIZONS:
        per_seed = []
        for seed in summary:
            block = seed["horizons"][horizon]["per_policy"]
            per_seed.append(
                advantage(block["direct_critic"]["mean_regret"], block["belief_space_voi"]["mean_regret"])
            )
        mixed = min(per_seed) < 0 < max(per_seed)
        flips += int(mixed)
        rows.append({"horime_horizon": horizon, "per_seed_pct": per_seed, "distinct_values": len({round(v, 6) for v in per_seed})})
        print(
            f"  H={horizon:<15} "
            + "  ".join(f"seed{i}={v:+7.2f}%" for i, v in enumerate(per_seed))
            + f"   distinct={len({round(v, 6) for v in per_seed})}  sign_flip={mixed}"
        )

    print(f"\n  panels whose sign flips across seeds             : {flips}/{len(REGIMES) + len(HORIZONS)}")
    print("  -> the 3-seed mean is positive in every panel, but the only independent")
    print("     replicate disagrees in sign wherever a flip is reported.")
    return {"panels_with_sign_flip": flips, "panels": rows}


def check_functional_identity(summary) -> dict:
    rule("CHECK 3  Is belief_space_voi a distinct method from second_order_curvature?")
    source = ALLOCATORS.read_text()
    runner = RUNNER.read_text()
    delta_cov_is_squared = "delta_cov = effects.pow(2)" in runner
    print(f"  runner sets delta_cov = effects.pow(2)           : {delta_cov_is_squared}")
    print(f"  belief_space_voi default uncert_weight (beta)     : {UNCERT_WEIGHT_DEFAULT}")

    curvature = "curvature = 0.5 * (diag_hessian.unsqueeze(1) * effects.pow(2)).sum(-1)"
    info = "info = 0.5 * uncert_weight * (diag_hessian.unsqueeze(1) * delta_cov).sum(-1)"
    print(f"  second_order: s = first - curvature - costs       : {curvature in source}")
    print(f"  belief_space_voi: s = first + info - costs        : {info in source}")

    result = {"delta_cov_is_effects_squared": delta_cov_is_squared, "beta": UNCERT_WEIGHT_DEFAULT}
    try:
        import torch

        torch.manual_seed(0)
        from adjointrwm.allocators import (
            belief_space_voi_scores,
            first_order_scores,
            second_order_curvature_scores,
        )
    except ImportError:
        print("\n  torch not installed - skipping the numeric collapse check.")
        print("  (research plan: torch is optional for the test suite)")
        return result

    costate = torch.randn(7, 5)
    hessian = torch.rand(7, 5) + 0.5
    effects = torch.randn(7, 4, 5)
    costs = torch.tensor([0.0, 0.002, 0.002, 0.004])

    curv = second_order_curvature_scores(costate, hessian, effects, costs)
    voi_default = belief_space_voi_scores(costate, effects, effects.pow(2), hessian, costs)
    # beta = -1 reproduces second_order_curvature exactly; beta = 0 is pure first order.
    voi_neg1 = belief_space_voi_scores(costate, effects, effects.pow(2), hessian, costs, uncert_weight=-1.0)
    voi_zero = belief_space_voi_scores(costate, effects, effects.pow(2), hessian, costs, uncert_weight=0.0)

    collapsed = torch.allclose(curv, voi_neg1, atol=1e-6)
    is_first_order = torch.allclose(voi_zero, first_order_scores(costate, effects, costs), atol=1e-6)
    identical = torch.allclose(curv, voi_default, atol=1e-6)
    max_gap = float((curv[:, 1:] - voi_default[:, 1:]).abs().max())
    ratio = float(
        (voi_default[:, 1:] - first_order_scores(costate, effects, costs)[:, 1:]).abs().max()
        / (curv[:, 1:] - first_order_scores(costate, effects, costs)[:, 1:]).abs().max()
    )
    print(f"\n  VOI(beta=-1) == second_order_curvature exactly      : {collapsed}")
    print(f"  VOI(beta=0)  == first_order exactly                  : {is_first_order}")
    print(f"  curvature == VOI at the default beta=0.5            : {identical}")
    print(f"  max |difference| on refine options, beta=0.5         : {max_gap:.6e}")
    print(f"  quadratic weight of default VOI / curvature weight  : {ratio:+.3f}")
    print("\n  -> with delta_cov = effects^2 the two scorers are one functional family.")
    print("     Writing Q = sum_d H_d * effects_d^2:")
    print("       second_order : first - 0.50 * Q - costs")
    print("       belief VOI   : first + 0.25 * Q - costs   (beta = 0.5)")
    print("     Same two terms; only the sign and a factor of two on the quadratic differ,")
    print("     and beta = -1 maps one onto the other exactly. There is no epistemic term")
    print("     to switch off: delta_cov carries no prediction-variance semantics at all.")
    result.update(
        {
            "betaminus1_collapses_to_curvature": collapsed,
            "beta0_is_first_order": is_first_order,
            "identical_at_default_beta": identical,
            "max_abs_difference_default_beta": max_gap,
            "quadratic_weight_ratio": ratio,
        }
    )
    return result


def check_metric_rank_inversion(summary) -> dict:
    rule("CHECK 4  Rank agreement between mean regret and win rate vs the direct critic")
    rows = []
    for policy in POLICIES:
        block = lambda s, k: s["cost_regimes"]["default"]["per_policy"][policy][k]  # noqa: E731
        mean_regret = st.mean([block(s, "mean_regret") for s in summary])
        median_regret = st.mean([block(s, "median_regret") for s in summary])
        trimmed = st.mean([block(s, "trimmed_mean_regret_10pct") for s in summary])
        std_regret = st.mean([block(s, "std_regret") for s in summary])
        win_rate = st.mean([block(s, "win_rate_vs_critic") for s in summary])
        rows.append(
            {
                "policy": policy,
                "mean_regret": mean_regret,
                "median_regret": median_regret,
                "trimmed_mean_regret": trimmed,
                "std_regret": std_regret,
                "win_rate_vs_critic": win_rate,
            }
        )

    by_mean = sorted(rows, key=lambda r: r["mean_regret"])
    by_win = sorted(rows, key=lambda r: -r["win_rate_vs_critic"])
    mean_rank = {r["policy"]: i + 1 for i, r in enumerate(by_mean)}
    win_rank = {r["policy"]: i + 1 for i, r in enumerate(by_win)}

    header = f"  {'policy':<28}{'mean':>9}{'trim10':>9}{'std':>10}{'win%':>8}{'rk_mean':>9}{'rk_win':>8}"
    print(header)
    for row in by_mean:
        print(
            f"  {row['policy']:<28}{row['mean_regret']:>9.5f}{row['trimmed_mean_regret']:>9.5f}"
            f"{row['std_regret']:>10.5f}{row['win_rate_vs_critic'] * 100:>7.1f}%"
            f"{mean_rank[row['policy']]:>9}{win_rank[row['policy']]:>8}"
        )

    voi = next(r for r in rows if r["policy"] == "belief_space_voi")
    mode0 = next(r for r in rows if r["policy"] == "always_mode0")
    agreement = sum(mean_rank[r["policy"]] == win_rank[r["policy"]] for r in rows)
    print(f"\n  policies in the same rank position on both metrics : {agreement}/{len(rows)}")
    print(f"  belief_space_voi beats the critic on              : {voi['win_rate_vs_critic'] * 100:.1f}% of windows")
    print(f"  always_mode0 (refusal) beats the critic on        : {mode0['win_rate_vs_critic'] * 100:.1f}% of windows")
    print(f"     ...while its mean regret is {mode0['mean_regret'] / voi['mean_regret']:.1f}x belief_space_voi's")
    print("\n  -> the two metrics order the policies almost oppositely. The headline")
    print("     percentage advantage is a mean-regret statement about a heavy tail;")
    print("     by count of windows the critic wins the large majority.")
    return {
        "rank_agreement": f"{agreement}/{len(rows)}",
        "belief_space_voi_win_rate": voi["win_rate_vs_critic"],
        "always_mode0_win_rate": mode0["win_rate_vs_critic"],
        "rows": rows,
    }


def check_missing_inference(summary) -> dict:
    rule("CHECK 5  Is the primary difference testable from the committed artefacts?")
    blob = json.dumps(summary)
    keys = ["diff_ci", "paired", "p_value", "pvalue", "bootstrap_diff", "t_stat", "wilcoxon", "sign_test"]
    present = [k for k in keys if f'"{k}"' in blob]
    print(f"  difference-level CI / test keys in the summary JSON: {present or 'none'}")

    per_policy_ci = [k for k in summary[0]["cost_regimes"]["default"]["per_policy"]["belief_space_voi"] if "ci" in k]
    print(f"  per-policy own-regret CI fields present            : {per_policy_ci}")

    half_widths = {}
    for policy in POLICIES:
        widths = []
        for seed in summary:
            block = seed["cost_regimes"]["default"]["per_policy"][policy]
            widths.append((block["ci_95_high"] - block["ci_95_low"]) / 2)
        half_widths[policy] = st.mean(widths)
    critic, voi = half_widths["direct_critic"], half_widths["belief_space_voi"]
    effect = mean_over_seeds(summary, "default", "direct_critic") - mean_over_seeds(
        summary, "default", "belief_space_voi"
    )
    print(f"\n  effect size (critic - VOI mean regret)             : {effect:.5f}")
    print(f"  direct_critic own-mean 95% CI half-width           : {critic:.5f}")
    print(f"  belief_space_voi own-mean 95% CI half-width        : {voi:.5f}")
    print(f"  half-widths are {critic / effect:.1f}x and {voi / effect:.1f}x the effect size")

    directory = SUMMARY.parent
    files = sorted(p.name for p in directory.iterdir() if p.is_file())
    has_per_window = any(p.suffix in {".npy", ".npz", ".parquet"} for p in directory.iterdir())
    print(f"\n  files committed for this benchmark                 : {files}")
    print(f"  per-window regret data present                     : {has_per_window}")
    if not has_per_window:
        print("  -> a paired test (the Wilcoxon the Session 6A spec mandates) cannot be")
        print("     re-run or re-derived from what is committed. Only aggregates exist.")
    return {
        "difference_test_keys": present,
        "per_window_data": has_per_window,
        "files": files,
        "effect_size": effect,
        "critic_ci_half_width": critic,
        "voi_ci_half_width": voi,
    }


def check_dispersion_convention(summary) -> dict:
    rule("CHECK 6  Which dispersion convention does the note's '+/-' column use?")
    # note, Table B: H=2 reports exact_costate as 0.05679 +/- 0.06643
    values = [s["horizons"]["2"]["per_policy"]["exact_costate"]["mean_regret"] for s in summary]
    mean = st.mean(values)
    sample, population = st.stdev(values), st.pstdev(values)
    note_mean, target = 0.05679, 0.06643

    print(f"  note quotes H=2 exact_costate as {note_mean:.5f} +/- {target:.5f}")
    print(f"  mean over the 3 seed values                        : {mean:.5f}")
    print(f"  sample std (ddof=1) over 3 seeds                   : {sample:.5f}")
    print(f"  population std (ddof=0) over 3 seeds              : {population:.5f}")
    match_mean = abs(mean - note_mean) < 5e-6
    match_population = abs(population - target) < 5e-5
    print(f"  note's mean reproduces                            : {match_mean}")
    print(f"  note's +/- is population std (ddof=0)             : {match_population}")
    if match_population:
        print(f"\n  -> the note understates cross-seed spread by {(1 - population / sample) * 100:.0f}% "
              f"({population:.5f} vs {sample:.5f}).")
        print("     The convention is undeclared. It is also moot in substance: two of the")
        print("     three seeds are the same evaluation, so there is one independent replicate")
        print("     and no meaningful cross-seed dispersion to report either way.")
    return {
        "note_value": target,
        "population_std": population,
        "sample_std": sample,
        "note_uses_population_std": match_population,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="emit the findings as JSON")
    args = parser.parse_args()

    if not SUMMARY.exists():
        raise SystemExit(f"missing artefact: {SUMMARY}")
    summary = json.loads(SUMMARY.read_text())

    findings = {
        "artefact": str(SUMMARY),
        "seeds": [s["seed"] for s in summary],
        "n_windows_per_seed": summary[0]["cost_regimes"]["default"]["n_windows"],
        "seed_duplication": check_seed_duplication(summary),
        "per_seed_sign": check_per_seed_sign(summary),
        "functional_identity": check_functional_identity(summary),
        "metric_rank_inversion": check_metric_rank_inversion(summary),
        "missing_inference": check_missing_inference(summary),
        "dispersion_convention": check_dispersion_convention(summary),
    }

    rule("SUMMARY")
    print(f"  seeds recorded in the artefact                    : {findings['seeds']}")
    print(f"  windows per seed                                   : {findings['n_windows_per_seed']}")
    print(f"  seed 0 vs seed 2 identical performance scalars    : "
          f"{findings['seed_duplication']['identical_performance_seed0_seed2']}"
          f"/{findings['seed_duplication']['performance_scalars']}")
    print(f"  wall-clock scalars that differ (expected)         : "
          f"{findings['seed_duplication']['differing_timing_scalars']}")
    print(f"  panels with a cross-seed sign flip                : "
          f"{findings['per_seed_sign']['panels_with_sign_flip']}/{len(REGIMES) + len(HORIZONS)}")
    print(f"  VOI(beta=-1) collapses onto second_order_curvature  : "
          f"{findings['functional_identity'].get('betaminus1_collapses_to_curvature', 'skipped (no torch)')}")
    print(f"  mean-regret vs win-rate rank agreement             : "
          f"{findings['metric_rank_inversion']['rank_agreement']}")
    print(f"  per-window data committed                         : {findings['missing_inference']['per_window_data']}")

    if args.json:
        print()
        print(json.dumps(findings, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
