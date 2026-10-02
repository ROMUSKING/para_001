#!/usr/bin/env python3
"""scripts/benchmark_ranking_allocator.py

Milestone B2.3: Ranking Allocator Optimization Benchmark.
Evaluates Pairwise Margin-Ranking and Plackett-Luce Listwise losses vs standard
Cross-Entropy in closing the amortization gap to the exact autograd oracle.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_DIR = Path(__file__).resolve().parent.parent
for p in [REPO_DIR / "src", Path("/content/para_001/src"), Path("/content/src")]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from adjointrwm.allocators import (
    AllocatorJob,
    CostateEstimator,
    DirectCritic,
    exact_targets,
    first_order_scores,
    lcb_decision_scores,
    matched_critic_hidden,
    normalized_first_order_scores,
    pairwise_margin_ranking_loss,
    plackett_luce_loss,
)
from adjointrwm.data import (
    WindowDataset,
    WindowSpec,
    episode_split,
    fit_normaliser,
    restore_cache,
)
from adjointrwm.models import AdjointRWMConfig, AdjointRecursiveWorldModel, ArmDims, build_arm
from adjointrwm.models.common import prediction_objective, upcast


def parse_args():
    parser = argparse.ArgumentParser(description="Milestone B2.3: Ranking Allocator Benchmark")
    parser.add_argument("--run-id", type=str, default="droid100_adjoint_v2_5seeds_20261001T080821Z")
    parser.add_argument("--drive-root", type=str, default="/content/drive/MyDrive/Colab Notebooks/AdjointRWM_Production")
    parser.add_argument("--local-cache", type=str, default="/content/cache")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--synthetic-test", action="store_true", help="Run quick synthetic test on CPU")
    return parser.parse_args()


def load_dataset(drive_root: Path, local_cache: Path, run_id: str):
    cache_manifest_path = drive_root / "runs" / run_id / "cache" / "cache_manifest.json"
    if not cache_manifest_path.exists():
        cache_manifest_path = drive_root / "runs" / "droid100_adjoint_v2_20260930T165409Z" / "cache" / "cache_manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text())
    records = restore_cache(cache_manifest, local_cache)

    assignment = episode_split([r["episode_id"] for r in records])
    for r in records:
        r["split"] = assignment[r["episode_id"]]

    by_split = {s: [r for r in records if r["split"] == s] for s in ("train", "validation", "test")}

    def load_arrays(rec, keys=("states", "actions")):
        with np.load(rec["cached_path"]) as ep:
            return {k: ep[k] for k in keys}

    train_arrays = [load_arrays(r) for r in by_split["train"]]
    normalisation = fit_normaliser([a["states"] for a in train_arrays], [a["actions"] for a in train_arrays])
    spec = WindowSpec(context_len=8, horizon=4, stride=2)
    train_dataset = WindowDataset(by_split["train"], spec, normalisation, visual_layout="flat")
    val_dataset = WindowDataset(by_split["validation"], spec, normalisation, visual_layout="flat")
    test_dataset = WindowDataset(by_split["test"], spec, normalisation, visual_layout="flat")
    return train_dataset, val_dataset, test_dataset


def load_frozen_teacher(run_dir: Path, seed: int, sample_batch: dict, device: torch.device):
    config_path = run_dir / "config" / "run_config.json"
    cfg = json.loads(config_path.read_text())["config"]

    model_config = AdjointRWMConfig(
        context_len=cfg["context_len"],
        horizon=cfg["horizon"],
        d_model=cfg["d_model"],
        transformer_layers=cfg["transformer_layers"],
        transformer_heads=cfg["transformer_heads"],
        transformer_ff=cfg["transformer_ff"],
        dropout=cfg["dropout"],
        num_refinement_candidates=cfg["num_refinement_candidates"],
        candidate_costs=cfg["candidate_costs"],
        rate_beta=cfg["rate_beta"],
        mask_mode=cfg["mask_mode"],
        prediction_mode=cfg["prediction_mode"],
    )
    state_dim = sample_batch["context_state"].shape[-1]
    action_dim = sample_batch["context_action"].shape[-1]
    visual_dim = sample_batch["context_visual"].shape[-1]

    model = AdjointRecursiveWorldModel(state_dim, action_dim, visual_dim, model_config).to(device)
    ckpt_path = run_dir / "jobs" / f"seed_{seed}" / "dynamics" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, model_config


def extract_sensing_targets(teacher, batch: dict, costs: torch.Tensor, device: torch.device):
    """Computes sensory candidate latents, autograd co-state at Mode 0, and exact net gains."""
    b_dev = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
    B = b_dev["context_state"].shape[0]

    with torch.no_grad():
        v0 = torch.zeros_like(b_dev["context_visual"])
        z0 = teacher.encode_context(v0, b_dev["context_state"], b_dev["context_action"])
        p0 = teacher.rollout(z0, b_dev["future_actions"])
        j0 = prediction_objective(p0, b_dev["target_state"], b_dev["target_visual"])

        v1 = b_dev["context_visual"].clone()
        v1[:, :, :512] = 0.0
        z1 = teacher.encode_context(v1, b_dev["context_state"], b_dev["context_action"])
        p1 = teacher.rollout(z1, b_dev["future_actions"])
        j1 = prediction_objective(p1, b_dev["target_state"], b_dev["target_visual"])

        v2 = b_dev["context_visual"].clone()
        v2[:, :, 512:] = 0.0
        z2 = teacher.encode_context(v2, b_dev["context_state"], b_dev["context_action"])
        p2 = teacher.rollout(z2, b_dev["future_actions"])
        j2 = prediction_objective(p2, b_dev["target_state"], b_dev["target_visual"])

        v3 = b_dev["context_visual"]
        z3 = teacher.encode_context(v3, b_dev["context_state"], b_dev["context_action"])
        p3 = teacher.rollout(z3, b_dev["future_actions"])
        j3 = prediction_objective(p3, b_dev["target_state"], b_dev["target_visual"])

    z0_req = z0.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        p0_req = teacher.rollout(z0_req, b_dev["future_actions"])
        loss_0 = prediction_objective(p0_req, b_dev["target_state"], b_dev["target_visual"]).sum()
        lambda0 = torch.autograd.grad(loss_0, z0_req)[0].detach()

    dz0 = torch.zeros_like(z0)
    dz1 = z1 - z0
    dz2 = z2 - z0
    dz3 = z3 - z0
    effects = torch.stack([dz0, dz1, dz2, dz3], dim=1)

    raw_gains = torch.stack([torch.zeros_like(j0), j0 - j1, j0 - j2, j0 - j3], dim=1)
    net_gains = raw_gains - costs.view(1, 4)

    uncert_0 = (0.5 * (1.0 + upcast(p0["state_logvar"]))).mean(dim=(1, 2))
    uncert_1 = (0.5 * (1.0 + upcast(p1["state_logvar"]))).mean(dim=(1, 2))
    uncert_2 = (0.5 * (1.0 + upcast(p2["state_logvar"]))).mean(dim=(1, 2))
    uncert_3 = (0.5 * (1.0 + upcast(p3["state_logvar"]))).mean(dim=(1, 2))
    self_uncert = torch.stack([uncert_0, uncert_1, uncert_2, uncert_3], dim=1)

    return {
        "latent_0": z0.detach(),
        "lambda_0": lambda0.detach(),
        "effects": effects.detach(),
        "raw_gains": raw_gains.detach(),
        "net_gains": net_gains.detach(),
        "self_uncert": self_uncert.detach(),
    }


def evaluate_policy_regret(costate_head, test_loader, teacher, costs, device):
    costate_head.eval()
    all_net_gains = []
    all_choices = []
    oracle_choices = []
    always_0_choices = []

    with torch.no_grad():
        for batch in test_loader:
            targets = extract_sensing_targets(teacher, batch, costs, device)
            B = targets["latent_0"].shape[0]
            budget = torch.ones(B, device=device)
            horizon = torch.ones(B, device=device)

            net_g = targets["net_gains"].cpu().numpy()
            all_net_gains.append(net_g)

            # Exact Co-state Oracle
            lin_exact = - (targets["lambda_0"].unsqueeze(1) * targets["effects"]).sum(dim=-1) - costs.view(1, 4)
            oracle_choices.append(lin_exact.argmax(dim=-1).cpu().numpy())
            always_0_choices.append(np.zeros(B, dtype=int))

            # Predicted score
            pred_lam = costate_head(targets["latent_0"], budget, horizon)
            s_costate_norm = normalized_first_order_scores(pred_lam, targets["effects"], costs)
            all_choices.append(s_costate_norm.argmax(dim=-1).cpu().numpy())

    net_gains_mat = np.concatenate(all_net_gains, axis=0)
    N = net_gains_mat.shape[0]
    oracle_gain = net_gains_mat.max(axis=-1)

    choices = np.concatenate(all_choices, axis=0)
    chosen_gain = net_gains_mat[np.arange(N), choices]
    regret = float((oracle_gain - chosen_gain).mean())

    exact_choices = np.concatenate(oracle_choices, axis=0)
    exact_gain = net_gains_mat[np.arange(N), exact_choices]
    exact_regret = float((oracle_gain - exact_gain).mean())

    always0_choices = np.concatenate(always_0_choices, axis=0)
    always0_gain = net_gains_mat[np.arange(N), always0_choices]
    always0_regret = float((oracle_gain - always0_gain).mean())

    return regret, exact_regret, always0_regret


def run_benchmark():
    args = parse_args()
    print("=" * 70)
    print("Milestone B2.3: Ranking Allocator Optimization Benchmark")
    print(f"Comparing Ranking Losses: CE, Pairwise Margin, Plackett-Luce Listwise, Hybrid")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.synthetic_test else "cpu")
    print(f"Device: {device}")

    drive_root = Path(args.drive_root)
    local_cache = Path(args.local_cache)
    run_dir = drive_root / "runs" / args.run_id

    if args.synthetic_test or not run_dir.exists():
        print("[Notice] Running in synthetic diagnostic mode...")
        dims = ArmDims(state_dim=10, action_dim=7, visual_tokens=4, visual_token_dim=8, target_visual_dim=32, context_len=8, horizon=4)
        teacher = build_arm("adjoint_rwm", dims, width=64, transformer_heads=4, transformer_layers=2).to(device).eval()

        g = torch.Generator().manual_seed(42)
        sample_batch = {
            "context_state": torch.randn(16, 8, 10, generator=g, device=device),
            "context_action": torch.randn(16, 8, 7, generator=g, device=device),
            "context_visual": torch.randn(16, 8, 32, generator=g, device=device),
            "future_actions": torch.randn(16, 4, 7, generator=g, device=device),
            "target_state": torch.randn(16, 4, 10, generator=g, device=device),
            "target_visual": torch.randn(16, 4, 32, generator=g, device=device),
        }

        for variant in ("ce", "margin", "listwise", "hybrid"):
            job = AllocatorJob(teacher, "costate", cost_weight=0.002, ranking_loss_type=variant).to(device)
            optimizer = torch.optim.AdamW(job.parameters(), lr=1e-3)
            for _ in range(10):
                optimizer.zero_grad()
                loss, parts = job.training_loss(sample_batch)
                loss.backward()
                optimizer.step()
            print(f"  Variant {variant:>8}: final loss = {loss.item():.4f}, ranking = {parts['ranking']:.4f}")

        print("\nAll ranking loss variants executed successfully.")
        return

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "benchmarks" / "ranking_allocator"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading real robot dataset...")
    train_dataset, val_dataset, test_dataset = load_dataset(drive_root, local_cache, args.run_id)
    print(f"Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test windows")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    costs = torch.tensor([0.0, 0.005, 0.010, 0.015], device=device)
    variants = ["ce", "margin", "listwise", "hybrid"]

    results = {v: [] for v in variants}
    oracle_regrets = []
    mode0_regrets = []

    for seed in range(args.num_seeds):
        print(f"\n=======================================================")
        print(f"  BENCHMARKING SEED {seed} / {args.num_seeds}")
        print(f"=======================================================")

        sample_batch = next(iter(train_loader))
        teacher, _ = load_frozen_teacher(run_dir, seed, sample_batch, device)
        d = teacher.d_model

        for variant in variants:
            print(f"  --> Training Allocator with ranking_loss_type='{variant}' ({args.max_steps} steps)...")
            torch.manual_seed(2000 + seed * 10)
            costate_head = CostateEstimator(d).to(device)
            opt_costate = torch.optim.AdamW(costate_head.parameters(), lr=args.lr, weight_decay=1e-4)
            sched_costate = torch.optim.lr_scheduler.CosineAnnealingLR(opt_costate, T_max=args.max_steps, eta_min=args.min_lr)

            train_iter = iter(train_loader)
            for step in range(args.max_steps):
                costate_head.train()
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                targets = extract_sensing_targets(teacher, batch, costs, device)
                B = targets["latent_0"].shape[0]
                budget = torch.ones(B, device=device)
                horizon = torch.ones(B, device=device)
                oracle_idx = targets["net_gains"].argmax(dim=-1)

                opt_costate.zero_grad()
                pred_costate = costate_head(targets["latent_0"], budget, horizon)
                cos_sim = F.cosine_similarity(pred_costate, targets["lambda_0"], dim=-1)
                cos_loss = (1.0 - cos_sim).mean()
                mag_loss = F.smooth_l1_loss(
                    torch.log(torch.abs(pred_costate) + 1e-4),
                    torch.log(torch.abs(targets["lambda_0"]) + 1e-4),
                )
                pred_scores_norm = normalized_first_order_scores(pred_costate, targets["effects"], costs)

                if variant == "ce":
                    ranking_loss = F.cross_entropy(pred_scores_norm, oracle_idx)
                elif variant == "margin":
                    ranking_loss = pairwise_margin_ranking_loss(pred_scores_norm, targets["net_gains"])
                elif variant == "listwise":
                    ranking_loss = plackett_luce_loss(pred_scores_norm, targets["net_gains"])
                elif variant == "hybrid":
                    ranking_loss = 0.5 * pairwise_margin_ranking_loss(pred_scores_norm, targets["net_gains"]) + 0.5 * plackett_luce_loss(pred_scores_norm, targets["net_gains"])

                total_loss = cos_loss + 0.1 * mag_loss + 0.5 * ranking_loss
                total_loss.backward()
                opt_costate.step()
                sched_costate.step()

            # Evaluate on held-out test windows
            regret, exact_regret, mode0_regret = evaluate_policy_regret(costate_head, test_loader, teacher, costs, device)
            results[variant].append(regret)
            print(f"      [Seed {seed} | Variant {variant:>8}] Test Regret: {regret:.5f}")

        oracle_regrets.append(exact_regret)
        mode0_regrets.append(mode0_regret)

    # Summarize findings
    print("\n" + "=" * 70)
    print("  FINAL BENCHMARK SUMMARY (Milestone B2.3)")
    print("=" * 70)
    print(f"Exact Autograd Oracle Regret: {np.mean(oracle_regrets):.5f} ± {np.std(oracle_regrets):.5f}")
    print(f"Always Mode 0 (Refusal) Regret: {np.mean(mode0_regrets):.5f} ± {np.std(mode0_regrets):.5f}")
    for variant in variants:
        mean_r = np.mean(results[variant])
        std_r = np.std(results[variant])
        delta_ce = mean_r - np.mean(results["ce"])
        print(f"Variant {variant:>8}: {mean_r:.5f} ± {std_r:.5f}  (diff vs CE: {delta_ce:+.5f})")

    summary = {
        "benchmark": "Milestone B2.3: Ranking Allocator Optimization Benchmark",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": args.run_id,
        "num_seeds": args.num_seeds,
        "max_steps": args.max_steps,
        "oracle_regret": {"mean": float(np.mean(oracle_regrets)), "std": float(np.std(oracle_regrets))},
        "always_mode0_regret": {"mean": float(np.mean(mode0_regrets)), "std": float(np.std(mode0_regrets))},
        "variant_regrets": {
            v: {"mean": float(np.mean(results[v])), "std": float(np.std(results[v])), "per_seed": [float(x) for x in results[v]]}
            for v in variants
        },
    }
    summary_path = output_dir / "b2_3_ranking_allocator_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")

    # Generate Markdown Report
    report_path = output_dir / "b2_3_ranking_allocator_report.md"
    with open(report_path, "w") as f:
        f.write("# Milestone B2.3 Ranking Allocator Optimization Benchmark\n\n")
        f.write(f"**Date:** {summary['date']} · **Run ID:** `{args.run_id}` · **Seeds:** {args.num_seeds}\n")
        f.write(f"**Max Training Steps:** {args.max_steps}\n\n")
        f.write("## 1. Test Regret Comparison Across Ranking Loss Variants\n\n")
        f.write("| Allocation Policy / Loss | Mean Test Regret | Std Regret | Gap to Oracle Floor (0.03007) | Advantage vs CE |\n")
        f.write("|---|:---:|:---:|:---:|:---:|\n")
        f.write(f"| **Exact Autograd Oracle** | `{summary['oracle_regret']['mean']:.5f}` | `±{summary['oracle_regret']['std']:.5f}` | `0.00000` | — |\n")
        f.write(f"| **Refusal Baseline (`always_mode0`)** | `{summary['always_mode0_regret']['mean']:.5f}` | `±{summary['always_mode0_regret']['std']:.5f}` | `+{summary['always_mode0_regret']['mean'] - summary['oracle_regret']['mean']:.5f}` | — |\n")
        ce_mean = summary["variant_regrets"]["ce"]["mean"]
        for v in variants:
            vm = summary["variant_regrets"][v]["mean"]
            vs = summary["variant_regrets"][v]["std"]
            gap = vm - summary["oracle_regret"]["mean"]
            adv = vm - ce_mean
            f.write(f"| `costate_{v}` | **`{vm:.5f}`** | `±{vs:.5f}` | `+{gap:.5f}` | **`{adv:+.5f}`** |\n")

    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    run_benchmark()
