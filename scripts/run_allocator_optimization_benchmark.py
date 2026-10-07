#!/usr/bin/env python3
"""Milestone B2.2: Allocator Optimization Benchmark on DROID-100.

Investigates closing the amortization gap between amortized allocation and the
exact autograd co-state oracle floor (0.03007) via:
1. Optimization Horizon: 300 steps (B2.1 baseline) vs 1,000 steps vs 2,500 steps with Cosine LR decay.
2. Coupling Metric: Raw First-Order Dot Product vs PARA Scale-Invariant Cosine Coupling.
3. Decision Rule: Standard Greedy Argmax vs Cost-Aware LCB Triggering (kappa in {0.5, 1.0}).
4. Fairness: Matched Direct Critic trained under identical schedules and parameter parity (~1.315M params).
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

from adjointrwm.data import (
    WindowDataset,
    WindowSpec,
    episode_split,
    fit_normaliser,
    restore_cache,
)
from adjointrwm.models import AdjointRecursiveWorldModel, AdjointRWMConfig
from adjointrwm.models.common import prediction_objective, upcast
from adjointrwm.allocators import (
    CostateEstimator,
    DirectCritic,
    first_order_scores,
    lcb_decision_scores,
    matched_critic_hidden,
    normalized_first_order_scores,
    with_hold_zero,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Allocator Optimization Benchmark")
    parser.add_argument("--run-id", type=str, default="droid100_adjoint_v2_5seeds_20261001T080821Z")
    parser.add_argument("--drive-root", type=str, default="/content/drive/MyDrive/Colab Notebooks/AdjointRWM_Production")
    parser.add_argument("--local-cache", type=str, default="/content/cache")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--eval-checkpoints", type=int, nargs="+", default=[300, 1000, 2500])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--critic-width-scale", type=float, default=1.0,
                        help="§1893 P2 capacity probe: scales DirectCritic hidden width "
                             "(0.5/1.0/2.0/4.0). Costate head untouched.")
    parser.add_argument("--probe-panels", action="store_true",
                        help="§1893 P0 floor: adjacent-cost-rung panels (oracle modes 0-1 vs "
                             "2-3) with frozen paired tests on the validation split.")
    parser.add_argument("--critic-only", action="store_true",
                        help="Skip costate-head training and costate policies (same critic "
                             "trajectory: separate optimisers, no gradient interaction). "
                             "Halves probe compute; critic lineage unchanged.")
    parser.add_argument("--surrogate-every", type=int, default=200,
                        help="§1893 P1 convergence: log train/val critic surrogate every N steps.")
    return parser.parse_args()


def adjacent_rung_panels(net_gains_mat: np.ndarray, choices: dict) -> dict:
    """Pure-numpy P0 floor panels (testable without torch or a teacher).

    Oracle-choice strata split the two adjacent cost rungs: low (oracle modes 0-1,
    costs 0.0/0.005) and high (modes 2-3, costs 0.010/0.015). Per stratum: trimmed-mean
    regrets for the critic, random-expected and uncertainty policies, paired Wilcoxon
    critic-vs-each-baseline, and adaptive gain (always_mode0 regret minus critic
    regret, positive favours the critic). ``choices`` maps policy name -> per-window
    mode indices; ``random_expected`` uses the per-window mean gain.
    """
    from adjointrwm.spatial_selection import trimmed_mean, wilcoxon_signed_rank

    oracle = net_gains_mat.argmax(axis=-1)
    oracle_gain = net_gains_mat.max(axis=-1)
    regrets = {}
    for name, chosen in choices.items():
        if name == "random_expected":
            regrets[name] = oracle_gain - net_gains_mat.mean(axis=-1)
        else:
            regrets[name] = oracle_gain - net_gains_mat[np.arange(len(oracle)), np.asarray(chosen)]
    panels = {}
    for stratum, mask in (("low_rung", oracle <= 1), ("high_rung", oracle >= 2)):
        entry = {"n": int(mask.sum())}
        if entry["n"] == 0:
            entry.update({"critic_trimmed": float("nan"), "random_trimmed": float("nan"),
                          "uncertainty_trimmed": float("nan"),
                          "adaptive_gain_trimmed": float("nan"),
                          "vs_random": None, "vs_uncertainty": None})
        else:
            entry["critic_trimmed"] = trimmed_mean(regrets["allocator_critic"][mask])
            entry["random_trimmed"] = trimmed_mean(regrets["random_expected"][mask])
            entry["uncertainty_trimmed"] = trimmed_mean(regrets["uncertainty"][mask])
            entry["adaptive_gain_trimmed"] = trimmed_mean(
                (regrets["always_mode0"] - regrets["allocator_critic"])[mask])
            entry["vs_random"] = wilcoxon_signed_rank(
                (regrets["random_expected"] - regrets["allocator_critic"])[mask])
            entry["vs_uncertainty"] = wilcoxon_signed_rank(
                (regrets["uncertainty"] - regrets["allocator_critic"])[mask])
        panels[stratum] = entry
    return panels


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
        # Mode 0: Proprio only
        v0 = torch.zeros_like(b_dev["context_visual"])
        z0 = teacher.encode_context(v0, b_dev["context_state"], b_dev["context_action"])
        p0 = teacher.rollout(z0, b_dev["future_actions"])
        j0 = prediction_objective(p0, b_dev["target_state"], b_dev["target_visual"])

        # Mode 1: Wrist only
        v1 = b_dev["context_visual"].clone()
        v1[:, :, :512] = 0.0
        z1 = teacher.encode_context(v1, b_dev["context_state"], b_dev["context_action"])
        p1 = teacher.rollout(z1, b_dev["future_actions"])
        j1 = prediction_objective(p1, b_dev["target_state"], b_dev["target_visual"])

        # Mode 2: Exterior only
        v2 = b_dev["context_visual"].clone()
        v2[:, :, 512:] = 0.0
        z2 = teacher.encode_context(v2, b_dev["context_state"], b_dev["context_action"])
        p2 = teacher.rollout(z2, b_dev["future_actions"])
        j2 = prediction_objective(p2, b_dev["target_state"], b_dev["target_visual"])

        # Mode 3: Full Observation
        v3 = b_dev["context_visual"]
        z3 = teacher.encode_context(v3, b_dev["context_state"], b_dev["context_action"])
        p3 = teacher.rollout(z3, b_dev["future_actions"])
        j3 = prediction_objective(p3, b_dev["target_state"], b_dev["target_visual"])

    # Autograd co-state at Mode 0
    z0_req = z0.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        p0_req = teacher.rollout(z0_req, b_dev["future_actions"])
        loss_0 = prediction_objective(p0_req, b_dev["target_state"], b_dev["target_visual"]).sum()
        lambda0 = torch.autograd.grad(loss_0, z0_req)[0].detach()

    # Latent effects relative to z0: Delta z_m
    dz0 = torch.zeros_like(z0)
    dz1 = z1 - z0
    dz2 = z2 - z0
    dz3 = z3 - z0
    effects = torch.stack([dz0, dz1, dz2, dz3], dim=1)  # [B, 4, D]

    # Raw gains: J0 - J_m
    raw_gains = torch.stack([torch.zeros_like(j0), j0 - j1, j0 - j2, j0 - j3], dim=1)  # [B, 4]
    net_gains = raw_gains - costs.view(1, 4)  # [B, 4]

    # Uncertainty heuristic: average logvar over future horizon
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


def evaluate_policies_at_step(teacher, costate_head, critic_head, test_loader, costs: torch.Tensor, device: torch.device,
                            return_panels: bool = False):
    if costate_head is not None:
        costate_head.eval()
    critic_head.eval()

    all_net_gains = []
    all_choices = {
        "exact_costate": [],
        "always_mode0": [],
        "always_mode1": [],
        "always_mode2": [],
        "always_mode3": [],
        "random_expected": [],
        "uncertainty": [],
        # Adjoint policies
        "allocator_costate": [],
        "allocator_costate_norm": [],
        "allocator_costate_lcb_05": [],
        "allocator_costate_norm_lcb_05": [],
        "allocator_costate_norm_lcb_10": [],
        # Matched Critic policies
        "allocator_critic": [],
        "allocator_critic_lcb_05": [],
        "allocator_critic_lcb_10": [],
    }

    with torch.no_grad():
        for batch in test_loader:
            targets = extract_sensing_targets(teacher, batch, costs, device)
            B = targets["latent_0"].shape[0]
            budget = torch.ones(B, device=device)
            horizon = torch.ones(B, device=device)

            net_g = targets["net_gains"].cpu().numpy()
            all_net_gains.append(net_g)
            uncert = targets["self_uncert"]

            # Exact Co-state Oracle
            lin_exact = - (targets["lambda_0"].unsqueeze(1) * targets["effects"]).sum(dim=-1) - costs.view(1, 4)
            all_choices["exact_costate"].append(lin_exact.argmax(dim=-1).cpu().numpy())

            # Fixed baselines
            all_choices["always_mode0"].append(np.zeros(B, dtype=int))
            all_choices["always_mode1"].append(np.ones(B, dtype=int))
            all_choices["always_mode2"].append(np.full(B, 2, dtype=int))
            all_choices["always_mode3"].append(np.full(B, 3, dtype=int))
            all_choices["uncertainty"].append(uncert.argmax(dim=-1).cpu().numpy())

            # Amortized Co-state predictions (skipped under --critic-only)
            if costate_head is not None:
                pred_lam = costate_head(targets["latent_0"], budget, horizon)
                s_costate = first_order_scores(pred_lam, targets["effects"], costs)
                s_costate_norm = normalized_first_order_scores(pred_lam, targets["effects"], costs)

                all_choices["allocator_costate"].append(s_costate.argmax(dim=-1).cpu().numpy())
                all_choices["allocator_costate_norm"].append(s_costate_norm.argmax(dim=-1).cpu().numpy())

                # LCB Co-state decisions
                s_costate_lcb_05 = lcb_decision_scores(s_costate, uncert, kappa=0.5)
                s_costate_norm_lcb_05 = lcb_decision_scores(s_costate_norm, uncert, kappa=0.5)
                s_costate_norm_lcb_10 = lcb_decision_scores(s_costate_norm, uncert, kappa=1.0)

                all_choices["allocator_costate_lcb_05"].append(s_costate_lcb_05.argmax(dim=-1).cpu().numpy())
                all_choices["allocator_costate_norm_lcb_05"].append(s_costate_norm_lcb_05.argmax(dim=-1).cpu().numpy())
                all_choices["allocator_costate_norm_lcb_10"].append(s_costate_norm_lcb_10.argmax(dim=-1).cpu().numpy())

            # Direct Critic predictions
            pred_gains = critic_head(targets["latent_0"], targets["effects"], costs, budget, horizon)
            all_choices["allocator_critic"].append(pred_gains.argmax(dim=-1).cpu().numpy())

            # LCB Critic decisions
            s_critic_lcb_05 = lcb_decision_scores(pred_gains, uncert, kappa=0.5)
            s_critic_lcb_10 = lcb_decision_scores(pred_gains, uncert, kappa=1.0)
            all_choices["allocator_critic_lcb_05"].append(s_critic_lcb_05.argmax(dim=-1).cpu().numpy())
            all_choices["allocator_critic_lcb_10"].append(s_critic_lcb_10.argmax(dim=-1).cpu().numpy())

    net_gains_mat = np.concatenate(all_net_gains, axis=0)  # [N, 4]
    N = net_gains_mat.shape[0]
    oracle_gain = net_gains_mat.max(axis=-1)  # [N]

    regret_dict = {}
    for policy_name, choice_list in all_choices.items():
        if policy_name == "random_expected":
            chosen_gain = net_gains_mat.mean(axis=-1)
        else:
            if len(choice_list) == 0:
                continue  # --critic-only: costate policies never evaluated
            choices = np.concatenate(choice_list, axis=0)
            chosen_gain = net_gains_mat[np.arange(N), choices]
        regret = oracle_gain - chosen_gain
        regret_dict[policy_name] = float(regret.mean())

    if not return_panels:
        return regret_dict
    concat = {}
    for name, lst in all_choices.items():
        if name == "random_expected":
            concat[name] = None
        elif len(lst) > 0:
            concat[name] = np.concatenate(lst, axis=0)
        # --critic-only: costate lists stay empty and are omitted, never concatenated.
    # Panels need per-window choices for the scored policies plus mode0 for adaptive gain.
    panel_choices = {k: v for k, v in concat.items()
                     if k in ("allocator_critic", "uncertainty", "always_mode0")}
    panel_choices["random_expected"] = None
    return regret_dict, adjacent_rung_panels(net_gains_mat, panel_choices)


def run_benchmark():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Allocator Optimization Benchmark on device: {device}")

    drive_root = Path(args.drive_root)
    local_cache = Path(args.local_cache)
    run_dir = drive_root / "runs" / args.run_id

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "benchmarks" / "allocator_optimization"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading DROID-100 dataset...")
    train_dataset, val_dataset, test_dataset = load_dataset(drive_root, local_cache, args.run_id)
    print(f"Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test windows")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    costs = torch.tensor([0.0, 0.005, 0.010, 0.015], device=device)

    # Tracking results: checkpoint -> seed -> policy -> regret
    checkpoint_results = {cp: [] for cp in args.eval_checkpoints}
    val_checkpoint_results = {cp: [] for cp in args.eval_checkpoints}
    val_panels_results = {cp: [] for cp in args.eval_checkpoints}
    surrogate_ledger = []  # (seed, step, train_critic_loss, val_critic_surrogate)
    seed_finite = []

    for seed in range(args.num_seeds):
        print(f"\n=======================================================")
        print(f"  OPTIMIZING & BENCHMARKING SEED {seed} / {args.num_seeds}")
        print(f"=======================================================")
        torch.manual_seed(1000 + seed)
        np.random.seed(1000 + seed)

        sample_batch = next(iter(train_loader))
        teacher, _ = load_frozen_teacher(run_dir, seed, sample_batch, device)

        d = teacher.d_model
        costate_head = CostateEstimator(d).to(device)
        critic_hidden = max(1, round(matched_critic_hidden(d) * args.critic_width_scale))
        critic_head = DirectCritic(d, critic_hidden).to(device)
        critic_params = sum(p.numel() for p in critic_head.parameters())

        opt_costate = torch.optim.AdamW(costate_head.parameters(), lr=args.lr, weight_decay=1e-4)
        opt_critic = torch.optim.AdamW(critic_head.parameters(), lr=args.lr, weight_decay=1e-4)

        sched_costate = torch.optim.lr_scheduler.CosineAnnealingLR(opt_costate, T_max=args.max_steps, eta_min=args.min_lr)
        sched_critic = torch.optim.lr_scheduler.CosineAnnealingLR(opt_critic, T_max=args.max_steps, eta_min=args.min_lr)

        train_iter = iter(train_loader)

        for step in range(args.max_steps):
            if not args.critic_only:
                costate_head.train()
            critic_head.train()

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

            # 1. Train Co-State Head
            if not args.critic_only:
                opt_costate.zero_grad()
                pred_costate = costate_head(targets["latent_0"], budget, horizon)
                cos_sim = F.cosine_similarity(pred_costate, targets["lambda_0"], dim=-1)
                cos_loss = (1.0 - cos_sim).mean()
                mag_loss = F.smooth_l1_loss(
                    torch.log(torch.abs(pred_costate) + 1e-4),
                    torch.log(torch.abs(targets["lambda_0"]) + 1e-4),
                )
                # Ranking loss with normalized coupling
                pred_scores_norm = normalized_first_order_scores(pred_costate, targets["effects"], costs)
                ce_loss_costate = F.cross_entropy(pred_scores_norm, oracle_idx)

                loss_costate = cos_loss + 0.1 * mag_loss + 0.5 * ce_loss_costate
                loss_costate.backward()
                opt_costate.step()
                sched_costate.step()
            else:
                loss_costate = torch.zeros((), device=device)

            # 2. Train Direct Critic
            opt_critic.zero_grad()
            pred_gains = critic_head(targets["latent_0"], targets["effects"], costs, budget, horizon)
            l1_loss = F.smooth_l1_loss(pred_gains, targets["net_gains"])
            ce_loss_critic = F.cross_entropy(pred_gains, oracle_idx)

            loss_critic = l1_loss + 0.5 * ce_loss_critic
            loss_critic.backward()
            opt_critic.step()
            sched_critic.step()

            cur_step = step + 1
            if cur_step % 200 == 0:
                if args.critic_only:
                    print(f"  [Seed {seed}] Step {cur_step:4d}/{args.max_steps} | Critic Loss: {loss_critic.item():.4f}")
                else:
                    print(f"  [Seed {seed}] Step {cur_step:4d}/{args.max_steps} | Costate Loss: {loss_costate.item():.4f} (cos: {cos_loss.item():.4f}) | Critic Loss: {loss_critic.item():.4f}")

            # §1893 P1 convergence ledger: train critic surrogate every step-log, plus a
            # no-grad validation surrogate pass every --surrogate-every steps. The B2.2 loss
            # curve exists only in lost stdout; this ledger is the convergence evidence.
            if cur_step % args.surrogate_every == 0:
                critic_head.eval()
                val_surrogate, val_batches = 0.0, 0
                with torch.no_grad():
                    for vbatch in val_loader:
                        vt = extract_sensing_targets(teacher, vbatch, costs, device)
                        vp = critic_head(vt["latent_0"], vt["effects"], costs,
                                         torch.ones(vt["latent_0"].shape[0], device=device),
                                         torch.ones(vt["latent_0"].shape[0], device=device))
                        vo = vt["net_gains"].argmax(dim=-1)
                        val_surrogate += (F.smooth_l1_loss(vp, vt["net_gains"])
                                          + 0.5 * F.cross_entropy(vp, vo)).item()
                        val_batches += 1
                        del vbatch
                critic_head.train()
                surrogate_ledger.append({
                    "seed": seed, "step": cur_step,
                    "train_critic_loss": float(loss_critic.item()),
                    "val_critic_surrogate": float(val_surrogate / max(1, val_batches)),
                })

            # Checkpoint evaluation
            if cur_step in args.eval_checkpoints:
                print(f"  Evaluating held-out test regret at Step {cur_step}...")
                eval_costate = None if args.critic_only else costate_head
                step_regrets = evaluate_policies_at_step(teacher, eval_costate, critic_head, test_loader, costs, device)
                checkpoint_results[cur_step].append(step_regrets)
                print(f"    Step {cur_step} | Oracle: {step_regrets['exact_costate']:.5f} | Mode0: {step_regrets['always_mode0']:.5f} | Critic: {step_regrets['allocator_critic']:.5f}" + (
                    f" | Costate Norm: {step_regrets['allocator_costate_norm']:.5f}" if "allocator_costate_norm" in step_regrets else ""))
                print(f"  Evaluating validation regret at Step {cur_step}...")
                val_out = evaluate_policies_at_step(teacher, eval_costate, critic_head, val_loader, costs, device,
                                                    return_panels=args.probe_panels)
                if args.probe_panels:
                    val_regrets, val_panels = val_out
                    val_panels_results[cur_step].append(val_panels)
                else:
                    val_regrets = val_out
                val_checkpoint_results[cur_step].append(val_regrets)
                print(f"    Step {cur_step} (val) | Critic: {val_regrets['allocator_critic']:.5f} | "
                      f"Random: {val_regrets['random_expected']:.5f} | Unc: {val_regrets['uncertainty']:.5f}")

        seed_finite.append(bool(np.isfinite([e["train_critic_loss"] for e in surrogate_ledger
                                             if e["seed"] == seed]).all()))
        # Partial flush: a destroyed session keeps every completed seed. Final assembly
        # below only re-aggregates these files; nothing is decided from partials alone.
        partial_path = write_seed_partial(
            output_dir, seed, args, checkpoint_results, val_checkpoint_results,
            val_panels_results, surrogate_ledger, seed_finite[-1])
        print(f"  [Seed {seed}] partial results flushed to {partial_path}")

    # Pool results across all 5 seeds for each checkpoint
    summary = {
        "benchmark": "Milestone B2.2: Allocator Optimization Benchmark",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": args.run_id,
        "num_seeds": args.num_seeds,
        "total_test_windows": len(test_dataset) * args.num_seeds,
        "total_val_windows": len(val_dataset) * args.num_seeds,
        "eval_checkpoints": args.eval_checkpoints,
        "sensing_costs": [float(c) for c in costs.cpu().numpy()],
        "critic_width_scale": args.critic_width_scale,
        "critic_only": args.critic_only,
        "probe_panels": args.probe_panels,
        "seed_finite": seed_finite,
        "checkpoints": {},
        "val_checkpoints": {},
        "surrogate_ledger": surrogate_ledger,
    }

    for cp in args.eval_checkpoints:
        seed_list = checkpoint_results[cp]
        policies = list(seed_list[0].keys())
        mean_regrets = {p: float(np.mean([s[p] for s in seed_list])) for p in policies}
        std_regrets = {p: float(np.std([s[p] for s in seed_list])) for p in policies}

        diff_norm_vs_critic = [s["allocator_costate_norm"] - s["allocator_critic"] for s in seed_list
                               if "allocator_costate_norm" in s]
        diff_lcb_vs_critic = [s["allocator_costate_norm_lcb_05"] - s["allocator_critic"] for s in seed_list
                              if "allocator_costate_norm_lcb_05" in s]

        summary["checkpoints"][str(cp)] = {
            "mean_regrets": mean_regrets,
            "std_regrets": std_regrets,
            "diff_norm_minus_critic": float(np.mean(diff_norm_vs_critic)) if diff_norm_vs_critic else float("nan"),
            "diff_norm_lcb_minus_critic": float(np.mean(diff_lcb_vs_critic)) if diff_lcb_vs_critic else float("nan"),
            "per_seed_regrets": seed_list,
        }
        val_list = val_checkpoint_results[cp]
        vpolicies = list(val_list[0].keys())
        summary["val_checkpoints"][str(cp)] = {
            "mean_regrets": {p: float(np.mean([s[p] for s in val_list])) for p in vpolicies},
            "per_seed_regrets": val_list,
        }
        if args.probe_panels:
            summary["val_checkpoints"][str(cp)]["panels_per_seed"] = val_panels_results[cp]

    # Save summary JSON
    summary_path = output_dir / "b2_2_allocator_optimization_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved benchmark summary to: {summary_path}")

    # Generate Markdown Report
    report_path = output_dir / "b2_2_allocator_optimization_report.md"
    with open(report_path, "w") as f:
        f.write("# Milestone B2.2 Allocator Optimization Benchmark: Closing the Amortization Gap\n\n")
        f.write(f"**Date:** {summary['date']} · **Run ID:** `{args.run_id}` · **Seeds:** {args.num_seeds}\n")
        f.write(f"**Evaluation Set:** {summary['total_test_windows']} Held-Out Windows ({len(test_dataset)} test windows × {args.num_seeds} seeds)\n\n")
        f.write("## 1. Regret Across Optimization Checkpoints (Lower is Better)\n\n")
        headers = " | ".join([f"{cp} Steps" for cp in args.eval_checkpoints])
        separators = " | ".join([":---:" for _ in args.eval_checkpoints])
        f.write(f"| Policy | {headers} | Role / Mechanism |\n")
        f.write(f"|---|{separators}|---|\n")

        p_list = [
            "exact_costate",
            "always_mode0",
            "allocator_costate_norm_lcb_05",
            "allocator_costate_norm_lcb_10",
            "allocator_costate_norm",
            "allocator_costate",
            "allocator_critic_lcb_05",
            "allocator_critic_lcb_10",
            "allocator_critic",
            "uncertainty",
            "always_mode3",
            "random_expected",
        ]
        for p in p_list:
            cols = []
            for cp in args.eval_checkpoints:
                val = summary["checkpoints"][str(cp)]["mean_regrets"].get(p, float("nan"))
                cols.append(f"**{val:.5f}**")
            f.write(f"| `{p}` | {' | '.join(cols)} | — |\n")

        f.write("\n## 2. Key Findings\n\n")
        first_cp = str(args.eval_checkpoints[0])
        last_cp = str(args.eval_checkpoints[-1])
        cp_first_diff = summary["checkpoints"][first_cp]["diff_norm_minus_critic"]
        cp_last_diff = summary["checkpoints"][last_cp]["diff_norm_minus_critic"]
        f.write(f"- **Step {first_cp} Advantage (Norm vs Critic):** {cp_first_diff:.5f}\n")
        f.write(f"- **Step {last_cp} Advantage (Norm vs Critic):** {cp_last_diff:.5f}\n")

    print(f"Saved Markdown report to: {report_path}")


def write_seed_partial(output_dir: Path, seed: int, args, checkpoint_results: dict,
                       val_checkpoint_results: dict, val_panels_results: dict,
                       surrogate_ledger: list, finite: bool) -> Path:
    """Flush one seed's results so a destroyed session keeps completed seeds.

    Pure assembly over the caller's dicts (no torch); unit-tested. Partial files are
    evidence of completed seeds only — readiness is never decided from partials alone.
    """
    partial = {
        "seed": seed,
        "critic_width_scale": args.critic_width_scale,
        "critic_only": args.critic_only,
        "checkpoints": {str(cp): checkpoint_results[cp][-1] for cp in args.eval_checkpoints
                        if len(checkpoint_results[cp]) > 0},
        "val_checkpoints": {str(cp): val_checkpoint_results[cp][-1] for cp in args.eval_checkpoints
                            if len(val_checkpoint_results[cp]) > 0},
        "val_panels": {str(cp): val_panels_results[cp][-1] for cp in args.eval_checkpoints
                       if len(val_panels_results[cp]) > 0},
        "surrogate_ledger": [e for e in surrogate_ledger if e["seed"] == seed],
        "finite": finite,
    }
    partial_path = output_dir / f"partial_seed_{seed}.json"
    with open(partial_path, "w") as pf:
        json.dump(partial, pf, indent=1)
    return partial_path


if __name__ == "__main__":
    run_benchmark()
