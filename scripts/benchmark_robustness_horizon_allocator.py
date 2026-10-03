#!/usr/bin/env python3
"""scripts/benchmark_robustness_horizon_allocator.py

Session 5: Regret Metric Robustness, Cost-Model Sensitivity & Multi-Horizon Generalization.

Empirical evaluation of AdjointRWM vs direct critics on held-out multi-site robotics data:
1. Robust Statistical Panel (Mean, Median, 10% Trimmed Mean, Bootstrap 95% CI, Win Rates)
2. Leave-One-Site-Out (LOSO) Analysis across 12 robotics laboratories
3. Cost-Model Sensitivity Sweep across 5 distinct cost regimes (Zero, Uniform, Default, High-Penalty, Latency-Weighted)
4. Multi-Horizon Generalization across H in {2, 4} with dynamic horizon conditioning
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_DIR = Path(__file__).resolve().parent.parent
for p in [REPO_DIR / "src", Path("/content/para_001/src"), Path("/content/src")]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from adjointrwm.allocators import (  # noqa: E402
    CostateEstimator,
    CurvatureCostateEstimator,
    DirectCritic,
    belief_space_voi_scores,
    first_order_scores,
    matched_critic_hidden,
    normalized_first_order_scores,
    plackett_luce_loss,
    second_order_curvature_scores,
    with_hold_zero,
)
from adjointrwm.data import WindowDataset, WindowSpec, fit_normaliser  # noqa: E402
from adjointrwm.models.common import ArmDims, prediction_objective  # noqa: E402
from adjointrwm.models.hybrid_adjoint import HybridAdjointRecursiveWorldModel  # noqa: E402

try:
    from adjointrwm.models import build_arm  # noqa: E402
except Exception:  # pragma: no cover
    build_arm = None

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


def curvature_matched_critic_hidden(d: int) -> int:
    """Critic hidden width whose parameter count matches CurvatureCostateEstimator."""
    target = sum(p.numel() for p in CurvatureCostateEstimator(d).parameters())
    return max(1, round((target - 1) / (2 * d + 5)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Session 5: Robustness, Cost Sensitivity & Horizon Generalization")
    parser.add_argument("--synthetic-test", action="store_true", help="Run fast synthetic smoke test on CPU")
    parser.add_argument("--cache-dir", type=str, default="/content/cache_e3_1")
    parser.add_argument("--drive-root", type=str, default="/content/drive/MyDrive/Colab Notebooks/AdjointRWM_Production")
    parser.add_argument("--run-id", type=str, default="harp_hybrid_dynamics_20261002")
    parser.add_argument("--output-dir", type=str, default="results/benchmarks/robustness_horizon")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-target-windows", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--cost-weight", type=float, default=0.002)
    parser.add_argument("--uncert-weight", type=float, default=0.5)
    return parser.parse_args(argv)


def load_e3_dataset(cache_dir: Path):
    cache_manifest_path = cache_dir / "cache_manifest.json"
    if not cache_manifest_path.exists():
        cache_manifest_path = cache_dir / "e3_1_droid_500_manifest.json"
    if not cache_manifest_path.exists():
        fallback = REPO_DIR / "results/data/droid_e3_1/e3_1_droid_500_manifest.json"
        if fallback.exists():
            cache_manifest_path = fallback
        else:
            raise FileNotFoundError(f"Missing cache manifest at {cache_manifest_path}")

    manifest_data = json.loads(cache_manifest_path.read_text())
    episodes = manifest_data.get("episodes", [])

    records_by_split = {"train": [], "val": [], "test": []}
    episodes_dir = cache_dir / "episodes"
    site_by_episode = {}
    for ep in episodes:
        raw_split = ep.get("split", "train")
        split = "val" if raw_split in ("val", "validation") else raw_split
        ep_id = ep["episode_id"]
        site_by_episode[ep_id] = ep.get("site", "unknown")
        ep_path = episodes_dir / f"{ep_id}.npz"
        if ep_path.exists():
            records_by_split[split].append({
                "episode_id": ep_id,
                "cached_path": str(ep_path),
                "length": ep.get("length", 100),
                "split": split,
                "site": ep.get("site", "unknown"),
            })

    norm_path = cache_dir / "normalisation.npz"
    if norm_path.exists():
        with np.load(norm_path) as npz:
            normalisation = {k: npz[k] for k in npz.files}
    else:
        def load_arrays(rec):
            with np.load(rec["cached_path"]) as ep:
                return {"states": ep["states"], "actions": ep["actions"]}
        train_arrays = [load_arrays(r) for r in records_by_split["train"]]
        normalisation = fit_normaliser([a["states"] for a in train_arrays], [a["actions"] for a in train_arrays])

    spec = WindowSpec(context_len=8, horizon=4, stride=2)
    train_dataset = WindowDataset(records_by_split["train"], spec, normalisation, visual_layout="flat")
    val_dataset = WindowDataset(records_by_split["val"], spec, normalisation, visual_layout="flat")
    test_dataset = WindowDataset(records_by_split["test"], spec, normalisation, visual_layout="flat")

    return train_dataset, val_dataset, test_dataset, site_by_episode


def extract_real_targets(
    model: HybridAdjointRecursiveWorldModel,
    batch: dict,
    costs: torch.Tensor,
    device: torch.device,
    horizon_val: int = 4,
) -> dict:
    b_dev = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
    B = b_dev["context_state"].shape[0]

    with torch.no_grad():
        v0 = torch.zeros_like(b_dev["context_visual"])
        z0 = model.encode_context(v0, b_dev["context_state"], b_dev["context_action"])

        v1 = b_dev["context_visual"].clone()
        v1[:, :, :512] = 0.0
        z1 = model.encode_context(v1, b_dev["context_state"], b_dev["context_action"])

        v2 = b_dev["context_visual"].clone()
        v2[:, :, 512:] = 0.0
        z2 = model.encode_context(v2, b_dev["context_state"], b_dev["context_action"])

        v3 = b_dev["context_visual"]
        z3 = model.encode_context(v3, b_dev["context_state"], b_dev["context_action"])

    future_acts = b_dev["future_actions"][:, :horizon_val]
    tgt_state = b_dev["target_state"][:, :horizon_val]
    tgt_vis = b_dev["target_visual"][:, :horizon_val]

    # Evaluate prediction objective for each candidate
    with torch.no_grad():
        p0 = model.rollout(z0, future_acts, context_state=b_dev["context_state"])
        j0 = prediction_objective(p0, tgt_state, tgt_vis)

        p1 = model.rollout(z1, future_acts, context_state=b_dev["context_state"])
        j1 = prediction_objective(p1, tgt_state, tgt_vis)

        p2 = model.rollout(z2, future_acts, context_state=b_dev["context_state"])
        j2 = prediction_objective(p2, tgt_state, tgt_vis)

        p3 = model.rollout(z3, future_acts, context_state=b_dev["context_state"])
        j3 = prediction_objective(p3, tgt_state, tgt_vis)

    # Autograd co-state at Mode 0 (create_graph=True for higher-order HVPs)
    z0_req = z0.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        p0_req = model.rollout(z0_req, future_acts, context_state=b_dev["context_state"])
        loss_0 = prediction_objective(p0_req, tgt_state, tgt_vis).sum()
        lambda0 = torch.autograd.grad(loss_0, z0_req, create_graph=True)[0]

        delta_z1 = (z1 - z0).detach()
        delta_z2 = (z2 - z0).detach()
        delta_z3 = (z3 - z0).detach()
        active_deltas = [delta_z1, delta_z2, delta_z3]

        exact_curvatures = []
        for dz_k in active_deltas:
            inner_k = torch.sum(lambda0 * dz_k)
            hvp_k = torch.autograd.grad(inner_k, z0_req, retain_graph=True)[0]
            curv_k = torch.sum(hvp_k * dz_k, dim=-1).detach()
            exact_curvatures.append(curv_k)
        exact_curv_tensor = torch.stack(exact_curvatures, dim=1)

    delta_z0 = torch.zeros_like(delta_z1)
    effects = torch.stack([delta_z0, delta_z1, delta_z2, delta_z3], dim=1)

    g0 = torch.zeros_like(j0)
    g1 = (j0 - j1) - costs[1]
    g2 = (j0 - j2) - costs[2]
    g3 = (j0 - j3) - costs[3]
    exact_gain = torch.stack([g0, g1, g2, g3], dim=1)

    return {
        "latent": z0.detach(),
        "effects": effects,
        "costs": costs,
        "exact_costate": lambda0.detach(),
        "exact_curvatures": exact_curv_tensor,
        "exact_gain": exact_gain,
        "context_visual": b_dev["context_visual"].detach(),
    }


def train_allocator_heads(
    teacher,
    train_loader,
    device: torch.device,
    args,
    costs: torch.Tensor,
    horizon_val: int = 4,
) -> dict:
    d = teacher.d_model
    torch.manual_seed(42)

    costate = CostateEstimator(d).to(device)
    h_crit = matched_critic_hidden(d)
    critic = DirectCritic(d, h_crit).to(device)

    h_curv_crit = curvature_matched_critic_hidden(d)
    critic_curv = DirectCritic(d, h_curv_crit).to(device)

    curvature = CurvatureCostateEstimator(d).to(device)

    opt = torch.optim.AdamW(
        list(costate.parameters())
        + list(critic.parameters())
        + list(critic_curv.parameters())
        + list(curvature.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    train_iter = iter(train_loader)
    for step in range(args.max_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        tgt = extract_real_targets(teacher, batch, costs, device, horizon_val=horizon_val)
        latent, effects, costs_t = tgt["latent"], tgt["effects"], tgt["costs"]
        B = latent.shape[0]
        budget = torch.full((B,), 1.0 / 3.0, device=device)
        horizon = torch.full((B,), float(horizon_val), device=device)

        opt.zero_grad()

        lam_hat = costate(latent, budget, horizon)
        cos_loss = (1.0 - F.cosine_similarity(lam_hat, tgt["exact_costate"], dim=-1)).mean()

        s_crit = with_hold_zero(critic(latent, effects[:, 1:], costs_t[1:], budget, horizon))
        crit_loss = F.smooth_l1_loss(s_crit[:, 1:], tgt["exact_gain"][:, 1:])

        s_crit_curv = with_hold_zero(critic_curv(latent, effects[:, 1:], costs_t[1:], budget, horizon))
        crit_curv_loss = F.smooth_l1_loss(s_crit_curv[:, 1:], tgt["exact_gain"][:, 1:])

        lam2, h2 = curvature(latent, budget, horizon)
        cos2 = (1.0 - F.cosine_similarity(lam2, tgt["exact_costate"], dim=-1)).mean()

        pred_curvs = []
        for k_idx in range(3):
            dz = effects[:, k_idx + 1]
            pred_curv_k = (h2 * dz.pow(2)).sum(dim=-1)
            pred_curvs.append(pred_curv_k)
        pred_curv_tensor = torch.stack(pred_curvs, dim=1)
        curv_loss = F.smooth_l1_loss(pred_curv_tensor, tgt["exact_curvatures"])

        s2 = second_order_curvature_scores(lam2, h2, effects, costs_t)
        rank2 = plackett_luce_loss(s2, tgt["exact_gain"], temperature=0.1)

        total_loss = cos_loss + crit_loss + crit_curv_loss + cos2 + 0.5 * curv_loss + 0.5 * rank2
        total_loss.backward()
        opt.step()

    return {
        "costate": costate.eval(),
        "critic": critic.eval(),
        "critic_curv_matched": critic_curv.eval(),
        "curvature": curvature.eval(),
    }


@torch.no_grad()
def score_all_policies(heads: dict, targets: dict, horizon_val: int = 4, uncert_weight: float = 0.5) -> dict:
    latent, effects, costs = targets["latent"], targets["effects"], targets["costs"]
    B = latent.shape[0]
    device = latent.device

    budget = torch.full((B,), 1.0 / 3.0, device=device)
    horizon = torch.full((B,), float(horizon_val), device=device)

    lam_hat = heads["costate"](latent, budget, horizon)
    lam_c, h_c = heads["curvature"](latent, budget, horizon)

    s_critic = with_hold_zero(heads["critic"](latent, effects[:, 1:], costs[1:], budget, horizon))
    s_critic_curv = with_hold_zero(heads["critic_curv_matched"](latent, effects[:, 1:], costs[1:], budget, horizon))

    delta_cov = effects.pow(2)

    return {
        "exact_costate": first_order_scores(targets["exact_costate"], effects, costs),
        "direct_critic": s_critic,
        "direct_critic_curv_matched": s_critic_curv,
        "first_order": first_order_scores(lam_hat, effects, costs),
        "normalized_first_order": normalized_first_order_scores(lam_hat, effects, costs),
        "second_order_curvature": second_order_curvature_scores(lam_c, h_c, effects, costs),
        "belief_space_voi": belief_space_voi_scores(lam_c, effects, delta_cov, h_c, costs, uncert_weight=uncert_weight),
    }


def compute_bootstrap_ci(data: np.ndarray, num_resamples: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    rng = np.random.default_rng(42)
    resamples = rng.choice(data, size=(num_resamples, len(data)), replace=True)
    means = resamples.mean(axis=1)
    low = float(np.percentile(means, 100 * (alpha / 2)))
    high = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return low, high


def evaluate_benchmark(
    teacher,
    heads: dict,
    loader: DataLoader,
    device: torch.device,
    costs: torch.Tensor,
    args,
    horizon_val: int = 4,
    site_by_episode: dict | None = None,
) -> dict:
    per_policy_gains = {p: [] for p in POLICIES}
    per_policy_choices = {p: [] for p in POLICIES}
    exact_gains_list = []
    episode_ids_list = []

    lat_total, lat_count = 0.0, 0
    n_windows = 0

    for batch in loader:
        targets = extract_real_targets(teacher, batch, costs, device, horizon_val=horizon_val)
        gain = targets["exact_gain"]
        B = gain.shape[0]
        n_windows += B

        t0 = time.perf_counter()
        with torch.no_grad():
            scores = score_all_policies(heads, targets, horizon_val=horizon_val, uncert_weight=args.uncert_weight)
        if device.type == "cuda":
            torch.cuda.synchronize()
        lat_total += time.perf_counter() - t0
        lat_count += 1

        exact_gains_list.append(gain.cpu().numpy())
        if "episode_id" in batch:
            episode_ids_list.extend(batch["episode_id"])

        for name in POLICIES:
            if name == "always_mode0":
                ch = torch.zeros(B, dtype=torch.long, device=gain.device)
            else:
                ch = scores[name].argmax(dim=1)
            ch_cpu = ch.cpu().numpy()
            g_cpu = gain.cpu().numpy()[np.arange(B), ch_cpu]
            per_policy_choices[name].append(ch_cpu)
            per_policy_gains[name].append(g_cpu)

    gains = {p: np.concatenate(v) for p, v in per_policy_gains.items()}
    all_exact_gains = np.concatenate(exact_gains_list, axis=0)  # [N, 4]
    true_oracle_best = all_exact_gains.max(axis=1)             # [N]

    mean_lat_ms = (lat_total / max(1, lat_count)) / args.batch_size * 1000.0
    throughput_hz = (n_windows / lat_total) if lat_total > 0 else 0.0

    summary = {}
    per_policy_regret = {}
    for p in POLICIES:
        reg = true_oracle_best - gains[p]
        per_policy_regret[p] = reg
        ci_low, ci_high = compute_bootstrap_ci(reg)
        summary[p] = {
            "mean_gain": float(gains[p].mean()),
            "mean_regret": float(reg.mean()),
            "median_regret": float(np.median(reg)),
            "trimmed_mean_regret_10pct": float(np.mean(np.sort(reg)[int(0.1*len(reg)):int(0.9*len(reg))])),
            "std_regret": float(reg.std()),
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "win_rate_vs_critic": float((gains[p] > gains["direct_critic"]).mean()),
            "win_rate_vs_curv_critic": float((gains[p] > gains["direct_critic_curv_matched"]).mean()),
            "win_rate_vs_mode0": float((gains[p] > gains["always_mode0"]).mean()),
            "win_rate_vs_first_order": float((gains[p] > gains["first_order"]).mean()),
            "n_windows": int(n_windows),
        }

    # Per-site breakdown and Leave-One-Site-Out (LOSO)
    site_summary = {}
    loso_summary = {}
    if site_by_episode and len(episode_ids_list) == n_windows:
        site_regrets = {}
        for idx, eid in enumerate(episode_ids_list):
            s = site_by_episode.get(eid, "unknown")
            if s not in site_regrets:
                site_regrets[s] = {p: [] for p in POLICIES}
            for p in POLICIES:
                site_regrets[s][p].append(per_policy_regret[p][idx])

        for s, p_dict in sorted(site_regrets.items()):
            site_summary[s] = {
                "n_windows": len(p_dict["always_mode0"]),
                "regret": {p: float(np.mean(p_dict[p])) for p in POLICIES},
                "median_regret": {p: float(np.median(p_dict[p])) for p in POLICIES},
            }

        all_sites = list(site_regrets.keys())
        for dropped_site in all_sites:
            rem_critic_reg = []
            rem_voi_reg = []
            for s in all_sites:
                if s != dropped_site:
                    rem_critic_reg.extend(site_regrets[s]["direct_critic"])
                    rem_voi_reg.extend(site_regrets[s]["belief_space_voi"])
            c_m = np.mean(rem_critic_reg)
            v_m = np.mean(rem_voi_reg)
            adv = float((c_m - v_m) / max(1e-8, c_m) * 100.0)
            loso_summary[dropped_site] = {
                "direct_critic_regret": float(c_m),
                "belief_space_voi_regret": float(v_m),
                "voi_advantage_pct": adv,
            }

    return {
        "per_policy": summary,
        "site_breakdown": site_summary,
        "loso_summary": loso_summary,
        "n_windows": int(n_windows),
        "mean_latency_ms_per_window": float(mean_lat_ms),
        "throughput_hz": float(throughput_hz),
    }


def make_synthetic_batch(batch_size: int, device: torch.device, seed: int) -> dict:
    g = torch.Generator(device="cpu").manual_seed(1000 + seed)
    mk = lambda *shape: torch.randn(*shape, generator=g).to(device)
    return {
        "context_state": mk(batch_size, 8, 14),
        "context_action": mk(batch_size, 8, 7),
        "context_visual": mk(batch_size, 8, 1024),
        "future_actions": mk(batch_size, 4, 7),
        "target_state": mk(batch_size, 4, 14),
        "target_visual": mk(batch_size, 4, 1024),
    }


def build_synthetic_teacher(device: torch.device):
    dims = ArmDims(
        state_dim=14,
        action_dim=7,
        visual_tokens=2,
        visual_token_dim=512,
        target_visual_dim=1024,
        context_len=8,
        horizon=4,
    )
    teacher = build_arm("hybrid_adjoint_rwm", dims, width=64, transformer_heads=4, transformer_layers=2)
    return teacher.to(device).eval()


def run_synthetic(args, device: torch.device) -> dict:
    print("[synthetic] running smoke test on synthetic batch...")
    teacher = build_synthetic_teacher(device)
    costs = torch.tensor([0.0, 0.001, 0.001, 0.002], device=device)

    synthetic_train = [make_synthetic_batch(args.batch_size, device, s) for s in range(5)]
    synthetic_test = [make_synthetic_batch(args.batch_size, device, 100 + s) for s in range(4)]

    fake_site_map = {f"ep_{i}": f"lab_{i % 3}" for i in range(len(synthetic_test) * args.batch_size)}
    for b_idx, b in enumerate(synthetic_test):
        b["episode_id"] = [f"ep_{b_idx * args.batch_size + i}" for i in range(args.batch_size)]

    heads = train_allocator_heads(teacher, synthetic_train, device, args, costs, horizon_val=4)
    ev = evaluate_benchmark(teacher, heads, synthetic_test, device, costs, args, horizon_val=4, site_by_episode=fake_site_map)
    return {
        "mode": "synthetic",
        "device": str(device),
        "horizon_eval": {4: ev},
        "cost_sweep_eval": {"default": ev},
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.synthetic_test else "cpu")
    print(f"[Session 5] Running Robustness, Cost Sensitivity & Multi-Horizon Benchmark on {device}...")

    if args.synthetic_test or not Path(args.cache_dir).exists():
        if not args.synthetic_test and not Path(args.cache_dir).exists():
            print(f"[warning] cache_dir {args.cache_dir} not found, falling back to synthetic test.")
        res = run_synthetic(args, device)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "robustness_horizon_summary.json", "w") as f:
            json.dump(res, f, indent=2)
        print("[Session 5] Synthetic smoke test complete.")
        return

    cache_dir = Path(args.cache_dir)
    drive_root = Path(args.drive_root)
    run_dir = drive_root / "runs" / args.run_id

    train_dataset, val_dataset, test_dataset, site_by_episode = load_e3_dataset(cache_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    dims = ArmDims(
        state_dim=14,
        action_dim=7,
        visual_tokens=2,
        visual_token_dim=512,
        target_visual_dim=1024,
        context_len=8,
        horizon=4,
    )

    cost_regimes = {
        "zero": torch.tensor([0.0, 0.0, 0.0, 0.0], device=device),
        "uniform": torch.tensor([0.0, 1.0, 1.0, 1.0], device=device) * args.cost_weight,
        "default": torch.tensor([0.0, 1.0, 1.0, 1.5, 2.0], device=device) * args.cost_weight,
        "high_penalty": torch.tensor([0.0, 1.0, 1.0, 3.0, 5.0], device=device) * args.cost_weight,
        "latency_weighted": torch.tensor([0.0, 0.8, 0.8, 1.8], device=device) * args.cost_weight,
    }

    horizons = [2, 4]

    seed_results = []
    for seed in range(args.num_seeds):
        print(f"\n=======================================================")
        print(f"  SEED {seed + 1} / {args.num_seeds} EVALUATION")
        print(f"=======================================================")
        torch.manual_seed(6000 + seed * 100)

        teacher = build_arm("hybrid_adjoint_rwm", dims, width=512, transformer_heads=8, transformer_layers=6).to(device)
        model_path = run_dir / f"seed_{seed}" / "best.pt"
        if not model_path.exists():
            model_path = run_dir / "best.pt"
        if not model_path.exists():
            model_path = run_dir / "seed_0" / "best.pt"
        if model_path.exists():
            print(f"  Loading trained teacher weights from {model_path}...")
            sd = torch.load(model_path, map_location=device, weights_only=True)
            teacher.load_state_dict(sd.get("model_state_dict", sd), strict=False)
        else:
            print(f"  [warning] Checkpoint not found at {model_path}; using initialized teacher.")
        teacher.eval()

        # 1. Cost sensitivity sweep at H=4
        cost_evals = {}
        for regime_name, c_vec in cost_regimes.items():
            print(f"  [Seed {seed}] Training & evaluating cost regime: {regime_name}...")
            heads = train_allocator_heads(teacher, train_loader, device, args, c_vec, horizon_val=4)
            ev = evaluate_benchmark(teacher, heads, test_loader, device, c_vec, args, horizon_val=4, site_by_episode=site_by_episode)
            cost_evals[regime_name] = ev

        # 2. Horizon sweep under default cost
        default_costs = cost_regimes["default"]
        horizon_evals = {}
        for h_val in horizons:
            print(f"  [Seed {seed}] Training & evaluating horizon H={h_val}...")
            heads_h = train_allocator_heads(teacher, train_loader, device, args, default_costs, horizon_val=h_val)
            ev_h = evaluate_benchmark(teacher, heads_h, test_loader, device, default_costs, args, horizon_val=h_val, site_by_episode=site_by_episode)
            horizon_evals[h_val] = ev_h

        seed_results.append({
            "seed": seed,
            "cost_regimes": cost_evals,
            "horizons": horizon_evals,
        })

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_file = out_dir / "robustness_horizon_summary.json"
    with open(summary_file, "w") as f:
        json.dump(seed_results, f, indent=2)

    # Markdown Report Generation
    report_file = out_dir / "robustness_horizon_report.md"
    rep = [
        "# Session 5: Robustness, Cost-Model Sensitivity & Multi-Horizon Report",
        "",
        f"**Date:** 2026-10-03  ",
        f"**Device:** {device}  ",
        f"**Num Seeds:** {args.num_seeds}  ",
        f"**Test Windows Evaluated:** {len(test_dataset) * args.num_seeds}  ",
        "",
        "## 1. Cost-Model Sensitivity Sweep (H=4)",
        "",
        "| Cost Regime | Baseline Critic Regret | Belief-Space VOI Regret | VOI Advantage (%) |",
        "|---|---|---|---|",
    ]

    for regime in cost_regimes.keys():
        c_regs = [s["cost_regimes"][regime]["per_policy"]["direct_critic"]["mean_regret"] for s in seed_results]
        v_regs = [s["cost_regimes"][regime]["per_policy"]["belief_space_voi"]["mean_regret"] for s in seed_results]
        c_mean, v_mean = float(np.mean(c_regs)), float(np.mean(v_regs))
        adv = (c_mean - v_mean) / max(1e-8, c_mean) * 100.0
        rep.append(f"| `{regime}` | {c_mean:.5f} | **{v_mean:.5f}** | **{adv:+.2f}%** |")

    rep.extend([
        "",
        "## 2. Multi-Horizon Generalization (Default Cost)",
        "",
        "| Horizon H | Baseline Critic Regret | Belief-Space VOI Regret | VOI Advantage (%) |",
        "|---|---|---|---|",
    ])

    for h_val in horizons:
        c_regs = [s["horizons"][h_val]["per_policy"]["direct_critic"]["mean_regret"] for s in seed_results]
        v_regs = [s["horizons"][h_val]["per_policy"]["belief_space_voi"]["mean_regret"] for s in seed_results]
        c_mean, v_mean = float(np.mean(c_regs)), float(np.mean(v_regs))
        adv = (c_mean - v_mean) / max(1e-8, c_mean) * 100.0
        rep.append(f"| H={h_val} | {c_mean:.5f} | **{v_mean:.5f}** | **{adv:+.2f}%** |")

    rep.extend([
        "",
        "## 3. Leave-One-Site-Out (LOSO) Regret Deltas (Default Cost, H=4)",
        "",
        "| Dropped Site | Remaining Critic Regret | Remaining VOI Regret | VOI Advantage (%) |",
        "|---|---|---|---|",
    ])

    # Aggregate LOSO from seed 0
    loso = seed_results[0]["cost_regimes"]["default"].get("loso_summary", {})
    for s_name, data in sorted(loso.items()):
        rep.append(f"| `{s_name}` | {data['direct_critic_regret']:.5f} | {data['belief_space_voi_regret']:.5f} | {data['voi_advantage_pct']:+.2f}% |")

    with open(report_file, "w") as f:
        f.write("\n".join(rep) + "\n")

    print(f"\n[Session 5] Results saved to {summary_file} and {report_file}.")


if __name__ == "__main__":
    main()
