#!/usr/bin/env python3
"""scripts/benchmark_curvature_voi_allocator.py

Direction 3: Curvature and Belief-Space VOI Allocation Benchmark.

Compares allocator heads on exact per-window gains with a hold option:

* ``exact_costate`` (Oracle floor: first-order scores from autograd co-state)
* ``always_mode0`` (Refusal baseline: always hold, gain 0 by definition)
* ``direct_critic`` (Parameter-matched DirectCritic)
* ``first_order`` (CostateEstimator + first_order_scores)
* ``normalized_first_order`` (CostateEstimator + normalized_first_order_scores)
* ``second_order_curvature`` (CurvatureCostateEstimator + second_order_curvature_scores)
* ``belief_space_voi`` (CurvatureCostateEstimator + belief_space_voi_scores)

Metrics per policy: mean test regret, gap to oracle, win rate vs critic and
vs mode0, and mean inference latency per window.

Runs a fast deterministic synthetic diagnostic when invoked with
``--synthetic-test`` or when the real ``run_dir`` is absent. With a real run
directory present (and ``--synthetic-test`` not given), trains the three head
types briefly on real windows and evaluates on held-out test windows.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_DIR = Path(__file__).resolve().parent.parent
for p in [REPO_DIR / "src", Path("/content/para_001/src"), Path("/content/src")]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from adjointrwm.allocators import (  # noqa: E402
    AllocatorJob,
    CostateEstimator,
    CurvatureCostateEstimator,
    DirectCritic,
    belief_space_voi_scores,
    exact_targets,
    first_order_scores,
    matched_critic_hidden,
    normalized_first_order_scores,
    second_order_curvature_scores,
    with_hold_zero,
)
from adjointrwm.models.common import ArmDims  # noqa: E402

try:
    from adjointrwm.models import build_arm  # noqa: E402
except Exception:  # pragma: no cover
    build_arm = None

POLICIES = (
    "exact_costate",
    "always_mode0",
    "direct_critic",
    "first_order",
    "normalized_first_order",
    "second_order_curvature",
    "belief_space_voi",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Direction 3: Curvature and Belief-Space VOI Allocation Benchmark")
    parser.add_argument("--synthetic-test", action="store_true", help="Run quick synthetic diagnostic on CPU")
    parser.add_argument("--run-id", type=str, default="droid100_adjoint_v2_5seeds_20261001T080821Z")
    parser.add_argument("--drive-root", type=str, default="/content/drive/MyDrive/Colab Notebooks/AdjointRWM_Production")
    parser.add_argument("--local-cache", type=str, default="/content/cache")
    parser.add_argument("--output-dir", type=str, default="results/benchmarks/curvature_voi")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--cost-weight", type=float, default=0.002)
    parser.add_argument("--num-batches", type=int, default=8, help="Synthetic test batches for evaluation")
    parser.add_argument("--uncert-weight", type=float, default=0.5)
    return parser.parse_args(argv)


def make_synthetic_batch(batch_size: int, device: torch.device, seed: int) -> dict:
    g = torch.Generator(device="cpu").manual_seed(1000 + seed)
    mk = lambda *shape: torch.randn(*shape, generator=g).to(device)
    return {
        "context_state": mk(batch_size, 8, 10),
        "context_action": mk(batch_size, 8, 7),
        "context_visual": mk(batch_size, 8, 32),
        "future_actions": mk(batch_size, 4, 7),
        "target_state": mk(batch_size, 4, 10),
        "target_visual": mk(batch_size, 4, 32),
    }


def build_synthetic_teacher(device: torch.device):
    if build_arm is None:
        raise RuntimeError("build_arm unavailable; cannot run synthetic teacher")
    dims = ArmDims(state_dim=10, action_dim=7, visual_tokens=4, visual_token_dim=8,
                   target_visual_dim=32, context_len=8, horizon=4)
    teacher = build_arm("adjoint_rwm", dims, width=64, transformer_heads=4, transformer_layers=2)
    return teacher.to(device).eval()


def train_heads_synthetic(teacher, device: torch.device, args) -> dict:
    """Briefly train the three head types on synthetic batches so policies differ from init."""
    d = teacher.d_model
    torch.manual_seed(7)
    costate = CostateEstimator(d).to(device)
    critic = DirectCritic(d, matched_critic_hidden(d)).to(device)
    curvature = CurvatureCostateEstimator(d).to(device)
    opt = torch.optim.AdamW(
        list(costate.parameters()) + list(critic.parameters()) + list(curvature.parameters()), lr=args.lr
    )
    for step in range(args.max_steps):
        batch = make_synthetic_batch(args.batch_size, device, seed=step)
        tgt = exact_targets(teacher, batch, args.cost_weight)
        latent, effects, costs = tgt["latent"], tgt["effects"], tgt["costs"]
        bsz = latent.shape[0]
        budget = torch.full((bsz,), 1.0 / teacher.num_candidates, device=device)
        horizon = torch.ones(bsz, device=device)
        opt.zero_grad()
        lam_hat = costate(latent, budget, horizon)
        cos_loss = (1.0 - F.cosine_similarity(lam_hat, tgt["exact_costate"], dim=-1)).mean()
        s_critic = with_hold_zero(critic(latent, effects[:, 1:], costs[1:], budget, horizon))
        critic_loss = F.smooth_l1_loss(s_critic[:, 1:], tgt["exact_gain"][:, 1:])
        lam2, h2 = curvature(latent, budget, horizon)
        cos2 = (1.0 - F.cosine_similarity(lam2, tgt["exact_costate"], dim=-1)).mean()
        s2 = second_order_curvature_scores(lam2, h2, effects, costs)
        rank2 = F.cross_entropy(s2, tgt["exact_gain"].argmax(dim=1))
        loss = cos_loss + critic_loss + cos2 + 0.5 * rank2
        loss.backward()
        opt.step()
    return {"costate": costate.eval(), "critic": critic.eval(), "curvature": curvature.eval()}


@torch.no_grad()
def score_all_policies(heads: dict, targets: dict, uncert_weight: float = 0.5) -> dict:
    latent, effects, costs = targets["latent"], targets["effects"], targets["costs"]
    bsz = latent.shape[0]
    device = latent.device
    # Budget/horizon conditioning matches AllocatorJob._conditions.
    n_cand = effects.shape[1] - 1
    budget = torch.full((bsz,), 1.0 / max(1, n_cand), device=device)
    horizon = torch.ones(bsz, device=device)
    lam_hat = heads["costate"](latent, budget, horizon)
    lam_c, h_c = heads["curvature"](latent, budget, horizon)
    s_critic = with_hold_zero(heads["critic"](latent, effects[:, 1:], costs[1:], budget, horizon))
    # Belief-space VOI needs a per-candidate covariance reduction proxy. With no
    # sensing model in this benchmark, use the squared-effect magnitude as the
    # documented proxy: delta_cov = effects^2 (hold row is exactly 0).
    delta_cov = effects.pow(2)
    return {
        "exact_costate": first_order_scores(targets["exact_costate"], effects, costs),
        "direct_critic": s_critic,
        "first_order": first_order_scores(lam_hat, effects, costs),
        "normalized_first_order": normalized_first_order_scores(lam_hat, effects, costs),
        "second_order_curvature": second_order_curvature_scores(lam_c, h_c, effects, costs),
        "belief_space_voi": belief_space_voi_scores(lam_c, effects, delta_cov, h_c, costs, uncert_weight=uncert_weight),
    }


def evaluate_gains(teacher, heads: dict, device: torch.device, args) -> dict:
    """Collect per-window gains/choices for every policy plus scoring latency."""
    per_policy_gains = {p: [] for p in POLICIES}
    per_policy_choices = {p: [] for p in POLICIES}
    lat_total, lat_count = 0.0, 0
    n_windows = 0
    for b in range(args.num_batches):
        batch = make_synthetic_batch(args.batch_size, device, seed=5000 + b)
        targets = exact_targets(teacher, batch, args.cost_weight)
        gain = targets["exact_gain"]
        n, k1 = gain.shape
        n_windows += n
        t0 = time.perf_counter()
        with torch.no_grad():
            scores = score_all_policies(heads, targets, uncert_weight=args.uncert_weight)
        torch.cuda.synchronize() if device.type == "cuda" else None
        lat_total += time.perf_counter() - t0
        lat_count += 1
        oracle_choice = gain.argmax(dim=1)
        for name in POLICIES:
            if name == "always_mode0":
                choice = torch.zeros(n, dtype=torch.long, device=gain.device)
            elif name in scores:
                choice = scores[name].argmax(dim=1).cpu()
                oracle_cpu = oracle_choice.cpu()
                per_policy_choices[name].append(choice.numpy())
                per_policy_gains[name].append(gain.cpu().numpy()[np.arange(n), choice.numpy()])
                continue
            else:  # pragma: no cover
                continue
            oracle_cpu = oracle_choice.cpu()
            ch = choice.cpu().numpy() if torch.is_tensor(choice) else np.asarray(choice)
            per_policy_choices[name].append(ch)
            per_policy_gains[name].append(gain.cpu().numpy()[np.arange(n), ch])
    gains = {p: np.concatenate(v) for p, v in per_policy_gains.items()}
    oracle_gain = gains["exact_costate"]  # oracle picks argmax exact gain by construction
    # Oracle regret is ~0; compute exactly from max for clarity.
    all_gains_stack = np.stack([gains[p] for p in POLICIES], axis=1)
    best = all_gains_stack.max(axis=1)
    mean_latency_ms = (lat_total / max(1, lat_count)) / args.batch_size * 1000.0
    summary = {}
    for p in POLICIES:
        regret = best - gains[p]
        summary[p] = {
            "mean_gain": float(gains[p].mean()),
            "mean_regret": float(regret.mean()),
            "std_regret": float(regret.std()),
            "gap_to_oracle": float(regret.mean() - (best - oracle_gain).mean()),
            "win_rate_vs_critic": float((gains[p] > gains["direct_critic"]).mean()),
            "win_rate_vs_mode0": float((gains[p] > gains["always_mode0"]).mean()),
            "tie_rate_vs_critic": float((gains[p] == gains["direct_critic"]).mean()),
            "n_windows": int(n_windows),
        }
    return {"per_policy": summary, "n_windows": int(n_windows),
            "mean_latency_ms_per_window": float(mean_latency_ms)}


def run_synthetic(args, device: torch.device) -> dict:
    print("[synthetic] building synthetic teacher...")
    teacher = build_synthetic_teacher(device)
    print(f"[synthetic] teacher d_model={teacher.d_model}, candidates={teacher.num_candidates}")
    # Smoke-test AllocatorJob loss paths (guards against API drift).
    batch = make_synthetic_batch(args.batch_size, device, seed=42)
    for head_type in ("costate", "critic"):
        job = AllocatorJob(teacher, head_type, cost_weight=args.cost_weight).to(device)
        opt = torch.optim.AdamW(job.parameters(), lr=args.lr)
        opt.zero_grad()
        loss, parts = job.training_loss(batch)
        loss.backward()
        opt.step()
        print(f"[synthetic] AllocatorJob({head_type}): loss={loss.item():.4f}")
    heads = train_heads_synthetic(teacher, device, args)
    results = []
    for seed in range(max(1, args.num_seeds)):
        torch.manual_seed(9000 + seed)
        ev = evaluate_gains(teacher, heads, device, args)
        results.append(ev)
        print(f"[synthetic] seed {seed}: " + ", ".join(
            f"{p} regret={ev['per_policy'][p]['mean_regret']:.4f}" for p in POLICIES))
    # Average across seeds.
    per_policy = {}
    for p in POLICIES:
        per_policy[p] = {
            "mean_regret": float(np.mean([r["per_policy"][p]["mean_regret"] for r in results])),
            "std_regret": float(np.mean([r["per_policy"][p]["std_regret"] for r in results])),
            "mean_gain": float(np.mean([r["per_policy"][p]["mean_gain"] for r in results])),
            "gap_to_oracle": float(np.mean([r["per_policy"][p]["gap_to_oracle"] for r in results])),
            "win_rate_vs_critic": float(np.mean([r["per_policy"][p]["win_rate_vs_critic"] for r in results])),
            "win_rate_vs_mode0": float(np.mean([r["per_policy"][p]["win_rate_vs_mode0"] for r in results])),
        }
    return {
        "mode": "synthetic",
        "device": str(device),
        "num_batches": args.num_batches,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "num_seeds": max(1, args.num_seeds),
        "cost_weight": args.cost_weight,
        "n_windows": int(results[0]["n_windows"]),
        "mean_latency_ms_per_window": float(np.mean([r["mean_latency_ms_per_window"] for r in results])),
        "per_policy": per_policy,
    }


def try_real_benchmark(args, device: torch.device):
    """Attempt the real-data path; return None when run_dir/cache is absent."""
    run_dir = Path(args.drive_root) / "runs" / args.run_id
    if not run_dir.exists():
        return None
    try:
        from torch.utils.data import DataLoader  # noqa: E402
        from adjointrwm.data import WindowDataset, WindowSpec, episode_split, fit_normaliser, restore_cache  # noqa: E402
        from adjointrwm.models import AdjointRWMConfig, AdjointRecursiveWorldModel  # noqa: E402
    except Exception as e:  # pragma: no cover
        print(f"[real] imports unavailable ({e}); falling back to synthetic")
        return None
    manifest = run_dir / "cache" / "cache_manifest.json"
    if not manifest.exists():
        print("[real] cache manifest absent; falling back to synthetic")
        return None
    print("[real] loading dataset...")
    cache_manifest = json.loads(manifest.read_text())
    records = restore_cache(cache_manifest, Path(args.local_cache))
    assignment = episode_split([r["episode_id"] for r in records])
    for r in records:
        r["split"] = assignment[r["episode_id"]]
    by_split = {s: [r for r in records if r["split"] == s] for s in ("train", "validation", "test")}

    def load_arrays(rec):
        with np.load(rec["cached_path"]) as ep:
            return {"states": ep["states"], "actions": ep["actions"]}

    train_arrays = [load_arrays(r) for r in by_split["train"]]
    normalisation = fit_normaliser([a["states"] for a in train_arrays], [a["actions"] for a in train_arrays])
    spec = WindowSpec(context_len=8, horizon=4, stride=2)
    train_ds = WindowDataset(by_split["train"], spec, normalisation, visual_layout="flat")
    test_ds = WindowDataset(by_split["test"], spec, normalisation, visual_layout="flat")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    cfg = json.loads((run_dir / "config" / "run_config.json").read_text())["config"]
    sample = next(iter(train_loader))
    model_config = AdjointRWMConfig(
        context_len=cfg["context_len"], horizon=cfg["horizon"], d_model=cfg["d_model"],
        transformer_layers=cfg["transformer_layers"], transformer_heads=cfg["transformer_heads"],
        transformer_ff=cfg["transformer_ff"], dropout=cfg["dropout"],
        num_refinement_candidates=cfg["num_refinement_candidates"],
        candidate_costs=cfg["candidate_costs"], rate_beta=cfg["rate_beta"],
        mask_mode=cfg["mask_mode"], prediction_mode=cfg["prediction_mode"],
    )
    sd, ad = sample["context_state"].shape[-1], sample["context_action"].shape[-1]
    vd = sample["context_visual"].shape[-1]
    teacher = AdjointRecursiveWorldModel(sd, ad, vd, model_config).to(device)
    ckpt = torch.load(run_dir / "jobs" / "seed_0" / "dynamics" / "best.pt", map_location=device, weights_only=False)
    teacher.load_state_dict(ckpt["model_state_dict"], strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    d = teacher.d_model
    torch.manual_seed(0)
    heads = {"costate": CostateEstimator(d).to(device),
             "critic": DirectCritic(d, matched_critic_hidden(d)).to(device),
             "curvature": CurvatureCostateEstimator(d).to(device)}
    opt = torch.optim.AdamW([p for h in heads.values() for p in h.parameters()], lr=args.lr)
    train_iter = iter(train_loader)
    from adjointrwm.training import move_batch  # noqa: E402
    for step in range(args.max_steps):
        try:
            b = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            b = next(train_iter)
        b = move_batch(b, device)
        tgt = exact_targets(teacher, b, args.cost_weight)
        latent, effects, costs = tgt["latent"], tgt["effects"], tgt["costs"]
        n = latent.shape[0]
        budget = torch.full((n,), 1.0 / teacher.num_candidates, device=device)
        horizon = torch.ones(n, device=device)
        opt.zero_grad()
        lam = heads["costate"](latent, budget, horizon)
        l1 = (1.0 - F.cosine_similarity(lam, tgt["exact_costate"], dim=-1)).mean()
        sc = with_hold_zero(heads["critic"](latent, effects[:, 1:], costs[1:], budget, horizon))
        l2 = F.smooth_l1_loss(sc[:, 1:], tgt["exact_gain"][:, 1:])
        lam3, h3 = heads["curvature"](latent, budget, horizon)
        l3 = (1.0 - F.cosine_similarity(lam3, tgt["exact_costate"], dim=-1)).mean()
        (l1 + l2 + l3).backward()
        opt.step()
    for h in heads.values():
        h.eval()
    # Evaluate on test windows.
    per_policy_gains = {p: [] for p in POLICIES}
    t0 = time.perf_counter()
    nb = 0
    with torch.no_grad():
        for b in test_loader:
            b = move_batch(b, device)
            tgt = exact_targets(teacher, b, args.cost_weight)
            gain = tgt["exact_gain"]
            n = gain.shape[0]
            scores = score_all_policies(heads, tgt, uncert_weight=args.uncert_weight)
            for name in POLICIES:
                ch = torch.zeros(n, dtype=torch.long) if name == "always_mode0" else scores[name].argmax(1).cpu()
                per_policy_gains[name].append(gain.cpu().numpy()[np.arange(n), ch.numpy()])
            nb += 1
            if nb >= args.num_batches and args.num_batches > 0:
                break
    lat_ms = (time.perf_counter() - t0) / max(1, nb) / args.batch_size * 1000.0
    gains = {p: np.concatenate(v) for p, v in per_policy_gains.items()}
    best = np.stack(list(gains.values()), axis=1).max(axis=1)
    oracle = gains["exact_costate"]
    per_policy = {}
    for p in POLICIES:
        reg = best - gains[p]
        per_policy[p] = {
            "mean_regret": float(reg.mean()), "std_regret": float(reg.std()),
            "mean_gain": float(gains[p].mean()),
            "gap_to_oracle": float(reg.mean() - (best - oracle).mean()),
            "win_rate_vs_critic": float((gains[p] > gains["direct_critic"]).mean()),
            "win_rate_vs_mode0": float((gains[p] > gains["always_mode0"]).mean()),
        }
    return {"mode": "real", "device": str(device), "run_id": args.run_id,
            "n_windows": int(len(gains["exact_costate"])),
            "mean_latency_ms_per_window": float(lat_ms), "per_policy": per_policy,
            "num_batches": nb, "batch_size": args.batch_size, "max_steps": args.max_steps,
            "cost_weight": args.cost_weight}


def save_outputs(summary: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "curvature_voi_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary to {summary_path}")
    oracle_r = summary["per_policy"]["exact_costate"]["mean_regret"]
    lines = [
        "# Direction 3 Curvature and Belief-Space VOI Allocation Benchmark",
        "",
        f"**Date (UTC):** {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} · **Mode:** `{summary['mode']}` · "
        f"**Device:** `{summary['device']}` · **Windows:** `{summary['n_windows']}`",
        f"**Mean scoring latency:** `{summary['mean_latency_ms_per_window']:.4f} ms/window` (all-policy scoring time)",
        "",
        "## Test regret, gap to oracle, and win rates",
        "",
        "| Allocator head | Mean test regret | Gap to oracle | Win rate vs critic | Win rate vs mode0 |",
        "|---|:---:|:---:|:---:|:---:|",
    ]
    for p in POLICIES:
        m = summary["per_policy"][p]
        lines.append(
            f"| `{p}` | `{m['mean_regret']:.5f}` | `+{m['gap_to_oracle']:.5f}` | "
            f"`{m['win_rate_vs_critic']:.3f}` | `{m['win_rate_vs_mode0']:.3f}` |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "* `exact_costate` is the diagnostic oracle floor (autograd co-state); deployable heads never read it.",
        "* `always_mode0` is the refusal baseline (always hold, gain exactly 0).",
        "* Belief-space VOI uses the documented `delta_cov = effects^2` proxy in synthetic mode.",
        f"* Oracle-floor mean regret in this run: `{oracle_r:.5f}`.",
    ]
    report_path = output_dir / "curvature_voi_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    print(f"Saved report to {report_path}")


def main(argv=None) -> int:
    args = parse_args(argv)
    print("=" * 70)
    print("Direction 3: Curvature and Belief-Space VOI Allocation Benchmark")
    print("Heads: exact_costate / always_mode0 / direct_critic / first_order /")
    print("       normalized_first_order / second_order_curvature / belief_space_voi")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    run_dir = Path(args.drive_root) / "runs" / args.run_id
    summary = None
    if not args.synthetic_test and run_dir.exists():
        summary = try_real_benchmark(args, device)
        if summary is None:
            print("[Notice] Real path unavailable; running synthetic diagnostic instead.")
    else:
        reason = "--synthetic-test flag" if args.synthetic_test else f"missing run_dir {run_dir}"
        print(f"[Notice] Running in synthetic diagnostic mode ({reason}).")
    if summary is None:
        summary = run_synthetic(args, device)
    summary["date_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary["policies"] = list(POLICIES)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_DIR / output_dir
    save_outputs(summary, output_dir)
    print("\nFINAL SUMMARY")
    for p in POLICIES:
        m = summary["per_policy"][p]
        print(f"  {p:>22}: regret={m['mean_regret']:.5f} gap=+{m['gap_to_oracle']:.5f} "
              f"win/critic={m['win_rate_vs_critic']:.3f} win/mode0={m['win_rate_vs_mode0']:.3f}")
    print(f"Latency: {summary['mean_latency_ms_per_window']:.4f} ms/window")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
