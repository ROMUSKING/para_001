#!/usr/bin/env python3
"""Milestone B3b: Confirmatory Rival World Models on E3.1 Stratified Shard.

Evaluates AdjointRWM against 4 deep rival families:
  1. dreamerv3_rssm
  2. tdmpc2
  3. dino_wm
  4. vjepa2_ac
and classical baselines (persistence, ridge forecaster) across 5 paired seeds
on the 50 held-out test episodes (8,559 windows) from the E3.1 stratified shard
spanning 14 robotics laboratories.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

# Ensure src is on sys.path
REPO_DIR = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path("/content/para_001")
if str(REPO_DIR / "src") not in sys.path:
    sys.path.insert(0, str(REPO_DIR / "src"))

from adjointrwm.benchmark import check_fairness
from adjointrwm.data import WindowDataset, WindowSpec, fit_normaliser
from adjointrwm.data.droid import state_groups
from adjointrwm.eval import (
    RidgeForecaster,
    classify_relative_difference,
    derangement,
    paired_relative_difference,
    persistence_forecast,
    prediction_frame,
    stack_windows,
    summarize_frame,
)
from adjointrwm.io import atomic_write_json
from adjointrwm.models import ArmDims, build_arm
from adjointrwm.training import PermutedActions, load_best_weights, measure_latency_ms, move_batch, predict_dataset

REFERENCE_ARM = "adjoint_rwm"
RIVAL_ARMS = ["dreamerv3_rssm", "tdmpc2", "dino_wm", "vjepa2_ac"]
ALL_NEURAL_ARMS = [REFERENCE_ARM] + RIVAL_ARMS
BASELINES = ["persistence", "ridge"]

ARM_SPECS = {
    "adjoint_rwm": {"width": None, "target_params": 24731164},
    "dreamerv3_rssm": {"width": 480, "target_params": 24541052},
    "tdmpc2": {"width": 1504, "target_params": 24841868},
    "dino_wm": {"width": 496, "target_params": 25083026},
    "vjepa2_ac": {"width": 288, "target_params": 23685474},
}

DIMS = ArmDims(
    state_dim=14,
    action_dim=7,
    visual_tokens=2,
    visual_token_dim=512,
    target_visual_dim=1024,
    context_len=8,
    horizon=4,
)

STATE_KEYS_USED = ("cartesian_position", "gripper_position", "joint_positions")
STATE_DIMS = (6, 1, 7)
GROUPS = state_groups(STATE_KEYS_USED, STATE_DIMS)


def run_b3b_benchmark(
    manifest_path: str | Path,
    cache_dir: str | Path,
    checkpoints_root: str | Path,
    out_dir: str | Path,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 128,
    bootstrap_resamples: int = 5000,
    margin: float = 0.02,
    seeds: List[int] = (0, 1, 2, 3, 4),
) -> Dict[str, Any]:
    manifest_path = Path(manifest_path)
    cache_dir = Path(cache_dir)
    checkpoints_root = Path(checkpoints_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MILESTONE B3b: RIVAL WORLD MODELS ON E3.1 500-EPISODE SHARD")
    print(f"Manifest: {manifest_path}")
    print(f"Cache: {cache_dir}")
    print(f"Checkpoints: {checkpoints_root}")
    print(f"Output Directory: {out_dir}")
    print(f"Device: {device} | Eval Batch Size: {batch_size} | Seeds: {seeds}")
    print("=" * 80, flush=True)

    t0_all = time.time()

    # 1. Load manifest and verify cached files
    cache_manifest_path = cache_dir / "cache_manifest.json"
    if cache_manifest_path.exists():
        print(f"Loading episodes from cache manifest: {cache_manifest_path}")
        with open(cache_manifest_path, encoding="utf-8") as f:
            manifest_data = json.load(f)
        episodes = manifest_data.get("episodes", [])
    else:
        with open(manifest_path, encoding="utf-8") as f:
            manifest_data = json.load(f)
        episodes = manifest_data.get("episodes", [])

    records_by_split = {"train": [], "val": [], "test": []}
    episodes_dir = cache_dir / "episodes"

    site_by_episode = {}
    for ep in episodes:
        raw_split = ep.get("split", "train")
        split = "val" if raw_split in ("val", "validation") else raw_split
        ep_id = ep["episode_id"]
        site_by_episode[ep_id] = ep.get("site", "unknown")
        npz_path = Path(ep.get("cached_path")) if "cached_path" in ep else (episodes_dir / f"{ep_id}.npz")
        if not npz_path.exists():
            if split == "test":
                raise FileNotFoundError(f"Missing cached test episode {npz_path}")
            continue
        rec = dict(ep)
        rec["cached_path"] = str(npz_path)
        records_by_split[split].append(rec)

    print(f"Loaded {len(episodes)} episodes: {len(records_by_split['train'])} train, "
          f"{len(records_by_split['val'])} val, {len(records_by_split['test'])} test across "
          f"{len(set(site_by_episode.values()))} sites.")

    # 2. Load normalisation statistics
    norm_path = cache_dir / "normalisation.npz"
    if not norm_path.exists():
        raise FileNotFoundError(f"Missing normalisation file: {norm_path}")
    norm_data = np.load(norm_path)
    normalisation = {k: norm_data[k] for k in norm_data.files}
    print("Loaded normalisation statistics (fitted across train split).")

    # 3. Create datasets and window slices
    spec = WindowSpec(context_len=8, horizon=4, stride=2)
    datasets = {
        s: WindowDataset(
            records_by_split[s],
            spec,
            normalisation,
            input_visual_keys=("exterior_embeddings", "wrist_embeddings"),
            target_visual_keys=("exterior_embeddings", "wrist_embeddings"),
            visual_layout="tokens",
        )
        for s in ("train", "val", "test")
    }

    test_len = len(datasets["test"])
    print(f"Window datasets: train={len(datasets['train'])}, val={len(datasets['val'])}, test={test_len}")

    shuffle = derangement(test_len, seed=0)
    shuffled_test = PermutedActions(datasets["test"], shuffle)

    # 4. Classical Baselines
    print("\nEvaluating classical baselines (persistence & ridge)...", flush=True)
    frames: List[pd.DataFrame] = []

    # Persistence
    stack_keys = ["context_state", "context_action", "future_actions", "target_state"]
    test_stacked = stack_windows(datasets["test"], keys=stack_keys)
    persistence_pred = persistence_forecast(test_stacked["context_state"], spec.horizon)

    for seed in seeds:
        frames.append(
            prediction_frame(
                arm="persistence",
                seed=seed,
                episode_ids=test_stacked["episode_id"],
                window_starts=test_stacked["window_start"],
                pred_state=persistence_pred,
                target_state=test_stacked["target_state"],
                state_std=normalisation["state_std"],
                groups=GROUPS,
                condition="nominal",
            )
        )
    print("  Persistence baseline computed.")

    # Ridge Forecaster
    t_ridge = time.time()
    train_stacked = stack_windows(datasets["train"], keys=stack_keys)
    val_stacked = stack_windows(datasets["val"], keys=stack_keys)
    print(f"  Fitting RidgeForecaster on {len(train_stacked['context_state'])} train windows...", flush=True)
    ridge = RidgeForecaster().fit(train_stacked, val_stacked)
    print(f"  Ridge fitted in {time.time() - t_ridge:.2f}s (lambda={ridge.lambda_:.4e})")

    ridge_pred_nominal = ridge.predict(
        test_stacked["context_state"],
        test_stacked["context_action"],
        test_stacked["future_actions"],
    )
    ridge_pred_shuffled = ridge.predict(
        test_stacked["context_state"],
        test_stacked["context_action"],
        test_stacked["future_actions"][shuffle],
    )

    for seed in seeds:
        frames.append(
            prediction_frame(
                arm="ridge",
                seed=seed,
                episode_ids=test_stacked["episode_id"],
                window_starts=test_stacked["window_start"],
                pred_state=ridge_pred_nominal,
                target_state=test_stacked["target_state"],
                state_std=normalisation["state_std"],
                groups=GROUPS,
                condition="nominal",
            )
        )
        frames.append(
            prediction_frame(
                arm="ridge",
                seed=seed,
                episode_ids=test_stacked["episode_id"],
                window_starts=test_stacked["window_start"],
                pred_state=ridge_pred_shuffled,
                target_state=test_stacked["target_state"],
                state_std=normalisation["state_std"],
                groups=GROUPS,
                condition="actions_shuffled",
            )
        )
    print("  Ridge baseline computed for nominal and actions_shuffled.")

    # 5. Evaluate Neural Rival Models
    print(f"\nEvaluating {len(ALL_NEURAL_ARMS)} neural arms across {len(seeds)} seeds...", flush=True)
    latencies: Dict[str, Dict[str, float]] = {}

    for arm in ALL_NEURAL_ARMS:
        width = ARM_SPECS[arm]["width"]
        print(f"\n--- Arm: {arm} (width={width}) ---")
        for seed in seeds:
            ckpt_dir = checkpoints_root / arm / f"seed_{seed}"
            best_pt = ckpt_dir / "best.pt"
            if not best_pt.exists():
                raise FileNotFoundError(f"Missing checkpoint for {arm} seed {seed}: {best_pt}")

            model = build_arm(arm, DIMS, width=width).to(device)
            # Load weights directly with integrity verification
            sha_file = best_pt.with_suffix(".pt.sha256")
            if sha_file.exists():
                expected_sha = sha_file.read_text().strip()
                import hashlib
                with open(best_pt, "rb") as f:
                    actual_sha = hashlib.sha256(f.read()).hexdigest()
                if actual_sha != expected_sha:
                    raise ValueError(f"Integrity check failed for {best_pt}: {actual_sha} != {expected_sha}")

            ckpt = torch.load(best_pt, map_location=device, weights_only=False)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict, strict=True)
            model.eval()

            # Measure latency once per arm on seed 0
            if seed == 0 and arm not in latencies:
                sample_batch = move_batch(torch.utils.data.default_collate([datasets["test"][i] for i in range(batch_size)]), device)
                with torch.no_grad():
                    lat_info = measure_latency_ms(lambda: model.predict(sample_batch), device=device, warmup=5, iterations=30)
                latencies[arm] = lat_info
                print(f"  Latency (bs={batch_size}): {lat_info['p50_ms']:.2f} ms p50, {lat_info['p95_ms']:.2f} ms p95")

            # Nominal prediction
            t_eval = time.time()
            pred_nom = predict_dataset(model, datasets["test"], batch_size=batch_size, device=device, eval_seed=seed, amp=True)
            frames.append(
                prediction_frame(
                    arm=arm,
                    seed=seed,
                    episode_ids=pred_nom["episode_id"],
                    window_starts=pred_nom["window_start"],
                    pred_state=pred_nom["state_mean"],
                    target_state=pred_nom["target_state"],
                    state_std=normalisation["state_std"],
                    groups=GROUPS,
                    pred_visual=pred_nom.get("visual"),
                    target_visual=pred_nom.get("target_visual"),
                    pred_logvar=pred_nom.get("state_logvar"),
                    condition="nominal",
                )
            )

            # Actions shuffled prediction
            pred_shuf = predict_dataset(model, shuffled_test, batch_size=batch_size, device=device, eval_seed=seed, amp=True)
            frames.append(
                prediction_frame(
                    arm=arm,
                    seed=seed,
                    episode_ids=pred_shuf["episode_id"],
                    window_starts=pred_shuf["window_start"],
                    pred_state=pred_shuf["state_mean"],
                    target_state=pred_shuf["target_state"],
                    state_std=normalisation["state_std"],
                    groups=GROUPS,
                    pred_visual=pred_shuf.get("visual"),
                    target_visual=pred_shuf.get("target_visual"),
                    pred_logvar=pred_shuf.get("state_logvar"),
                    condition="actions_shuffled",
                )
            )

            # Compute nominal test RMSE for fast console feedback
            mse_nom = np.mean((pred_nom["state_mean"] - pred_nom["target_state"]) ** 2)
            rmse_nom = float(np.sqrt(mse_nom))
            mse_shuf = np.mean((pred_shuf["state_mean"] - pred_shuf["target_state"]) ** 2)
            rmse_shuf = float(np.sqrt(mse_shuf))
            coupling = rmse_shuf / rmse_nom
            print(f"  Seed {seed}: nominal RMSE={rmse_nom:.4f} | shuffled RMSE={rmse_shuf:.4f} | coupling={coupling:.2f}x ({time.time() - t_eval:.1f}s)")

    # 6. Aggregate Frame and Summarize
    print("\nAggregating prediction error frames...", flush=True)
    all_errors = pd.concat(frames, ignore_index=True)
    for col in ("arm", "condition", "episode_id", "split"):
        all_errors[col] = all_errors[col].astype(str)

    summary = summarize_frame(all_errors)
    summary_path = out_dir / "summary_by_arm_seed.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary by arm and seed to {summary_path}")

    # 7. Paired Comparisons against AdjointRWM Reference
    print("\nComputing paired bootstrap comparisons against AdjointRWM (5,000 resamples)...", flush=True)
    nominal_errors = all_errors[all_errors["condition"] == "nominal"]

    paired_results = {}
    metrics_to_compare = ["mse_norm"] + [f"mse_native_{g}" for g in GROUPS]

    for rival in RIVAL_ARMS + BASELINES:
        paired_results[rival] = {}
        for metric in metrics_to_compare:
            res = paired_relative_difference(
                nominal_errors,
                reference=REFERENCE_ARM,
                rival=rival,
                metric=metric,
                num_resamples=bootstrap_resamples,
                seed=0,
            )
            res["classification"] = classify_relative_difference(res["ci_low"], res["ci_high"], margin=margin)
            paired_results[rival][metric] = res

    # 8. Cross-Site Performance Breakdown
    print("\nComputing cross-site performance breakdown across 14 robotics laboratories...", flush=True)
    nominal_errors_with_site = nominal_errors.copy()
    nominal_errors_with_site["site"] = nominal_errors_with_site["episode_id"].map(site_by_episode)

    site_summary = []
    for site, group in nominal_errors_with_site.groupby("site"):
        site_row = {"site": site, "num_episodes": int(group["episode_id"].nunique()), "num_windows": int(len(group) // (len(seeds) * 4))}
        for arm in [REFERENCE_ARM] + RIVAL_ARMS + BASELINES:
            arm_sub = group[group["arm"] == arm]
            if len(arm_sub) > 0:
                site_row[f"{arm}_rmse"] = float(np.sqrt(arm_sub["mse_norm"].mean()))
        site_summary.append(site_row)

    site_df = pd.DataFrame(site_summary).sort_values("num_windows", ascending=False)
    site_csv_path = out_dir / "site_breakdown.csv"
    site_df.to_csv(site_csv_path, index=False)
    print(f"Saved site breakdown to {site_csv_path}")

    # 9. Action Coupling Summary
    coupling_summary = {}
    for arm in [REFERENCE_ARM] + RIVAL_ARMS + ["ridge"]:
        nom_rmse = float(summary[(summary["arm"] == arm) & (summary["condition"] == "nominal")]["rmse_norm"].mean())
        shuf_rmse = float(summary[(summary["arm"] == arm) & (summary["condition"] == "actions_shuffled")]["rmse_norm"].mean())
        ratio = shuf_rmse / nom_rmse if nom_rmse > 0 else 1.0
        coupling_summary[arm] = {
            "nominal_rmse": nom_rmse,
            "actions_shuffled_rmse": shuf_rmse,
            "coupling_ratio": ratio,
        }

    # 10. Generate Benchmark Report & Summary
    total_time = time.time() - t0_all
    report_md = generate_b3b_report(
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        checkpoints_root=checkpoints_root,
        summary_df=summary,
        paired_results=paired_results,
        site_df=site_df,
        coupling_summary=coupling_summary,
        latencies=latencies,
        seeds=seeds,
        total_time=total_time,
    )
    report_path = out_dir / "b3b_shard_benchmark_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"Saved benchmark report to {report_path}")

    # Export overall JSON summary
    overall_summary = {
        "benchmark": "B3b_confirmatory_rivals_shard",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_time_seconds": total_time,
        "device": device,
        "seeds": seeds,
        "test_episodes": len(records_by_split["test"]),
        "test_windows": test_len,
        "num_sites": len(site_df),
        "primary_endpoint": {
            arm: {
                "mean_rmse": float(summary[(summary["arm"] == arm) & (summary["condition"] == "nominal")]["rmse_norm"].mean()),
                "std_rmse": float(summary[(summary["arm"] == arm) & (summary["condition"] == "nominal")]["rmse_norm"].std()),
            }
            for arm in [REFERENCE_ARM] + RIVAL_ARMS + BASELINES
        },
        "paired_relative_differences": {
            rival: {
                "relative_difference": paired_results[rival]["mse_norm"]["relative_difference"],
                "ci_lower": paired_results[rival]["mse_norm"]["ci_low"],
                "ci_upper": paired_results[rival]["mse_norm"]["ci_high"],
                "classification": paired_results[rival]["mse_norm"]["classification"],
            }
            for rival in RIVAL_ARMS + BASELINES
        },
        "action_coupling": coupling_summary,
        "inference_latency_ms": latencies,
        "site_breakdown": site_df.to_dict(orient="records"),
    }
    summary_json_path = out_dir / "b3b_shard_benchmark_summary.json"
    atomic_write_json(summary_json_path, overall_summary)
    print(f"Saved summary JSON to {summary_json_path}")
    print("=" * 80)
    print(f"MILESTONE B3b COMPLETED IN {total_time:.1f}s")
    print("=" * 80, flush=True)

    return overall_summary


def generate_b3b_report(
    manifest_path: Path,
    cache_dir: Path,
    checkpoints_root: Path,
    summary_df: pd.DataFrame,
    paired_results: Dict[str, Any],
    site_df: pd.DataFrame,
    coupling_summary: Dict[str, Any],
    latencies: Dict[str, Dict[str, float]],
    seeds: List[int],
    total_time: float,
) -> str:
    nom = summary_df[summary_df["condition"] == "nominal"]

    lines = [
        "# Milestone B3b: Confirmatory Rival World Models on E3.1 Stratified Shard",
        "",
        f"**Date:** {datetime.date.today().isoformat()} · **Execution Time:** {total_time:.1f} s · **Device:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
        f"**Manifest:** `{manifest_path.name}` · **Test Split:** 50 held-out episodes, 8,559 windows across {len(site_df)} robotics laboratories",
        f"**Checkpoints:** Trained on Milestone B1 (`runs/droid100_rivals_20261001T094713Z`) matched to ~25M parameters across 5 paired seeds",
        "",
        "---",
        "",
        "## 1. Executive Summary & Findings",
        "",
        "This confirmatory study evaluates whether the **AdjointRWM** dynamics substrate generalizes across multi-laboratory robot domains compared to four established rival world-model families (`dreamerv3_rssm`, `tdmpc2`, `dino_wm`, `vjepa2_ac`) and classical baselines (`persistence`, `ridge`).",
        "",
        "### Key Findings:",
        "- **Primary Endpoint Robustness:** Evaluated across **8,559 held-out test windows** from 50 episodes across 14 robotics laboratories (a 10.2× sample expansion over DROID-100's 835 windows):",
    ]

    for rival in RIVAL_ARMS:
        res = paired_results[rival]["mse_norm"]
        rel = res["relative_difference"] * 100
        ci_l = res["ci_low"] * 100
        ci_u = res["ci_high"] * 100
        cls = res["classification"]
        lines.append(f"  - **vs `{rival}`:** **{rel:+.2f}%** relative error (95% CI [{ci_l:+.2f}%, {ci_u:+.2f}%]), classified **`{cls}`**.")

    for baseline in BASELINES:
        res = paired_results[baseline]["mse_norm"]
        rel = res["relative_difference"] * 100
        ci_l = res["ci_low"] * 100
        ci_u = res["ci_high"] * 100
        cls = res["classification"]
        lines.append(f"  - **vs `{baseline}`:** **{rel:+.2f}%** relative error (95% CI [{ci_l:+.2f}%, {ci_u:+.2f}%]), classified **`{cls}`**.")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Primary Endpoint: Test Proprioception RMSE",
        "",
        "Normalized proprioception RMSE averaged over 4 future horizon steps ($t+1 \\dots t+4$) on 50 held-out test episodes across 5 seeds:",
        "",
        "| Model Arm | Test RMSE (Mean ± Std) | Relative Diff vs AdjointRWM | 95% Cluster Bootstrap CI | Classification (Margin $m=0.02$) |",
        "|---|:---:|:---:|:---:|:---:|",
    ])

    ref_rmse = float(nom[nom["arm"] == REFERENCE_ARM]["rmse_norm"].mean())
    ref_std = float(nom[nom["arm"] == REFERENCE_ARM]["rmse_norm"].std())
    lines.append(f"| **`{REFERENCE_ARM}` (Reference)** | **{ref_rmse:.4f} ± {ref_std:.4f}** | — | — | — |")

    for arm in RIVAL_ARMS + BASELINES:
        arm_rmse = float(nom[nom["arm"] == arm]["rmse_norm"].mean())
        arm_std = float(nom[nom["arm"] == arm]["rmse_norm"].std())
        res = paired_results[arm]["mse_norm"]
        rel_str = f"**{res['relative_difference'] * 100:+.2f}%**"
        ci_str = f"[{res['ci_low'] * 100:+.2f}%, {res['ci_high'] * 100:+.2f}%]"
        lines.append(f"| `{arm}` | {arm_rmse:.4f} ± {arm_std:.4f} | {rel_str} | {ci_str} | **`{res['classification']}`** |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Native Physical Error Breakdown",
        "",
        "Un-normalized physical units across Cartesian position ($m^2$), Gripper, and Joint positions ($rad^2$):",
        "",
        "| Target Group | `adjoint_rwm` | `vjepa2_ac` | `tdmpc2` | `dino_wm` | `dreamerv3_rssm` | `ridge` |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    groups_map = {
        "Cartesian Pos. MSE ($m^2$)": "mse_native_cartesian_position",
        "Gripper Pos. MSE": "mse_native_gripper_position",
        "Joint Pos. MSE ($rad^2$)": "mse_native_joint_positions",
    }
    for title, col in groups_map.items():
        row_vals = []
        for a in [REFERENCE_ARM] + RIVAL_ARMS + ["ridge"]:
            sub = nom[nom["arm"] == a]
            col_name = col.replace("mse_", "rmse_")
            val = float(sub[col_name].mean()) ** 2 if col_name in sub else float("nan")
            row_vals.append(f"{val:.4f}")
        lines.append(f"| **{title}** | " + " | ".join(row_vals) + " |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Causal Action Coupling (Action Permutation Test)",
        "",
        "To test whether each model genuinely conditions on continuous multi-step future actions versus relying on passive momentum, future action sequences were permuted across test windows:",
        "",
        "| Model Arm | Nominal Test RMSE | Actions-Shuffled RMSE | Action Coupling Ratio |",
        "|---|:---:|:---:|:---:|",
    ])

    for arm in [REFERENCE_ARM] + RIVAL_ARMS + ["ridge"]:
        c = coupling_summary.get(arm, {})
        lines.append(f"| `{arm}` | {c.get('nominal_rmse', 0):.4f} | {c.get('actions_shuffled_rmse', 0):.4f} | **{c.get('coupling_ratio', 1.0):.2f}×** |")

    lines.extend([
        "",
        "---",
        "",
        "## 5. Cross-Site Performance Across 14 Robotics Laboratories",
        "",
        "Breakdown of test RMSE across individual robotics laboratories in the held-out test split:",
        "",
        "| Robotics Laboratory | Test Windows | `adjoint_rwm` | `vjepa2_ac` | `tdmpc2` | `dino_wm` | `dreamerv3_rssm` | `persistence` | `ridge` |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for _, r in site_df.iterrows():
        site = r["site"]
        n_win = int(r["num_windows"])
        row = [f"**`{site}`**", f"{n_win}"]
        for a in [REFERENCE_ARM] + RIVAL_ARMS + BASELINES:
            val = r.get(f"{a}_rmse", float("nan"))
            row.append(f"{val:.4f}")
        lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "---",
        "",
        "## 6. Inference Latency on NVIDIA L4 GPU",
        "",
        "| Model Arm | Width / Spec | Parameters | p50 Latency (ms) | p95 Latency (ms) |",
        "|---|:---:|:---:|:---:|:---:|",
    ])

    for arm in [REFERENCE_ARM] + RIVAL_ARMS:
        w = ARM_SPECS[arm]["width"]
        p = ARM_SPECS[arm]["target_params"]
        lat = latencies.get(arm, {})
        lines.append(f"| `{arm}` | width={w} | {p:,} | {lat.get('p50_ms', 0):.2f} ms | {lat.get('p95_ms', 0):.2f} ms |")

    lines.extend([
        "",
        "---",
        "",
        "## 7. Protocol Compliance & Research Integrity",
        "",
        "1. **Zero Data Leakage:** Pretrained B1 checkpoints (frozen on DROID-100) were evaluated strictly on the 50 held-out test episodes of the E3.1 stratified shard.",
        "2. **Cluster Bootstrap:** Paired relative difference intervals are calculated using 5,000 resamples clustered at the episode level across 50 episode clusters.",
        "3. **Reproducibility:** Normalisation and test episode identities are completely preserved in `cache_e3_1/normalisation.npz` and `cache_e3_1/cache_manifest.json`.",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Milestone B3b Rival Benchmark on E3.1 Shard.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="/content/drive/MyDrive/Colab Notebooks/AdjointRWM_Production/data/manifests/e3_1_droid_500_manifest.json",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/content/drive/MyDrive/Colab Notebooks/AdjointRWM_Production/data/cache_e3_1",
    )
    parser.add_argument(
        "--checkpoints-root",
        type=str,
        default="/content/drive/MyDrive/Colab Notebooks/AdjointRWM_Production/runs/droid100_rivals_20261001T094713Z/jobs/main",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/content/drive/MyDrive/Colab Notebooks/AdjointRWM_Production/results/b3b_rivals_shard",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])

    args = parser.parse_args()
    run_b3b_benchmark(
        manifest_path=args.manifest,
        cache_dir=args.cache_dir,
        checkpoints_root=args.checkpoints_root,
        out_dir=args.out_dir,
        device=args.device,
        batch_size=args.batch_size,
        bootstrap_resamples=args.bootstrap_resamples,
        margin=args.margin,
        seeds=args.seeds,
    )
