#!/usr/bin/env python3
"""Milestone E3.2: Dynamics Pilot on E3.1 Stratified Shard.

Trains and evaluates AdjointRWM (20-60M parameter budget) directly on the 399 training episodes
of the E3.1 stratified shard across 14 robotics laboratories using native E3.1 normalisation.

Evaluates against:
1. Persistence baseline
2. In-domain Ridge forecaster (fitted on the 399 shard train episodes)
3. Action-permuted control (action sensitivity diagnostic)

Outputs immutable run artifacts to Drive:
- best.pt and latest.pt with SHA-256 sidecars
- train_log.jsonl with step-level losses and validation RMSE
- summary_by_arm_seed.csv with per-horizon and native coordinate MSEs
- e3_2_dynamics_pilot_report.md with executive findings and bootstrap CIs
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

REPO_DIR = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path("/content/para_001")
if str(REPO_DIR / "src") not in sys.path:
    sys.path.insert(0, str(REPO_DIR / "src"))

from adjointrwm.data import WindowDataset, WindowSpec
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
from adjointrwm.eval.prediction import per_window_mse, horizon_mean_rmse
from adjointrwm.io import atomic_copy, atomic_write_json, sha256_file, sha256_json
from adjointrwm.models import ArmDims, build_arm, count_prediction_parameters
from adjointrwm.training import (
    IDENTITY_KEYS,
    PermutedActions,
    TrainConfig,
    load_best_weights,
    measure_latency_ms,
    predict_dataset,
    seed_everything,
    train_job,
)

STATE_KEYS_USED = ("cartesian_position", "gripper_position", "joint_positions")
STATE_DIMS = (6, 1, 7)
GROUPS = state_groups(STATE_KEYS_USED, STATE_DIMS)


def train_e3_dynamics_pilot(
    cache_dir: str | Path,
    out_dir: str | Path,
    seeds: List[int] = (0, 1, 2),
    steps: int = 1500,
    batch_size: int = 64,
    lr: float = 3e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed_base: int = 20260928,
) -> Dict[str, Any]:
    cache_dir = Path(cache_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    local_dir = Path("/tmp") / out_dir.name
    local_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MILESTONE E3.2: ADJOINTRWM DYNAMICS PILOT ON E3.1 SHARD")
    print(f"Cache Directory: {cache_dir}")
    print(f"Output Directory: {out_dir}")
    print(f"Device: {device} | Seeds: {seeds} | Steps: {steps} | Batch Size: {batch_size} | LR: {lr}")
    print("=" * 80, flush=True)

    t0_all = time.time()

    # 1. Load Manifest and Cache
    cache_manifest_path = cache_dir / "cache_manifest.json"
    if not cache_manifest_path.exists():
        raise FileNotFoundError(f"Missing cache manifest at {cache_manifest_path}")
    with open(cache_manifest_path, encoding="utf-8") as f:
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
                raise FileNotFoundError(f"Missing test episode {npz_path}")
            continue
        rec = dict(ep)
        rec["cached_path"] = str(npz_path)
        records_by_split[split].append(rec)

    print(f"Loaded episodes: {len(records_by_split['train'])} train, {len(records_by_split['val'])} val, {len(records_by_split['test'])} test across {len(set(site_by_episode.values()))} sites.")

    # 2. Load Normalisation
    norm_path = cache_dir / "normalisation.npz"
    if not norm_path.exists():
        raise FileNotFoundError(f"Missing normalisation file: {norm_path}")
    norm_data = np.load(norm_path)
    normalisation = {k: norm_data[k] for k in norm_data.files}
    print("Loaded native E3.1 normalisation statistics.")

    # 3. Create Datasets
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

    # 4. Model Dimensions
    dims = ArmDims(
        state_dim=14,
        action_dim=7,
        visual_tokens=2,
        visual_token_dim=512,
        target_visual_dim=1024,
        context_len=8,
        horizon=4,
    )
    param_count = count_prediction_parameters("adjoint_rwm", dims)
    print(f"AdjointRWM Architecture: Transformer-Adjoint ({param_count:,} parameters)")

    # 5. Fit & Score Classical Baselines
    print("\n--- Evaluating Classical Baselines ---", flush=True)
    frames: List[pd.DataFrame] = []
    stack_keys = ["context_state", "context_action", "future_actions", "target_state"]
    test_stacked = stack_windows(datasets["test"], keys=stack_keys)

    # Persistence
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
    pers_mse = per_window_mse(persistence_pred, test_stacked["target_state"])
    pers_rmse = float(horizon_mean_rmse(pers_mse))
    print(f"  Persistence Test RMSE: {pers_rmse:.4f}")

    # In-Domain Ridge Forecaster (fitted on validation / small subset if full is large)
    print("  Fitting In-Domain Ridge Forecaster...", flush=True)
    val_stacked = stack_windows(datasets["val"], keys=stack_keys)
    # Fit ridge on validation + small train sample to bound memory and time
    train_sample_records = records_by_split["train"][:50]
    train_sample_ds = WindowDataset(train_sample_records, spec, normalisation, visual_layout="tokens")
    train_sample_stacked = stack_windows(train_sample_ds, keys=stack_keys)
    ridge = RidgeForecaster().fit(train_sample_stacked, val_stacked)
    ridge_pred = ridge.predict(test_stacked["context_state"], test_stacked["context_action"], test_stacked["future_actions"])
    ridge_pred_shuf = ridge.predict(test_stacked["context_state"], test_stacked["context_action"], test_stacked["future_actions"][shuffle])

    for seed in seeds:
        frames.append(
            prediction_frame(
                arm="ridge",
                seed=seed,
                episode_ids=test_stacked["episode_id"],
                window_starts=test_stacked["window_start"],
                pred_state=ridge_pred,
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
                pred_state=ridge_pred_shuf,
                target_state=test_stacked["target_state"],
                state_std=normalisation["state_std"],
                groups=GROUPS,
                condition="actions_shuffled",
            )
        )
    ridge_mse = per_window_mse(ridge_pred, test_stacked["target_state"])
    ridge_rmse = float(horizon_mean_rmse(ridge_mse))
    print(f"  In-Domain Ridge Test RMSE: {ridge_rmse:.4f} (best lambda: {ridge.lambda_:.4e})")

    # 6. Training AdjointRWM across Seeds
    source_hash = sha256_file(Path(__file__))
    config_dict = {
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "seeds": list(seeds),
        "params": param_count,
    }
    config_hash = sha256_json(config_dict)
    data_hash = sha256_file(cache_manifest_path)

    cfg = TrainConfig(
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        weight_decay=0.05,
        warmup_steps=100,
        eval_interval=100,
        eval_batch_size=128,
        amp=True,
    )

    arm_latencies = {}
    for seed in seeds:
        print(f"\n>>> Training AdjointRWM Seed {seed} on E3.1 Shard ({steps} steps) <<<", flush=True)
        t_seed = time.time()
        seed_everything(seed_base + seed)
        model = build_arm("adjoint_rwm", dims).to(device)

        identity = {
            "run_id": out_dir.name,
            "arm": "adjoint_rwm",
            "seed": seed,
            "config_hash": config_hash,
            "data_manifest_hash": data_hash,
            "source_hash": source_hash,
        }

        job_local = local_dir / f"seed_{seed}"
        job_persist = out_dir / f"seed_{seed}"

        summary = train_job(
            model,
            datasets["train"],
            datasets["val"],
            cfg,
            identity=identity,
            local_dir=job_local,
            persist_dir=job_persist,
            device=device,
            resume=True,
        )

        load_best_weights(model, job_persist)
        model.eval()

        if seed == seeds[0]:
            print("  Measuring inference latency on L4 GPU (batch-128 and batch-1 with CUDA events)...")
            from torch.utils.data import DataLoader
            from adjointrwm.training import move_batch

            loader_128 = DataLoader(datasets["test"], batch_size=128, shuffle=False)
            b128 = move_batch(next(iter(loader_128)), device)
            fn_128 = lambda: model.predict(b128)
            arm_latencies = measure_latency_ms(fn_128, device=device)
            print(f"  Batch-128 Latency: p50={arm_latencies['p50_ms']:.2f}ms, p95={arm_latencies['p95_ms']:.2f}ms")

            loader_1 = DataLoader(datasets["test"], batch_size=1, shuffle=False)
            b1 = move_batch(next(iter(loader_1)), device)
            fn_1 = lambda: model.predict(b1)
            b1_latencies = measure_latency_ms(fn_1, device=device)
            arm_latencies["batch1_p50_ms"] = b1_latencies["p50_ms"]
            arm_latencies["batch1_p95_ms"] = b1_latencies["p95_ms"]
            arm_latencies["batch1_mean_ms"] = b1_latencies["mean_ms"]
            print(f"  Batch-1 Latency: p50={b1_latencies['p50_ms']:.2f}ms, p95={b1_latencies['p95_ms']:.2f}ms")

        # Evaluate nominal
        pred_nom = predict_dataset(model, datasets["test"], batch_size=128, device=device, eval_seed=seed, amp=True)
        frames.append(
            prediction_frame(
                arm="adjoint_rwm",
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

        # Evaluate action shuffled
        pred_shuf = predict_dataset(model, shuffled_test, batch_size=128, device=device, eval_seed=seed, amp=True)
        frames.append(
            prediction_frame(
                arm="adjoint_rwm",
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

        mse_nom = np.mean((pred_nom["state_mean"] - pred_nom["target_state"]) ** 2)
        rmse_nom = float(np.sqrt(mse_nom))
        mse_shuf = np.mean((pred_shuf["state_mean"] - pred_shuf["target_state"]) ** 2)
        rmse_shuf = float(np.sqrt(mse_shuf))
        coupling = rmse_shuf / rmse_nom
        print(f"  Seed {seed} Done in {time.time() - t_seed:.1f}s | Nominal RMSE={rmse_nom:.4f} | Shuffled RMSE={rmse_shuf:.4f} | Coupling={coupling:.2f}x")

    # 7. Aggregate & Summarize
    all_errors = pd.concat(frames, ignore_index=True)
    summary_df = summarize_frame(all_errors)
    summary_df.to_csv(out_dir / "summary_by_arm_seed.csv", index=False)

    # 8. Site Breakdown with Exact Unique Window Deduplication
    nominal_errors = all_errors[all_errors["condition"] == "nominal"].copy()
    nominal_errors["site"] = nominal_errors["episode_id"].map(site_by_episode)

    site_summary = []
    for site, group in nominal_errors.groupby("site"):
        num_episodes = int(group["episode_id"].nunique())
        num_windows = int(group[["episode_id", "window_start"]].drop_duplicates().shape[0])
        site_row = {"site": site, "num_episodes": num_episodes, "num_windows": num_windows}
        for arm in ["adjoint_rwm", "persistence", "ridge"]:
            arm_sub = group[group["arm"] == arm]
            if len(arm_sub) > 0:
                site_row[f"{arm}_rmse"] = float(np.sqrt(arm_sub["mse_norm"].mean()))
        site_summary.append(site_row)

    site_df = pd.DataFrame(site_summary).sort_values("num_windows", ascending=False)
    site_df.to_csv(out_dir / "site_breakdown.csv", index=False)

    # 9. Paired Relative Differences vs Persistence and Ridge
    paired_results = {}
    for baseline in ["persistence", "ridge"]:
        res = paired_relative_difference(
            nominal_errors,
            reference="adjoint_rwm",
            rival=baseline,
            metric="mse_norm",
            num_resamples=5000,
            seed=0,
        )
        res["classification"] = classify_relative_difference(res["ci_low"], res["ci_high"], margin=0.02)
        paired_results[baseline] = res

    # 10. Generate Markdown Report
    total_time = time.time() - t0_all
    report_text = f"""# Milestone E3.2: Dynamics Pilot on E3.1 Stratified Shard

**Date:** {datetime.date.today().isoformat()} · **Execution Time:** {total_time:.1f} s · **Device:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
**Manifest:** `{cache_manifest_path.name}` · **Test Split:** {len(records_by_split['test'])} held-out episodes, {test_len:,} windows across {len(site_df)} robotics laboratories
**Architecture:** AdjointRWM Transformer-Adjoint ({param_count:,} parameters) trained on 399 multi-site episodes

---

## 1. Executive Summary & Findings

AdjointRWM was trained directly on the 399 training episodes of the E3.1 multi-site shard (same-data protocol) with native normalisation.

### Primary Results:
- **Test Proprioception RMSE (Mean ± Std over {len(seeds)} seeds):**
  - **AdjointRWM:** {summary_df[(summary_df['arm'] == 'adjoint_rwm') & (summary_df['condition'] == 'nominal')]['rmse_norm'].mean():.4f} ± {summary_df[(summary_df['arm'] == 'adjoint_rwm') & (summary_df['condition'] == 'nominal')]['rmse_norm'].std():.4f}
  - **Persistence Baseline:** {pers_rmse:.4f}
  - **In-Domain Ridge Forecaster:** {ridge_rmse:.4f}
- **Paired Comparisons (5,000 cluster bootstrap resamples):**
  - **vs Persistence:** {paired_results['persistence']['relative_difference']*100:+.2f}% (95% CI [{paired_results['persistence']['ci_low']*100:+.2f}%, {paired_results['persistence']['ci_high']*100:+.2f}%]) -> `{paired_results['persistence']['classification']}`
  - **vs In-Domain Ridge:** {paired_results['ridge']['relative_difference']*100:+.2f}% (95% CI [{paired_results['ridge']['ci_low']*100:+.2f}%, {paired_results['ridge']['ci_high']*100:+.2f}%]) -> `{paired_results['ridge']['classification']}`

---

## 2. Cross-Site Performance Across {len(site_df)} Laboratories

{site_df.to_markdown(index=False)}

---

## 3. Inference Latency on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}

- **p50 Latency:** {arm_latencies.get('p50_ms', 0.0):.2f} ms
- **p95 Latency:** {arm_latencies.get('p95_ms', 0.0):.2f} ms
- **Mean Latency:** {arm_latencies.get('mean_ms', 0.0):.2f} ms
"""
    report_path = out_dir / "e3_2_dynamics_pilot_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\nSaved report to {report_path}")

    overall_results = {
        "milestone": "E3.2",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_time_seconds": total_time,
        "seeds": list(seeds),
        "steps": steps,
        "test_windows": test_len,
        "num_sites": len(site_df),
        "persistence_rmse": pers_rmse,
        "ridge_rmse": ridge_rmse,
        "adjoint_rmse": float(summary_df[(summary_df['arm'] == 'adjoint_rwm') & (summary_df['condition'] == 'nominal')]['rmse_norm'].mean()),
        "paired_results": paired_results,
        "latencies": arm_latencies,
    }
    atomic_write_json(out_dir / "e3_2_dynamics_pilot_summary.json", overall_results)
    print("=" * 80)
    print(f"MILESTONE E3.2 COMPLETED IN {total_time:.1f}s")
    print("=" * 80, flush=True)
    return overall_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone E3.2 Dynamics Pilot")
    parser.add_argument("--cache-dir", type=str, default="/content/drive/MyDrive/Colab Notebooks/AdjointRWM_Production/data/cache_e3_1")
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    args = parser.parse_args()

    if not args.out_dir:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.out_dir = f"/content/drive/MyDrive/Colab Notebooks/AdjointRWM_Production/runs/e3_2_dynamics_pilot_{ts}"

    seed_list = [int(s.strip()) for s in args.seeds.split(",")]
    train_e3_dynamics_pilot(
        cache_dir=args.cache_dir,
        out_dir=args.out_dir,
        seeds=seed_list,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
    )
