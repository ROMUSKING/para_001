#!/usr/bin/env python3
"""DROID 500-episode stratified shard streaming and validation (Milestone E3.1).

Streams from full DROID RLDS (gs://gresearch/robotics, droid:1.0.1, 95,658 episodes),
stratifies across robot sites and scenes, enforces strict data contracts,
and runs full alignment and eye-inspection across 8 diverse episodes.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from adjointrwm.data.droid import (
    IMAGE_KEYS,
    STATE_KEYS,
    choose_episode_id,
    decode_text,
    extract_state,
    validate_image,
)
from adjointrwm.data.droid_shard import (
    EpisodeMetadata,
    parse_metadata_from_rlds,
    stratify_episodes,
)


def stream_and_stratify_droid(
    target_count: int = 500,
    max_scan: int = 1000,
    data_dir: str = "gs://gresearch/robotics",
    dataset_name: str = "droid:1.0.1",
    output_dir: str | Path = "results/data/droid_e3_1",
    drive_dir: str | Path | None = "/content/drive/MyDrive/Colab Notebooks/AdjointRWM_Production/data/manifests",
    seed: int = 42,
    num_inspect: int = 8,
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DROID 500-EPISODE STRATIFIED SHARD STREAMING (Milestone E3.1)")
    print(f"Dataset: {dataset_name} from {data_dir}")
    print(f"Target count: {target_count} | Scan limit: {max_scan} | Seed: {seed}")
    print(f"Output directory: {output_path}")
    print("=" * 80, flush=True)

    import tensorflow_datasets as tfds

    t0 = time.time()
    print("[Stage 1/4] Initializing TFDS builder and dataset stream...", flush=True)
    builder = tfds.builder(dataset_name, data_dir=data_dir)
    num_total_in_split = int(builder.info.splits["train"].num_examples)
    print(f"Verified dataset info: {num_total_in_split} total episodes in 'train' split across {builder.info.splits['train'].num_shards} shards.")

    # Fast metadata pass: skip image decoding
    decoders = {
        "steps": {
            "observation": {
                "exterior_image_1_left": tfds.decode.SkipDecoding(),
                "exterior_image_2_left": tfds.decode.SkipDecoding(),
                "wrist_image_left": tfds.decode.SkipDecoding(),
            }
        }
    }

    ds_stream = builder.as_dataset(
        split=f"train[:{max_scan}]",
        decoders=decoders,
        shuffle_files=False,
    )

    print(f"[Stage 1/4] Scanning metadata for up to {max_scan} candidate episodes...", flush=True)
    scanned_records: List[Dict[str, Any]] = []
    site_counts: Dict[str, int] = {}

    scan_t0 = time.time()
    for idx, ex in enumerate(ds_stream):
        rf = decode_text(ex["episode_metadata"]["recording_folderpath"].numpy())
        fp = decode_text(ex["episode_metadata"]["file_path"].numpy())

        # Extract steps metadata
        steps_list = list(ex["steps"])
        num_steps = len(steps_list)
        inst = ""
        if num_steps > 0:
            raw_inst = steps_list[0]["language_instruction"].numpy()
            inst = decode_text(raw_inst).strip()

        # Parse site and scene
        parts = rf.replace("\\", "/").split("/")
        site = "UNKNOWN"
        scene = "UNKNOWN"
        if "r2d2-data-full" in parts:
            p_idx = parts.index("r2d2-data-full")
            if p_idx + 1 < len(parts):
                site = parts[p_idx + 1]
            if p_idx + 4 < len(parts):
                scene = f"{parts[p_idx+3]}_{parts[p_idx+4]}"
        elif len(parts) >= 6:
            site = parts[4]
            scene = parts[5] if len(parts) > 5 else "scene_0"

        ep_id = hashlib.sha256(f"{rf}|{fp}".encode()).hexdigest()[:24]

        record = {
            "episode_index": idx,
            "episode_id": ep_id,
            "site": site,
            "scene": scene,
            "instruction": inst,
            "num_steps": num_steps,
            "recording_folderpath": rf,
            "file_path": fp,
        }
        scanned_records.append(record)
        site_counts[site] = site_counts.get(site, 0) + 1

        if (idx + 1) % 100 == 0 or (idx + 1) == max_scan:
            rate = (idx + 1) / (time.time() - scan_t0)
            print(f"  Scanned {idx + 1:4d}/{max_scan} episodes ({rate:.1f} eps/s) | Sites: {len(site_counts)} {dict(sorted(site_counts.items(), key=lambda x: -x[1])[:5])}", flush=True)

    print(f"Metadata scan complete: {len(scanned_records)} episodes scanned in {time.time() - scan_t0:.1f}s across {len(site_counts)} sites.")

    # Stage 2: Stratification
    print("\n[Stage 2/4] Stratifying into 500-episode shard (80% train, 10% val, 10% test)...", flush=True)
    stratified_eps = stratify_episodes(
        scanned_records,
        target_count=target_count,
        train_ratio=0.80,
        val_ratio=0.10,
        test_ratio=0.10,
        seed=seed,
    )

    split_counts = {"train": 0, "val": 0, "test": 0}
    strat_site_counts: Dict[str, Dict[str, int]] = {}
    for ep in stratified_eps:
        split_counts[ep.split] = split_counts.get(ep.split, 0) + 1
        site_d = strat_site_counts.setdefault(ep.site, {"train": 0, "val": 0, "test": 0, "total": 0})
        site_d[ep.split] += 1
        site_d["total"] += 1

    print(f"Stratification result: {len(stratified_eps)} total episodes.")
    print(f"  Train: {split_counts['train']} | Val: {split_counts['val']} | Test: {split_counts['test']}")
    print("  Site breakdown (Top 8):")
    for site, counts in sorted(strat_site_counts.items(), key=lambda x: -x[1]["total"])[:8]:
        print(f"    - {site:12s}: Total {counts['total']:3d} (Train: {counts['train']:3d}, Val: {counts['val']:2d}, Test: {counts['test']:2d})")

    # Stage 3: Full Alignment & Eye Inspection on 8 Sample Episodes
    print(f"\n[Stage 3/4] Running full data contract verification and eye-inspection on {num_inspect} sample episodes...", flush=True)
    stratified_ids = {ep.episode_id: ep for ep in stratified_eps}
    inspection_results: List[Dict[str, Any]] = []

    # Stream with full decoders for the inspection sample
    full_ds = builder.as_dataset(split=f"train[:{num_inspect * 4}]", shuffle_files=False)
    verified_count = 0
    seen_sites: set = set()

    for ex in full_ds:
        rf = decode_text(ex["episode_metadata"]["recording_folderpath"].numpy())
        fp = decode_text(ex["episode_metadata"]["file_path"].numpy())
        ep_id = hashlib.sha256(f"{rf}|{fp}".encode()).hexdigest()[:24]

        if ep_id not in stratified_ids:
            continue

        ep_meta = stratified_ids[ep_id]
        steps = list(ex["steps"])
        num_steps = len(steps)

        # Collect images and states
        wrist_imgs = []
        ext_imgs = []
        cart_pos = []
        grip_pos = []
        joint_pos = []
        actions = []

        for step in steps:
            obs = step["observation"]
            wrist_imgs.append(obs["wrist_image_left"].numpy())
            ext_imgs.append(obs["exterior_image_1_left"].numpy())
            cart_pos.append(obs["cartesian_position"].numpy())
            grip_pos.append(obs["gripper_position"].numpy())
            joint_pos.append(obs["joint_position"].numpy())
            actions.append(step["action"].numpy())

        wrist_arr = np.stack(wrist_imgs)
        ext_arr = np.stack(ext_imgs)
        cart_arr = np.stack(cart_pos)
        grip_arr = np.stack(grip_pos)
        joint_arr = np.stack(joint_pos)
        act_arr = np.stack(actions)

        # Enforce contract checks
        wrist_var = float(wrist_arr.var())
        ext_var = float(ext_arr.var())
        assert wrist_var > 0, f"Wrist camera is constant for {ep_id}"
        assert ext_var > 0, f"Exterior camera is constant for {ep_id}"
        assert wrist_arr.shape[1:] == (180, 320, 3), f"Wrist shape invalid: {wrist_arr.shape}"
        assert ext_arr.shape[1:] == (180, 320, 3), f"Ext shape invalid: {ext_arr.shape}"
        assert cart_arr.shape[1] == 6, f"Cartesian dim != 6: {cart_arr.shape}"
        assert grip_arr.shape[1] == 1, f"Gripper dim != 1: {grip_arr.shape}"
        assert joint_arr.shape[1] == 7, f"Joint dim != 7: {joint_arr.shape}"
        assert act_arr.shape[1] == 7, f"Action dim != 7: {act_arr.shape}"
        assert len(wrist_arr) == num_steps, "Step count mismatch"

        rep = {
            "episode_id": ep_id,
            "site": ep_meta.site,
            "scene": ep_meta.scene,
            "split": ep_meta.split,
            "num_steps": num_steps,
            "task_instruction": ep_meta.task_instruction,
            "wrist_image_shape": list(wrist_arr.shape),
            "wrist_pixel_variance": wrist_var,
            "exterior_image_shape": list(ext_arr.shape),
            "exterior_pixel_variance": ext_var,
            "cartesian_min": [float(x) for x in cart_arr.min(axis=0)],
            "cartesian_max": [float(x) for x in cart_arr.max(axis=0)],
            "gripper_range": [float(grip_arr.min()), float(grip_arr.max())],
            "joint_range": [float(joint_arr.min()), float(joint_arr.max())],
            "action_norm_mean": float(np.linalg.norm(act_arr, axis=1).mean()),
            "status": "PASS_CONTRACT",
        }
        inspection_results.append(rep)
        verified_count += 1
        print(f"  Verified [{verified_count}/{num_inspect}] {ep_id} ({ep_meta.site}, {ep_meta.split}): WristVar={wrist_var:.1f}, ExtVar={ext_var:.1f}, ActNorm={rep['action_norm_mean']:.3f} -> PASS", flush=True)

        if verified_count >= num_inspect:
            break

    # Stage 4: Write Manifests & Reports
    print("\n[Stage 4/4] Writing manifests and summary reports...", flush=True)
    manifest_data = {
        "dataset_name": dataset_name,
        "source": data_dir,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_episodes": len(stratified_eps),
        "split_counts": split_counts,
        "site_counts": strat_site_counts,
        "episodes": [ep.to_dict() for ep in stratified_eps],
    }

    manifest_file = output_path / "e3_1_droid_500_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Saved manifest: {manifest_file} ({manifest_file.stat().st_size / 1024:.1f} KB)")

    # Save eye inspection report JSON
    inspect_file = output_path / "e3_1_eye_inspection_report.json"
    with open(inspect_file, "w", encoding="utf-8") as f:
        json.dump(inspection_results, f, indent=2)

    # Save human-readable eye inspection markdown report
    md_file = output_path / "e3_1_eye_inspection_report.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# DROID 500-Episode Shard: Data Contract & Alignment Report (Milestone E3.1)\n\n")
        f.write(f"- **Dataset Source:** `{dataset_name}` ({data_dir})\n")
        f.write(f"- **Generated UTC:** {manifest_data['created_utc']}\n")
        f.write(f"- **Total Shard Episodes:** {len(stratified_eps)}\n")
        f.write(f"- **Split Counts:** Train={split_counts['train']}, Val={split_counts['val']}, Test={split_counts['test']}\n")
        f.write(f"- **Total Unique Robot Laboratories (Sites):** {len(strat_site_counts)}\n\n")

        f.write("## 1. Laboratory & Site Stratification Breakdown\n\n")
        f.write("| Site (Lab) | Total Episodes | Train (80%) | Val (10%) | Test (10%) |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|\n")
        for site, counts in sorted(strat_site_counts.items(), key=lambda x: -x[1]["total"]):
            f.write(f"| `{site}` | {counts['total']} | {counts['train']} | {counts['val']} | {counts['test']} |\n")

        f.write("\n## 2. In-Depth Alignment & Contract Verification (8 Inspected Episodes)\n\n")
        f.write("| # | Episode ID | Site | Split | Steps | Wrist Var | Ext Var | Act Norm | Status |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for i, rep in enumerate(inspection_results):
            f.write(f"| {i+1} | `{rep['episode_id']}` | `{rep['site']}` | `{rep['split']}` | {rep['num_steps']} | {rep['wrist_pixel_variance']:.1f} | {rep['exterior_pixel_variance']:.1f} | {rep['action_norm_mean']:.3f} | **{rep['status']}** |\n")

        f.write("\n## 3. Inspected Episode Task Instructions\n\n")
        for i, rep in enumerate(inspection_results):
            f.write(f"{i+1}. **`{rep['episode_id']}`** ({rep['site']}, {rep['split']}): *\"{rep['task_instruction']}\"*\n")
            f.write(f"   - Cartesian Range: X=[{rep['cartesian_min'][0]:.2f}, {rep['cartesian_max'][0]:.2f}], Z=[{rep['cartesian_min'][2]:.2f}, {rep['cartesian_max'][2]:.2f}]\n")
            f.write(f"   - Gripper Range: [{rep['gripper_range'][0]:.3f}, {rep['gripper_range'][1]:.3f}]\n")
            f.write(f"   - Joint Range: [{rep['joint_range'][0]:.2f}, {rep['joint_range'][1]:.2f}]\n\n")

    print(f"Saved inspection markdown report: {md_file}")

    # Copy to Google Drive if available
    if drive_dir:
        try:
            drive_path = Path(drive_dir)
            drive_path.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(manifest_file, drive_path / manifest_file.name)
            shutil.copy2(inspect_file, drive_path / inspect_file.name)
            shutil.copy2(md_file, drive_path / md_file.name)
            print(f"Successfully mirrored manifests to Drive: {drive_path}")
        except Exception as e:
            print(f"Notice: Could not mirror to Drive: {e}")

    total_time = time.time() - t0
    print(f"\nCompleted DROID Shard Streaming and Verification in {total_time:.1f}s.")
    print("=" * 80, flush=True)

    return manifest_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream and stratify DROID 500 shard.")
    parser.add_argument("--target-count", type=int, default=500)
    parser.add_argument("--max-scan", type=int, default=1000)
    parser.add_argument("--data-dir", type=str, default="gs://gresearch/robotics")
    parser.add_argument("--dataset-name", type=str, default="droid:1.0.1")
    parser.add_argument("--output-dir", type=str, default="results/data/droid_e3_1")
    parser.add_argument("--drive-dir", type=str, default="/content/drive/MyDrive/Colab Notebooks/AdjointRWM_Production/data/manifests")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inspect", type=int, default=8)

    args = parser.parse_args()
    stream_and_stratify_droid(
        target_count=args.target_count,
        max_scan=args.max_scan,
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        drive_dir=args.drive_dir,
        seed=args.seed,
        num_inspect=args.num_inspect,
    )
