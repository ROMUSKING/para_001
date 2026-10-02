#!/usr/bin/env python3
"""Build the 500-episode feature cache for Milestone E3.1 / E3.2 on Colab L4.

Streams the 500 stratified episodes identified in e3_1_droid_500_manifest.json,
extracts 14-D proprio state, 7-D action, and computes frozen ResNet-18
embeddings for both exterior and wrist cameras.
Saves .npz files and cache_manifest.json directly to Drive.
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
import torch

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from adjointrwm.data.droid import decode_text, extract_episode
from adjointrwm.data.windows import fit_normaliser
from adjointrwm.features import ResNet18Embedder
from adjointrwm.io import atomic_write_json, sha256_file


def build_e3_cache(
    manifest_path: str | Path,
    out_dir: str | Path,
    data_dir: str = "gs://gresearch/robotics",
    dataset_name: str = "droid:1.0.1",
    device: str = "cuda",
    batch_log: int = 25,
) -> Dict[str, Any]:
    manifest_path = Path(manifest_path)
    out_dir = Path(out_dir)
    episodes_dir = out_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DROID 500-EPISODE FEATURE CACHE BUILDER (Milestone E3.1 / E3.2)")
    print(f"Manifest: {manifest_path}")
    print(f"Output Directory: {out_dir}")
    print(f"Dataset: {dataset_name} ({data_dir}) | Device: {device}")
    print("=" * 80, flush=True)

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    target_episodes = {ep["episode_id"]: ep for ep in manifest["episodes"]}
    target_count = len(target_episodes)
    print(f"Loaded manifest with {target_count} target episodes across {len(manifest.get('site_counts', {}))} sites.")

    # Initialize Embedder
    print(f"Initializing ResNet18Embedder on {device}...", flush=True)
    embedder = ResNet18Embedder(device=device)

    import tensorflow_datasets as tfds

    print("Opening TFDS dataset stream...", flush=True)
    builder = tfds.builder(dataset_name, data_dir=data_dir)
    # Stream train split without skipping images since we need to embed
    ds = builder.as_dataset(split="train[:1200]", shuffle_files=False)

    records: List[Dict[str, Any]] = []
    train_states: List[np.ndarray] = []
    train_actions: List[np.ndarray] = []

    t0 = time.time()
    cached_count = 0

    for idx, ex in enumerate(ds):
        rf = decode_text(ex["episode_metadata"]["recording_folderpath"].numpy())
        fp = decode_text(ex["episode_metadata"]["file_path"].numpy())
        ep_id = hashlib.sha256(f"{rf}|{fp}".encode()).hexdigest()[:24]

        if ep_id not in target_episodes:
            continue

        meta = target_episodes[ep_id]
        npz_path = episodes_dir / f"{ep_id}.npz"

        # Check if already cached
        if npz_path.exists():
            data = np.load(npz_path)
            cached_count += 1
            rec = {
                "episode_id": ep_id,
                "site": meta["site"],
                "scene": meta["scene"],
                "split": meta["split"],
                "length": int(data["states"].shape[0]),
                "num_steps": meta.get("num_steps", int(data["states"].shape[0])),
                "cached_path": str(npz_path),
                "cached_sha256": sha256_file(npz_path),
            }
            records.append(rec)
            if meta["split"] == "train":
                train_states.append(data["states"])
                train_actions.append(data["actions"])
        else:
            steps_list = list(ex["steps"])
            if len(steps_list) < 13:
                continue

            parsed = extract_episode(steps_list, frame_stride=2)
            states = parsed["states"].astype(np.float32)
            actions = parsed["actions"].astype(np.float32)

            ext_imgs = parsed["images"]["exterior_image_1_left"]
            wrist_imgs = parsed["images"]["wrist_image_left"]

            # Compute visual embeddings on GPU in BF16/FP16
            with torch.no_grad():
                ext_emb = embedder(ext_imgs).astype(np.float32)
                wrist_emb = embedder(wrist_imgs).astype(np.float32)

            # Atomic save
            tmp_path = npz_path.with_suffix(".tmp.npz")
            np.savez_compressed(
                tmp_path,
                states=states,
                actions=actions,
                exterior_embeddings=ext_emb,
                wrist_embeddings=wrist_emb,
            )
            tmp_path.replace(npz_path)

            cached_count += 1
            rec = {
                "episode_id": ep_id,
                "site": meta["site"],
                "scene": meta["scene"],
                "split": meta["split"],
                "length": int(states.shape[0]),
                "num_steps": meta.get("num_steps", len(steps_list)),
                "cached_path": str(npz_path),
                "cached_sha256": sha256_file(npz_path),
            }
            records.append(rec)
            if meta["split"] == "train":
                train_states.append(states)
                train_actions.append(actions)

        if cached_count % batch_log == 0 or cached_count == target_count:
            elapsed = time.time() - t0
            rate = cached_count / max(1e-3, elapsed)
            print(f"  Cached {cached_count:3d}/{target_count} episodes ({rate:.1f} eps/s, {elapsed:.1f}s elapsed) | Last: {meta['site']} ({meta['split']})", flush=True)

        if cached_count >= target_count:
            break

    print(f"\nCaching complete: {cached_count} episodes cached in {time.time() - t0:.1f}s.")

    # Compute and save normalisation statistics from train episodes
    print("Computing normalisation statistics across train episodes...", flush=True)
    normaliser = fit_normaliser(train_states, train_actions)
    norm_path = out_dir / "normalisation.npz"
    np.savez(norm_path, **normaliser)
    print(f"Saved normaliser: {norm_path} (trained on {len(train_states)} episodes)")

    # Save cache manifest
    cache_manifest = {
        "dataset_name": dataset_name,
        "source": data_dir,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_cached": len(records),
        "split_counts": {
            "train": len([r for r in records if r["split"] == "train"]),
            "val": len([r for r in records if r["split"] in ("val", "validation")]),
            "test": len([r for r in records if r["split"] == "test"]),
        },
        "episodes": records,
    }

    cache_manifest_path = out_dir / "cache_manifest.json"
    atomic_write_json(cache_manifest_path, cache_manifest)
    print(f"Saved cache manifest: {cache_manifest_path} ({cache_manifest_path.stat().st_size / 1024:.1f} KB)")
    print("=" * 80, flush=True)

    return cache_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build E3.1 500-episode feature cache.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="/content/drive/MyDrive/Colab Notebooks/AdjointRWM_Production/data/manifests/e3_1_droid_500_manifest.json",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/content/drive/MyDrive/Colab Notebooks/AdjointRWM_Production/data/cache_e3_1",
    )
    parser.add_argument("--data-dir", type=str, default="gs://gresearch/robotics")
    parser.add_argument("--dataset-name", type=str, default="droid:1.0.1")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-log", type=int, default=25)

    args = parser.parse_args()
    build_e3_cache(
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        device=args.device,
        batch_log=args.batch_log,
    )
