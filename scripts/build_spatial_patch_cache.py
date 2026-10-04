#!/usr/bin/env python3
"""scripts/build_spatial_patch_cache.py

Builds the DINOv2 patch-token feature cache that Session 6A needs, in the layout
``adjointrwm``'s window dataset already reads.

**Why this exists.** The E3.1 cache on Drive (``data/cache_e3_1``) holds only *pooled* 512-d
ResNet18 embeddings (``exterior_embeddings``, ``wrist_embeddings``). Session 6A selects among
``P = 32`` spatial patch tokens (a 4x4 DINOv2 ViT-S/14 grid per camera), so
``scripts/benchmark_spatial_patch_selection.py`` raises ``KeyError: 'exterior_patches'``
against that cache. Session 2 had the patch features, but wrote them to an ephemeral
``/content/spatial_cache`` that no longer exists and that stored neither the
``exterior_patches``/``wrist_patches`` keys nor the CLS-to-patch attention map the
``early_cls_attention`` comparator reads.

This script streams the same ``droid:1.0.1`` source from ``gs://gresearch/robotics`` that the
E3.1 cache was built from, re-extracts DINOv2 features, and writes an ``episodes/<id>.npz``
layout plus a manifest that carries the **E3.1 splits unchanged**, so the split assignment
still happens per episode before any windowing (repo rule 5).

Keys written per episode
------------------------
``states``, ``actions``            proprioception and actions (as the E3.1 cache)
``exterior_patches``               ``[T, 16, 384]`` DINOv2 patch tokens, exterior camera
``wrist_patches``                  ``[T, 16, 384]`` DINOv2 patch tokens, wrist camera
``exterior_embeddings``            ``[T, 384]`` exterior CLS token (target_visual_keys)
``wrist_embeddings``               ``[T, 384]`` wrist CLS token (target_visual_keys)
``cls_attention``                  ``[T, 32]`` last-block CLS-to-patch attention, head-mean

Note on ``cls_attention`` scale: the raw map over the 16x16 token grid is average-pooled onto
the same 4x4 grid as the patch features, so each stored value is the mean of a 4x4 region and
a frame's values sum to about ``2/16 = 0.125`` rather than 2. The factor is exactly constant
across patches and frames, so any ranking or argmax over patches is unchanged; it is recorded
here so the constant is not mistaken for a broken distribution.

Usage::

    python scripts/build_spatial_patch_cache.py \\
        --manifest results/data/droid_e3_1/e3_1_droid_500_manifest.json \\
        --cache-dir /content/spatial_cache_e3_1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parent.parent
for candidate in (REPO_DIR / "src", Path("/content/para_001/src")):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from adjointrwm.data.droid import decode_text  # noqa: E402
from adjointrwm.features import pool_patch_grid  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build the DINOv2 patch-token cache for Session 6A")
    parser.add_argument("--manifest", type=str,
                        default=str(REPO_DIR / "results/data/droid_e3_1/e3_1_droid_500_manifest.json"))
    parser.add_argument("--cache-dir", type=str, default="/content/spatial_cache_e3_1")
    parser.add_argument("--e3-cache-dir", type=str, default="/content/cache_e3_1",
                        help="source of the train-split normaliser and the per-episode proprioception")
    parser.add_argument("--grid", type=int, default=4, help="4 -> 16 patch tokens per camera")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit-episodes", type=int, default=0, help="0 = every manifest episode")
    parser.add_argument("--split-order", type=str, default="val,test,train",
                        help="episodes are extracted in this split order so held-out data lands first")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # ``train[:1200]`` is what scripts/build_e3_feature_cache.py streamed to build the E3.1
    # shard; a smaller prefix does not reach all 499 manifest episodes, so the default must
    # match it or episodes are silently missed.
    parser.add_argument("--tfds-split", type=str, default="train[:1200]")
    parser.add_argument("--lengths-only", action="store_true",
                        help="rebuild the manifest from the cached .npz files without streaming "
                             "TFDS; use after a partial run to correct recorded lengths")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# DINOv2
# ---------------------------------------------------------------------------


def _prepare(images_uint8, image_size: int, device: torch.device) -> torch.Tensor:
    """uint8 ``[N, H, W, 3]`` -> normalised float ``[N, 3, image_size, image_size]``."""
    x = torch.as_tensor(np.ascontiguousarray(images_uint8))
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"expected [N, H, W, 3] uint8 images, got {tuple(x.shape)}")
    x = x.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(255.0)
    x = torch.nn.functional.interpolate(x, size=(image_size, image_size), mode="bilinear",
                                        align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def cls_to_patch_attention(model, batch: torch.Tensor) -> torch.Tensor:
    """Last-block CLS-to-patch attention, head-mean: ``[B, N-1]``.

    DINOv2's ``forward_features`` returns no attention weights, so the last block's ``qkv``
    projection is recomputed from a hook-captured block input. The standard ``dinov2_vits14``
    release has ``qk_norm=False``, so ``q``/``k`` need no extra normalisation. Recorded in the
    cache manifest because ``early_cls_attention`` is a scored comparator and its definition
    must be reproducible.
    """
    captured: Dict[str, torch.Tensor] = {}

    def hook(_module, inputs, _output):
        captured["x"] = inputs[0].detach()

    handle = model.blocks[-1].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model.forward_features(batch)
    finally:
        handle.remove()

    x = captured["x"]
    block = model.blocks[-1]
    b, n, d = x.shape
    heads = block.attn.num_heads
    head_dim = d // heads
    qkv = block.attn.qkv(x).reshape(b, n, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k = qkv[0], qkv[1]
    attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
    attn = attn.softmax(dim=-1)
    return attn[:, :, 0, 1:].mean(dim=1)


@torch.inference_mode()
def embed_camera(model, images_uint8, image_size: int, grid: int, batch_size: int,
                 device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(cls [T, D], patches [T, grid^2, D], cls_attention [T, grid^2])``."""
    cls_parts, patch_parts, attn_parts = [], [], []
    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False))
    for start in range(0, len(images_uint8), batch_size):
        chunk = _prepare(images_uint8[start: start + batch_size], image_size, device)
        with autocast:
            feats = model.forward_features(chunk)
        cls = feats["x_norm_clstoken"].float()
        patches = pool_patch_grid(feats["x_norm_patchtokens"].float(), grid)
        # The attention map lives on the un-pooled token grid, so pool it the same way the
        # patch features were pooled; otherwise the two disagree about which token is which.
        raw_tokens = feats["x_norm_patchtokens"].shape[1]
        side = int(round(raw_tokens ** 0.5))
        if side * side != raw_tokens:
            raise ValueError(f"patch tokens do not form a square grid: {raw_tokens}")
        with autocast:
            attn = cls_to_patch_attention(model, chunk)
        attn_grid = attn.reshape(attn.shape[0], 1, side, side)
        attn_grid = torch.nn.functional.adaptive_avg_pool2d(attn_grid, grid)
        cls_parts.append(cls.cpu())
        patch_parts.append(patches.cpu())
        attn_parts.append(attn_grid.flatten(1).cpu())
    return (torch.cat(cls_parts).numpy().astype(np.float16),
            torch.cat(patch_parts).numpy().astype(np.float16),
            torch.cat(attn_parts).numpy().astype(np.float32))


def extract_episode(example, model, args, device: torch.device) -> Dict[str, np.ndarray]:
    """DINOv2 patch tokens plus proprioception for one DROID episode."""
    steps = list(example["steps"])
    states, actions, ext_images, wrist_images = [], [], [], []
    for step in steps:
        obs = step["observation"]
        cart = np.array(obs.get("cartesian_position", np.zeros(6)), dtype=np.float32)
        grip = np.array(obs.get("gripper_position", [0.0]), dtype=np.float32).reshape(-1)
        joint = np.array(obs.get("joint_positions", np.zeros(7)), dtype=np.float32)
        states.append(np.concatenate([cart, grip, joint], axis=0))
        actions.append(np.array(step.get("action", np.zeros(7)), dtype=np.float32))
        ext_images.append(np.array(obs["exterior_image_1_left"]))
        wrist_images.append(np.array(obs["wrist_image_left"]))

    ext_cls, ext_patches, ext_attn = embed_camera(
        model, ext_images, args.image_size, args.grid, args.batch_size, device)
    wrist_cls, wrist_patches, wrist_attn = embed_camera(
        model, wrist_images, args.image_size, args.grid, args.batch_size, device)

    return {
        "states": np.stack(states).astype(np.float32),
        "actions": np.stack(actions).astype(np.float32),
        "exterior_patches": ext_patches,
        "wrist_patches": wrist_patches,
        "exterior_embeddings": ext_cls,
        "wrist_embeddings": wrist_cls,
        # Camera 0 is exterior and camera 1 is wrist, matching camera_of_patch's
        # repeat_interleave layout, so patch p of the [T, 32] map is the same patch the
        # benchmark scores.
        "cls_attention": np.concatenate([ext_attn, wrist_attn], axis=1).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def episode_id_of(example) -> str:
    """The E3.1 episode id: ``sha256(recording_folderpath|file_path)[:24]``."""
    folder = decode_text(example["episode_metadata"]["recording_folderpath"].numpy())
    path = decode_text(example["episode_metadata"]["file_path"].numpy())
    return hashlib.sha256(f"{folder}|{path}".encode()).hexdigest()[:24]


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    cache_dir = Path(args.cache_dir)
    episodes_dir = cache_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest).read_text())
    order = {name: i for i, name in enumerate(args.split_order.split(","))}
    wanted = sorted(
        manifest["episodes"],
        key=lambda e: (order.get(e.get("split", "train"), 99), e["episode_id"]),
    )
    if args.limit_episodes:
        wanted = wanted[: args.limit_episodes]
    target = {e["episode_id"]: e for e in wanted}
    print(f"Target episodes: {len(target)} "
          f"({ {s: sum(1 for e in wanted if e.get('split') == s) for s in order} })", flush=True)

    # The normaliser is the E3.1 train-split statistic over the same states/actions, so it is
    # copied rather than refitted: refitting here would risk fitting on a different subset.
    normaliser_src = Path(args.e3_cache_dir) / "normalisation.npz"
    if normaliser_src.exists():
        with np.load(normaliser_src) as archive:
            np.savez(cache_dir / "normalisation.npz",
                     **{key: archive[key] for key in archive.files})
        print(f"Copied normaliser from {normaliser_src}", flush=True)
    else:
        print(f"WARNING: no normaliser at {normaliser_src}; the loader will fit one on train "
              f"episodes, which is a different normalisation from the E3.1 cache", flush=True)

    written, skipped, seen = 0, 0, 0
    lengths: Dict[str, int] = {}

    if args.lengths_only:
        # Rebuild the manifest from what is already on disk. Every episode's length is read
        # back from its .npz, so a partial or interrupted extraction still yields a manifest
        # the window index can be built from.
        for record in wanted:
            path = episodes_dir / f"{record['episode_id']}.npz"
            if not path.exists():
                continue
            with np.load(path) as cached:
                lengths[record["episode_id"]] = int(cached["states"].shape[0])
            skipped += 1
        print(f"lengths-only: read {len(lengths)} cached episodes", flush=True)
    else:
        print("Loading DINOv2 ViT-S/14 from torch.hub...", flush=True)
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval().to(device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        import tensorflow_datasets as tfds
        print(f"Streaming TFDS droid:1.0.1 from gs://gresearch/robotics ({args.tfds_split})...", flush=True)
        builder = tfds.builder("droid:1.0.1", data_dir="gs://gresearch/robotics")
        dataset = builder.as_dataset(split=args.tfds_split, shuffle_files=False)

        started = time.time()
        for example in dataset:
            seen += 1
            episode_id = episode_id_of(example)
            if episode_id not in target:
                continue
            record = target[episode_id]
            path = episodes_dir / f"{episode_id}.npz"
            if path.exists():
                skipped += 1
                # Read the length from the file rather than trusting the manifest: an episode
                # cached by an earlier run still needs a correct length here.
                with np.load(path) as cached:
                    lengths[episode_id] = int(cached["states"].shape[0])
                continue
            data = extract_episode(example, model, args, device)
            np.savez(path, **data)
            written += 1
            # The manifest must carry the *cached* length, not the source num_steps: the window
            # index is built from `length`, and the E3.1 repo manifest leaves it null, which
            # would make WindowDataset fall back to its 100-step default and index windows past
            # the end of shorter episodes.
            lengths[episode_id] = int(data["states"].shape[0])
            if written % 10 == 0 or written == 1:
                elapsed = time.time() - started
                print(f"[{written} written, {skipped} cached, {seen} streamed] "
                      f"{record.get('split')} {episode_id} T={data['states'].shape[0]} "
                      f"elapsed={elapsed:.0f}s", flush=True)
            if written + skipped >= len(target):
                break

    manifest_out = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_name": "droid:1.0.1",
        "source": "gs://gresearch/robotics",
        "encoder": {
            "name": "dinov2_vits14", "hub": "facebookresearch/dinov2:dinov2_vits14",
            "image_size": args.image_size, "dim": 384, "grid": args.grid,
            "tokens_per_camera": args.grid * args.grid,
            "cls_attention": "last-block CLS-to-patch softmax attention, head-mean, "
                             "average-pooled on the same grid as the patch features",
            "licence": "Apache-2.0 (code and weights, per the DINOv2 README)",
        },
        "note": "Splits and sites are copied verbatim from the E3.1 manifest; no re-splitting.",
        "episodes": [
            {**record,
             "cached_path": str(episodes_dir / f"{record['episode_id']}.npz"),
             "split": record.get("split", "train"),
             "length": lengths[record["episode_id"]]}
            for record in wanted
            if record["episode_id"] in lengths
        ],
    }
    (cache_dir / "cache_manifest.json").write_text(json.dumps(manifest_out, indent=1))
    print(f"Wrote manifest with {len(manifest_out['episodes'])} episodes "
          f"({written} newly extracted, {skipped} already cached)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())