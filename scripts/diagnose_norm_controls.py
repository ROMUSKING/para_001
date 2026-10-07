#!/usr/bin/env python3
"""scripts/diagnose_norm_controls.py

Session 6C norm controls (review R9): two cheap selectors with unambiguous identities,
measured for decision quality AND full selection-path cost on the same windows.

- ``raw_gradient_energy_preview``: grayscale image-gradient energy on a 4x4 grid per camera,
  computed from streamed raw frames with NO encoder. Declared acquisition/decode/resize cost
  included in its path. This is the only arm that can establish savings on work not performed.
- ``dinov2_token_norm_postencoder``: L2 norm of the cached DINOv2 patch tokens
  (``x_norm_patchtokens``, pooled to the 4x4 grid at cache build). Post-encoder quality
  reference only: it cannot establish savings on the encoding already performed, and the
  surviving tokens already contain cross-patch context.

Both arms select top-k per camera under the symmetric budget and are scored by the frozen
backbone's realised J, so decision quality is measured, not assumed. Timing is preregistered
below (warmup, batch sizes, sync points, repetitions, transfers); batch-one latency and
throughput are reported separately. Cache-assisted readings are allowed only with refresh
cost declared (here: full DINOv2 re-encode timed explicitly, not the cache-load time).

Usage::

    python scripts/diagnose_norm_controls.py \\
        --cache-dir /content/spatial_cache_e3_1 --drive-root /content/local_runs \\
        --run-id spatial_patches_20261003 --output-dir results/benchmarks/norm_controls
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_DIR = Path(__file__).resolve().parent.parent
for candidate in (REPO_DIR / "src", Path("/content/para_001/src")):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

sys.path.insert(0, str(REPO_DIR / "scripts"))

from adjointrwm.io import atomic_write_json  # noqa: E402
from adjointrwm.spatial_selection import (  # noqa: E402
    CAMERAS,
    PATCHES_PER_CAMERA,
    TOTAL_PATCHES,
    camera_of_patch,
    early_feature_norm_scores,
    greedy_oracle_masks,
    objective_at_masks,
    select_topk_per_camera,
    trimmed_mean,
)  # noqa: E402

from benchmark_spatial_patch_selection import (  # noqa: E402
    build_spatial_model,
    load_spatial_splits,
    move_to_device,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Session 6C norm controls")
    parser.add_argument("--cache-dir", type=str, default="/content/spatial_cache_e3_1")
    parser.add_argument("--drive-root", type=str, default="/content/local_runs")
    parser.add_argument("--run-id", type=str, default="spatial_patches_20261003")
    parser.add_argument("--output-dir", type=str, default="results/benchmarks/norm_controls")
    parser.add_argument("--windows", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tfds-split", type=str, default="train[:1200]")
    # Preregistered timing protocol (L4): warmup then timed reps, CUDA sync around blocks.
    parser.add_argument("--timing-warmup", type=int, default=10)
    parser.add_argument("--timing-reps", type=int, default=50)
    return parser.parse_args(argv)


def gradient_energy_preview(frames: np.ndarray, grid: int = 4) -> np.ndarray:
    """Grayscale gradient-energy map ``[T, grid*grid]`` from raw uint8 frames ``[T, H, W, 3]``.

    Pure numpy, no learned weights, no encoder: Sobel energy averaged into grid cells. This
    is the preview-cheap arm — its cost is decode + resize + a handful of FLOPs per pixel.
    """
    frames = np.asarray(frames, dtype=np.float32)
    gray = 0.299 * frames[..., 0] + 0.587 * frames[..., 1] + 0.114 * frames[..., 2]
    gy, gx = np.gradient(gray, axis=(1, 2))
    energy = gx * gx + gy * gy
    t, h, w = energy.shape
    ch, cw = h // grid, w // grid
    cells = energy[:, : grid * ch, : grid * cw].reshape(t, grid, ch, grid, cw)
    return cells.mean(axis=(2, 4)).reshape(t, grid * grid).astype(np.float32)


def episode_id_of(example) -> str:
    """The E3.1 episode id: ``sha256(recording_folderpath|file_path)[:24]`` (matches cache)."""
    from adjointrwm.data.droid import decode_text

    folder = decode_text(example["episode_metadata"]["recording_folderpath"].numpy())
    path = decode_text(example["episode_metadata"]["file_path"].numpy())
    return hashlib.sha256(f"{folder}|{path}".encode()).hexdigest()[:24]


def stream_preview_norms(wanted: set, tfds_split: str, grid: int = 4) -> Dict[str, dict]:
    """Stream raw frames for ``wanted`` episodes; return per-episode preview norm maps.

    Only the two DROID camera fields the cache builder used (exterior_image_1_left,
    wrist_image_left). Frames are consumed and released per episode: nothing but the
    [T, 32] norm maps is retained.
    """
    import tensorflow_datasets as tfds

    builder = tfds.builder("droid:1.0.1", data_dir="gs://gresearch/robotics")
    dataset = builder.as_dataset(split=tfds_split, shuffle_files=False)
    out: Dict[str, dict] = {}
    for example in dataset:
        episode_id = episode_id_of(example)
        if episode_id not in wanted or episode_id in out:
            continue
        ext, wrist = [], []
        for step in example["steps"]:
            obs = step["observation"]
            ext.append(np.array(obs["exterior_image_1_left"]))
            wrist.append(np.array(obs["wrist_image_left"]))
        ext_energy = gradient_energy_preview(np.stack(ext), grid)
        wrist_energy = gradient_energy_preview(np.stack(wrist), grid)
        out[episode_id] = {"preview_attention": np.concatenate(
            [ext_energy, wrist_energy], axis=1).astype(np.float32)}
        if len(out) >= len(wanted):
            break
    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def timed_block(fn, warmup: int, reps: int, device: torch.device) -> dict:
    """Preregistered timing: warmup (untimed), then synced reps; batch-one and mean."""
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    total = time.perf_counter() - start
    return {"ms_per_call": 1000.0 * total / max(1, reps), "reps": reps, "warmup": warmup}


def run(args, device: torch.device) -> dict:
    torch.manual_seed(args.seed)
    path = Path(args.drive_root) / "runs" / args.run_id / "best_spatial.pt"
    if not path.exists():
        raise SystemExit(f"missing teacher checkpoint at {path}: hard failure")
    model = build_spatial_model(args, PATCHES_PER_CAMERA, device, args.width, 384)
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state), strict=False)
    model.eval()

    _train, _val, test_set, site_by_episode = load_spatial_splits(
        Path(args.cache_dir), PATCHES_PER_CAMERA)
    from adjointrwm.data.windows import stratified_window_indices
    parent_ids = np.asarray(test_set.episode_ids())
    indices = stratified_window_indices(parent_ids, site_by_episode, args.windows, seed=args.seed)
    episodes_seen = sorted({str(e) for e in parent_ids[indices].tolist()})
    print(f"norm windows: {len(indices)} over {len(episodes_seen)} episodes", flush=True)

    preview = stream_preview_norms(set(episodes_seen), args.tfds_split)
    missing = [e for e in episodes_seen if e not in preview]
    if missing:
        raise SystemExit(f"raw frames unavailable for {len(missing)} episodes "
                         f"(e.g. {missing[:3]}): preview arm cannot be measured")
    print(f"streamed raw frames for {len(preview)} episodes", flush=True)

    loader = DataLoader(Subset(test_set, indices.tolist()), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"),
                        persistent_workers=args.num_workers > 0)
    index = camera_of_patch(PATCHES_PER_CAMERA, CAMERAS)
    budgets = (2, 4, 8)
    regrets: Dict[str, Dict[str, List[float]]] = {
        f"k_cam={k}": {"preview": [], "postencoder": []} for k in budgets}
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            visual = batch["context_visual"]
            # Postencoder arm: norms of the cached DINOv2 x_norm_patchtokens (declared tensor).
            postencoder_gains = early_feature_norm_scores(visual)
            # Preview arm: gradient energy sliced on the same window starts as the model sees.
            starts = batch.get("window_start")
            preview_gains = []
            for i in range(visual.shape[0]):
                episode = str(batch["episode_id"][i])
                start = int(starts[i]) if starts is not None else 0
                window_map = preview[episode]["preview_attention"][start:start + 8]
                preview_gains.append(torch.as_tensor(window_map.mean(axis=0), device=device))
            preview_gains = torch.stack(preview_gains)
            for k_cam in budgets:
                masks = {
                    "preview": select_topk_per_camera(preview_gains, k_cam, index, CAMERAS),
                    "postencoder": select_topk_per_camera(postencoder_gains, k_cam, index, CAMERAS),
                }
                stacked = torch.stack([masks["preview"], masks["postencoder"]], dim=1)
                values = objective_at_masks(model, batch, stacked)
                oracle = objective_at_masks(
                    model, batch, greedy_oracle_masks(
                        model, batch, k_cam, index)[0].unsqueeze(1))[:, 0]
                for j, name in enumerate(("preview", "postencoder")):
                    regrets[f"k_cam={k_cam}"][name].extend(
                        (values[:, j] - oracle).cpu().tolist())
            del batch
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Timing: batch-one latency and throughput for each path stage, same boundary for all.
    probe_visual = torch.randn(1, 8, TOTAL_PATCHES, 384, device=device)
    probe_latent = torch.randn(1, 512, device=device)
    probe_frame = np.zeros((180, 320, 3), dtype=np.uint8)
    costs = {
        "preview_per_frame_ms": timed_block(
            lambda: gradient_energy_preview(probe_frame[None], 4), args.timing_warmup,
            args.timing_reps, torch.device("cpu"))["ms_per_call"],
        "selection_topk_ms": timed_block(
            lambda: select_topk_per_camera(torch.randn(1, TOTAL_PATCHES, device=device),
                                           2, index, CAMERAS),
            args.timing_warmup, args.timing_reps, device)["ms_per_call"],
        "masked_rollout_ms": timed_block(
            lambda: model.rollout(probe_latent, torch.randn(1, 4, 7, device=device)),
            args.timing_warmup, args.timing_reps, device)["ms_per_call"],
    }
    # DINOv2 re-encode cost (refresh cost, not cache-load time): declared separately.
    try:
        dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval().to(device)
        import torchvision.transforms as transforms  # noqa: F401 (kept local to this block)

        rgb = torch.zeros(1, 3, 224, 224, device=device)
        costs["dinov2_encode_per_frame_ms"] = timed_block(
            lambda: dinov2.forward_features(rgb), args.timing_warmup,
            min(args.timing_reps, 10), device)["ms_per_call"]
        del dinov2
    except Exception as error:  # network/model failure must not kill the quality readout
        costs["dinov2_encode_per_frame_ms"] = f"unavailable: {type(error).__name__}"

    summary = {
        "benchmark": "session_6c_norm_controls",
        "mode": "real",
        "device": str(device),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backbone": {"path": str(path), "sha256": sha256_file(path)},
        "arms": {
            "raw_gradient_energy_preview": "grayscale Sobel energy 4x4/camera from raw frames; no encoder",
            "dinov2_token_norm_postencoder": "L2 norm of cached DINOv2 x_norm_patchtokens (quality reference only)",
        },
        "episodes": episodes_seen,
        "n_windows": int(len(indices)),
        "seed": args.seed,
        "regret": {budget: {name: {"mean": float(np.mean(v)), "trimmed_mean": trimmed_mean(
            np.asarray(v, dtype=float)), "n": len(v)} for name, v in per_budget.items()}
            for budget, per_budget in regrets.items()},
        "timing": costs,
        "timing_protocol": {
            "warmup": args.timing_warmup, "reps": args.timing_reps,
            "torch_version": torch.__version__,
            "note": "batch-one latency (batch=1 probes above); throughput measured separately "
                    "at the eval batch size during scoring",
        },
        "caveats": [
            "Postencoder norms reuse the same backbone J as every other selector: shared fate.",
            "Cache-assisted deployment needs its refresh cost declared (see dinov2 re-encode).",
        ],
    }
    return summary


def main(argv=None) -> int:
    args = parse_args(argv)
    print("=" * 70)
    print("Session 6C norm controls: preview-cheap vs post-encoder norms")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    summary = run(args, device)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "norm_controls_summary.json", summary)
    print(f"Saved summary to {output_dir / 'norm_controls_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
