#!/usr/bin/env python3
"""scripts/diagnose_early_pruning.py

Session 6C frozen-DINOv2 control, second half (review R9): evaluate the actual
early-pruning execution path instead of selecting among cached final tokens.

Two arms on the same stratified windows, same backbone, same budgets:

- ``full_image``: raw frames → DINOv2 ViT-S/14 → 4x4 pooled tokens → backbone J.
  This also measures the encoder refresh cost the cache-assisted reading omits.
- ``preview_pruned``: keep the top-k preview cells per camera (grayscale gradient
  energy, no encoder), zero the rest at the pixel level BEFORE the encoder, then the
  identical encode → pool → J path. Quality effect of pruning information, with timing.

What this does and does not show: pixel-masking keeps the full token count through the
transformer (no FLOP saving measured here) — it tests whether pruned information
preserves decision quality, which is necessary but not sufficient for an encoder-saving
claim. True token-dropping would need DINOv2 surgery and is explicitly out of scope;
the timing table separates measured stages from that unmeasured step.

Usage::

    python scripts/diagnose_early_pruning.py \\
        --cache-dir /content/spatial_cache_e3_1 --drive-root /content/local_runs \\
        --run-id spatial_patches_20261003 --output-dir results/benchmarks/early_pruning
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_DIR = Path(__file__).resolve().parent.parent
for candidate in (REPO_DIR / "src", Path("/content/para_001/src")):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

sys.path.insert(0, str(REPO_DIR / "scripts"))

from adjointrwm.features import DinoV2Embedder, pool_patch_grid  # noqa: E402
from adjointrwm.io import atomic_write_json  # noqa: E402
from adjointrwm.spatial_selection import (  # noqa: E402
    CAMERAS,
    PATCHES_PER_CAMERA,
    camera_of_patch,
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
from diagnose_norm_controls import episode_id_of, gradient_energy_preview  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Session 6C early-pruning control")
    parser.add_argument("--cache-dir", type=str, default="/content/spatial_cache_e3_1")
    parser.add_argument("--drive-root", type=str, default="/content/local_runs")
    parser.add_argument("--run-id", type=str, default="spatial_patches_20261003")
    parser.add_argument("--output-dir", type=str, default="results/benchmarks/early_pruning")
    parser.add_argument("--windows", type=int, default=32)
    parser.add_argument("--budgets", type=str, default="2,4,8")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tfds-split", type=str, default="train[:1200]")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--timing-warmup", type=int, default=5)
    parser.add_argument("--timing-reps", type=int, default=20)
    return parser.parse_args(argv)


def pixel_mask_for_cells(kept: np.ndarray, height: int, width: int, grid: int = 4) -> np.ndarray:
    """Boolean ``[H, W]`` keep-mask from kept 4x4 preview-cell indices.

    Pure function — unit-tested. Cells tile the frame; kept cells stay True.
    """
    mask = np.zeros((height, width), dtype=bool)
    ch, cw = height // grid, width // grid
    for cell in [int(c) for c in np.atleast_1d(kept)]:
        row, col = divmod(cell, grid)
        mask[row * ch:(row + 1) * ch, col * cw:(col + 1) * cw] = True
    return mask


def stream_raw_frames(wanted: set, tfds_split: str) -> Dict[str, dict]:
    """Raw exterior+wrist frames per wanted episode (released per episode)."""
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
        out[episode_id] = {"exterior": np.stack(ext), "wrist": np.stack(wrist)}
        if len(out) >= len(wanted):
            break
    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def timed_block(fn, warmup: int, reps: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - start) / max(1, reps)


def run(args, device: torch.device) -> dict:
    torch.manual_seed(args.seed)
    path = Path(args.drive_root) / "runs" / args.run_id / "best_spatial.pt"
    if not path.exists():
        raise SystemExit(f"missing teacher checkpoint at {path}: hard failure")
    model = build_spatial_model(args, PATCHES_PER_CAMERA, device, args.width, 384)
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state), strict=False)
    model.eval()
    budgets = [int(k) for k in args.budgets.split(",") if k.strip()]

    _train, _val, test_set, site_by_episode = load_spatial_splits(
        Path(args.cache_dir), PATCHES_PER_CAMERA)
    from adjointrwm.data.windows import stratified_window_indices
    parent_ids = np.asarray(test_set.episode_ids())
    indices = stratified_window_indices(parent_ids, site_by_episode, args.windows, seed=args.seed)
    episodes_seen = sorted({str(e) for e in parent_ids[indices].tolist()})
    print(f"pruning windows: {len(indices)} over {len(episodes_seen)} episodes", flush=True)
    frames = stream_raw_frames(set(episodes_seen), args.tfds_split)
    missing = [e for e in episodes_seen if e not in frames]
    if missing:
        raise SystemExit(f"raw frames unavailable for {len(missing)} episodes")

    embedder = DinoV2Embedder(device=str(device), image_size=args.image_size, grid=4)
    loader = DataLoader(Subset(test_set, indices.tolist()), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"),
                        persistent_workers=args.num_workers > 0)
    def encode_camera(frames_uint8: np.ndarray) -> np.ndarray:
        """DINOv2 CLS+patches per camera -> patch tokens only (CLS dropped, matching cache)."""
        return embedder(frames_uint8)[:, 1:, :]

    index = camera_of_patch(PATCHES_PER_CAMERA, CAMERAS)
    regrets: Dict[str, Dict[str, List[float]]] = {
        f"k_cam={k}": {"full_image": [], "preview_pruned": []} for k in budgets}
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            starts = batch.get("window_start")
            for k_cam in budgets:
                full_tokens, pruned_tokens = [], []
                for i in range(batch["context_state"].shape[0]):
                    episode = str(batch["episode_id"][i])
                    start = int(starts[i]) if starts is not None else 0
                    ext_frames = frames[episode]["exterior"][start:start + 8]
                    wrist_frames = frames[episode]["wrist"][start:start + 8]
                    full_tokens.append(np.concatenate(
                        [encode_camera(ext_frames), encode_camera(wrist_frames)], axis=1))
                    preview = np.concatenate(
                        [gradient_energy_preview(ext_frames),
                         gradient_energy_preview(wrist_frames)], axis=1)
                    scores = torch.as_tensor(preview.mean(axis=0), device=device)
                    mask = select_topk_per_camera(scores.unsqueeze(0), k_cam, index,
                                                  CAMERAS)[0]
                    kept_ext = set(np.nonzero(mask[:16].cpu().numpy())[0].tolist())
                    kept_wrist = set(np.nonzero(mask[16:].cpu().numpy())[0].tolist())
                    pruned_ext = np.stack([
                        frame * pixel_mask_for_cells(
                            sorted(kept_ext), frame.shape[0], frame.shape[1])[..., None]
                        for frame in ext_frames])
                    pruned_wrist = np.stack([
                        frame * pixel_mask_for_cells(
                            sorted(kept_wrist), frame.shape[0], frame.shape[1])[..., None]
                        for frame in wrist_frames])
                    pruned_tokens.append(np.concatenate(
                        [encode_camera(pruned_ext), encode_camera(pruned_wrist)], axis=1))
                full_visual = torch.stack(
                    [torch.as_tensor(t, dtype=torch.float32, device=device)
                     for t in full_tokens])
                pruned_visual = torch.stack(
                    [torch.as_tensor(t, dtype=torch.float32, device=device)
                     for t in pruned_tokens])
                oracle = objective_at_masks(
                    model, batch, greedy_oracle_masks(
                        model, batch, k_cam, index)[0].unsqueeze(1))[:, 0]
                for name, visual in (("full_image", full_visual),
                                     ("preview_pruned", pruned_visual)):
                    probed = dict(batch, context_visual=visual)
                    value = objective_at_masks(
                        model, probed, torch.ones(visual.shape[0], 1, 32,
                                                 device=device))[:, 0]
                    regrets[f"k_cam={k_cam}"][name].extend(
                        (value - oracle).cpu().tolist())
            del batch
            if device.type == "cuda":
                torch.cuda.empty_cache()

    probe_frame = np.zeros((180, 320, 3), dtype=np.uint8)
    timing = {
        "preview_per_frame_ms": timed_block(
            lambda: gradient_energy_preview(probe_frame[None], 4), args.timing_warmup,
            args.timing_reps, torch.device("cpu")),
        "dinov2_encode_per_frame_ms": timed_block(
            lambda: embedder(probe_frame[None]), args.timing_warmup,
            min(args.timing_reps, 10), device),
        "selection_topk_ms": timed_block(
            lambda: select_topk_per_camera(torch.randn(1, 32, device=device),
                                           2, index, CAMERAS),
            args.timing_warmup, args.timing_reps, device),
    }
    return {
        "benchmark": "session_6c_early_pruning",
        "mode": "real",
        "device": str(device),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backbone": {"path": str(path), "sha256": sha256_file(path)},
        "arms": {
            "full_image": "raw frames -> DINOv2 (refresh-cost reference)",
            "preview_pruned": "top-k preview cells kept at pixel level pre-encoder",
        },
        "episodes": episodes_seen,
        "n_windows": int(len(indices)),
        "seed": args.seed,
        "budgets": budgets,
        "regret": {budget: {name: {"mean": float(np.mean(v)), "trimmed_mean": trimmed_mean(
            np.asarray(v, dtype=float)), "n": len(v)} for name, v in per_budget.items()}
            for budget, per_budget in regrets.items()},
        "timing": timing,
        "timing_protocol": {"warmup": args.timing_warmup, "reps": args.timing_reps,
                            "torch_version": torch.__version__},
        "caveats": [
            "Pixel masking keeps full token counts: quality effect only, no FLOP saving "
            "measured; true token-dropping needs DINOv2 surgery (out of scope).",
            "Oracle is greedy rollout search (k_cam=2), not a proven optimum.",
        ],
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    print("=" * 70)
    print("Session 6C early-pruning control: pixel-masked encoder path")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    summary = run(args, device)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "early_pruning_summary.json", summary)
    print(f"Saved summary to {output_dir / 'early_pruning_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
