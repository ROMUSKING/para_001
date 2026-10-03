#!/usr/bin/env python3
"""Session 0: DataLoader + torch.profiler baseline for world-model dynamics training.

Synthetic throughput microbenchmark (NOT scientific evidence of model quality):
measures host-to-GPU pipeline throughput for two representation shapes at batch
size 32 across DataLoader worker counts [0, 2, 4]:

* ``pooled``: 1D spatially-pooled vectors, D=512  -> batch (32, 512)
* ``spatial``: 3 cameras x 256 patches x 384 dim -> batch (32, 768, 384),
  processed by a shared per-patch encoder (384->384) + mean-pool + MLP, so
  activation volume scales with token count while parameters stay compact.

Per config it reports: median step time (ms), dataloader wait % of step time
(time blocked in batch fetch + host-to-device copy vs total step wall), GPU
kernel execution % of active-window wall (from torch.profiler Kineto CUDA
events), and peak allocated / reserved VRAM (MiB).

Profiler: ``torch.profiler.profile(activities=[CPU, CUDA],
schedule(wait=2, warmup=2, active=10))`` per the Session 0 protocol.
A Chrome trace is exported for the (pooled, nw=0) config only.

Outputs (default --out-dir results/benchmarks/profiler):
* profiler_baseline_summary.json
* profiler_baseline_report.md
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from torch.profiler import ProfilerActivity, profile, schedule
    _HAS_PROFILER = True
except Exception:  # pragma: no cover - profiler always present with torch
    _HAS_PROFILER = False

BATCH_SIZE = 32
NUM_WORKERS_GRID = [0, 2, 4]
POOLED_DIM = 512
SPATIAL_CAMERAS, SPATIAL_PATCHES, SPATIAL_DIM = 3, 256, 384
SPATIAL_TOKENS = SPATIAL_CAMERAS * SPATIAL_PATCHES  # 768


# ---------------------------------------------------------------------------
# Synthetic pipeline datasets (on-the-fly generation simulates decode cost)
# ---------------------------------------------------------------------------

class SyntheticPooledDataset(Dataset):
    """One item: obs (512,) float32 + target (512,) float32, generated on the fly."""

    def __init__(self, length: int = 1024, dim: int = POOLED_DIM, seed: int = 0):
        self.length, self.dim = int(length), int(dim)
        self.seed = int(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed + int(i))
        obs = rng.standard_normal(self.dim, dtype=np.float32)
        tgt = rng.standard_normal(self.dim, dtype=np.float32)
        return {"obs": torch.from_numpy(obs), "target": torch.from_numpy(tgt)}


class SyntheticSpatialDataset(Dataset):
    """One item: obs (768, 384) float32 + target (512,) float32, generated on the fly."""

    def __init__(self, length: int = 1024, tokens: int = SPATIAL_TOKENS,
                 dim: int = SPATIAL_DIM, seed: int = 0):
        self.length, self.tokens, self.dim = int(length), int(tokens), int(dim)
        self.seed = int(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed + int(i))
        obs = rng.standard_normal((self.tokens, self.dim), dtype=np.float32)
        tgt = rng.standard_normal((POOLED_DIM,), dtype=np.float32)
        return {"obs": torch.from_numpy(obs), "target": torch.from_numpy(tgt)}


# ---------------------------------------------------------------------------
# Tiny dynamics proxies (parameter-compact; spatial FLOPs scale with tokens)
# ---------------------------------------------------------------------------

class PooledDynamics(torch.nn.Module):
    def __init__(self, dim: int = POOLED_DIM, width: int = 1024):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, width), torch.nn.ReLU(),
            torch.nn.Linear(width, dim),
        )

    def forward(self, obs):
        return self.net(obs)


class SpatialDynamics(torch.nn.Module):
    """Shared per-patch encoder -> mean pool -> MLP head."""

    def __init__(self, patch_dim: int = SPATIAL_DIM, width: int = 1024,
                 out_dim: int = POOLED_DIM):
        super().__init__()
        self.patch = torch.nn.Sequential(
            torch.nn.Linear(patch_dim, patch_dim), torch.nn.LayerNorm(patch_dim),
            torch.nn.GELU(),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(patch_dim, width), torch.nn.ReLU(),
            torch.nn.Linear(width, out_dim),
        )

    def forward(self, obs):  # obs: (B, T=768, 384)
        h = self.patch(obs)
        pooled = h.mean(dim=1)
        return self.head(pooled)


def _move(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def _train_step(model, optimizer, batch, device) -> float:
    batch = _move(batch, device)
    optimizer.zero_grad(set_to_none=True)
    pred = model(batch["obs"])
    loss = torch.nn.functional.mse_loss(pred, batch["target"])
    loss.backward()
    optimizer.step()
    return float(loss.detach())


# ---------------------------------------------------------------------------
# One configuration benchmark
# ---------------------------------------------------------------------------

def benchmark_config(repr_name: str, num_workers: int, batch_size: int,
                     timed_steps: int, device: torch.device,
                     export_trace_path: Path | None = None) -> dict:
    dataset = (SyntheticPooledDataset() if repr_name == "pooled"
               else SyntheticSpatialDataset())
    model = (PooledDynamics() if repr_name == "pooled" else SpatialDynamics()).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        generator=torch.Generator().manual_seed(0),
    )
    iterator = iter(loader)

    def next_batch():
        nonlocal iterator
        try:
            return next(iterator)
        except StopIteration:
            iterator = iter(loader)
            return next(iterator)

    # Warmup (untimed)
    for _ in range(5):
        _train_step(model, optimizer, next_batch(), device)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed loop: split data-wait vs total step wall
    step_ms, data_ms = [], []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(timed_steps):
        t0 = time.perf_counter()
        batch = next_batch()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        _train_step(model, optimizer, batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        data_ms.append(1000.0 * (t1 - t0))
        step_ms.append(1000.0 * (t2 - t0))

    step_arr, data_arr = np.asarray(step_ms), np.asarray(data_ms)
    dataloader_wait_pct = float(100.0 * data_arr.sum() / max(step_arr.sum(), 1e-9))

    # Profiler window: wait=2, warmup=2, active=10 -> 14 steps
    cuda_kernel_pct: float | None = None
    profiler_note = ""
    if _HAS_PROFILER:
        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        prof_sched = schedule(wait=2, warmup=2, active=10, repeat=1)
        wall_t0 = time.perf_counter()
        with profile(activities=activities, schedule=prof_sched,
                     record_shapes=False, with_stack=False) as prof:
            for _ in range(14):
                _train_step(model, optimizer, next_batch(), device)
                prof.step()
        wall_us = max((time.perf_counter() - wall_t0) * 1e6, 1.0)
        if export_trace_path is not None:
            export_trace_path.parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(export_trace_path))
        if device.type == "cuda":
            try:
                ka = prof.key_averages()
                total_cuda_us = sum(float(getattr(e, "self_device_time_total", 0.0) or 0.0) for e in ka)
                cuda_kernel_pct = float(100.0 * total_cuda_us / wall_us)
            except Exception as exc:  # pragma: no cover - defensive
                profiler_note = f"key_averages parse failed: {exc}"
        else:
            profiler_note = "CPU-only device: no CUDA kernel events."
    else:  # pragma: no cover - defensive
        profiler_note = "torch.profiler unavailable."

    if device.type == "cuda":
        peak_alloc_mib = float(torch.cuda.max_memory_allocated(device) / 1024**2)
        peak_reserved_mib = float(torch.cuda.max_memory_reserved(device) / 1024**2)
    else:
        peak_alloc_mib, peak_reserved_mib = 0.0, 0.0

    return {
        "representation": repr_name,
        "num_workers": num_workers,
        "batch_size": batch_size,
        "device": str(device),
        "timed_steps": timed_steps,
        "step_time_ms_median": float(np.median(step_arr)),
        "step_time_ms_mean": float(step_arr.mean()),
        "step_time_ms_p95": float(np.percentile(step_arr, 95)),
        "dataloader_wait_ms_mean": float(data_arr.mean()),
        "dataloader_wait_pct": dataloader_wait_pct,
        "gpu_kernel_exec_pct": cuda_kernel_pct,
        "peak_allocated_mib": peak_alloc_mib,
        "peak_reserved_mib": peak_reserved_mib,
        "profiler_note": profiler_note,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Session 0 DataLoader + profiler baseline.")
    parser.add_argument("--out-dir", type=str, default="results/benchmarks/profiler")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--timed-steps", type=int, default=30)
    parser.add_argument("--num-workers", type=int, nargs="*", default=list(NUM_WORKERS_GRID))
    parser.add_argument("--reps", type=str, nargs="*", default=["pooled", "spatial"])
    args, _ = parser.parse_known_args()  # known-only: remote kernels inject -f flags

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    np.random.seed(0)

    env = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "total_vram_gib": (round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
                           if torch.cuda.is_available() else None),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "batch_size": args.batch_size,
        "note": ("SYNTHETIC throughput microbenchmark: random tensors used only to "
                 "measure pipeline timing/VRAM scaling. Not model-quality evidence."),
    }

    results = []
    first = True
    for rep in args.reps:
        for nw in args.num_workers:
            print(f"--- repr={rep} num_workers={nw} batch={args.batch_size} ---", flush=True)
            trace_path = (out_dir / "chrome_trace_pooled_nw0.json") if first else None
            row = benchmark_config(rep, int(nw), args.batch_size, args.timed_steps,
                                   device, export_trace_path=trace_path)
            results.append(row)
            print(f"  step_med={row['step_time_ms_median']:.2f}ms "
                  f"data_wait={row['dataloader_wait_pct']:.1f}% "
                  f"cuda_kernel={row['gpu_kernel_exec_pct']} "
                  f"alloc={row['peak_allocated_mib']:.1f}MiB "
                  f"rsvd={row['peak_reserved_mib']:.1f}MiB", flush=True)
            first = False

    summary = {"environment": env, "configs": results}
    (out_dir / "profiler_baseline_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    def fmt(x, digits=2):
        return "n/a" if x is None else f"{x:.{digits}f}"

    lines = [
        "# Session 0 Profiler Baseline Report",
        "",
        f"**Date (UTC):** {env['timestamp_utc']} · **Device:** {env['device']} · "
        f"**GPU:** {env['gpu_name']} · **VRAM:** {env['total_vram_gib']} GiB · "
        f"**torch:** {env['torch_version']} · **Batch size:** {env['batch_size']}",
        "",
        "> SYNTHETIC throughput microbenchmark: batches are on-the-fly random tensors "
        "used only to measure pipeline timing and VRAM scaling. These numbers are not "
        "model-quality evidence and must not be cited as held-out gains.",
        "",
        "## Per-config results",
        "",
        "| Representation | num_workers | Step median (ms) | Step p95 (ms) | "
        "DataLoader wait % | GPU kernel exec % | Peak alloc (MiB) | Peak reserved (MiB) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['representation']} | {r['num_workers']} | "
            f"{r['step_time_ms_median']:.2f} | {r['step_time_ms_p95']:.2f} | "
            f"{r['dataloader_wait_pct']:.1f} | {fmt(r['gpu_kernel_exec_pct'], 1)} | "
            f"{r['peak_allocated_mib']:.1f} | {r['peak_reserved_mib']:.1f} |")
    # Gate verdict for G4-1 (profiler saturation): GPU-bound iff kernel share
    # dominates and dataloader wait is low on the best-worker pooled config.
    pooled = [r for r in results if r["representation"] == "pooled"]
    best = min(pooled, key=lambda r: r["step_time_ms_median"]) if pooled else None
    lines += ["", "## Gate G4-1 reading (profiler saturation)", ""]
    if best is not None:
        gpu_bound = ((best["gpu_kernel_exec_pct"] or 0.0) > 50.0
                     and best["dataloader_wait_pct"] < 15.0)
        lines.append(
            f"Best pooled config: num_workers={best['num_workers']}, "
            f"step median {best['step_time_ms_median']:.2f} ms, "
            f"dataloader wait {best['dataloader_wait_pct']:.1f}%, "
            f"GPU kernel exec {fmt(best['gpu_kernel_exec_pct'], 1)}%.")
        lines.append(
            f"Verdict: **{'GPU-BOUND (supports G4-1)' if gpu_bound else 'NOT GPU-saturated (G4-1 not met)'}** "
            f"— G4-1 requires GPU kernel execution to dominate step time with "
            f"dataloader wait < 15%.")
    else:
        lines.append("No pooled configs measured; no verdict.")
    lines += ["",
              "## Chrome trace",
              "",
              "Full Kineto trace for (pooled, num_workers=0): `chrome_trace_pooled_nw0.json` "
              "(if present; omitted from git if > 5 MB per repo policy).",
              ""]
    (out_dir / "profiler_baseline_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved summary + report to {out_dir}")


if __name__ == "__main__":
    main()
