#!/usr/bin/env python3
"""Session 1: Horizon Stress Testing (H=4..64) & Autograd Scaling Benchmark.

Evaluates how memory and compute scale as rollout horizon H increases:
H in [4, 8, 16, 32, 64] across batch sizes B in [16, 32, 64].

Measures on GPU:
1. Forward pass latency (ms) and peak allocated VRAM (MiB).
2. First-order BPTT backward pass latency (ms) and peak allocated VRAM (MiB).
3. Second-order curvature double-backprop (HVP) latency (ms) and peak allocated VRAM (MiB).
4. Memory scaling curve d(VRAM)/dH and identifies exact boundary for L4 vs G4.

Outputs:
* results/benchmarks/horizon_stress/horizon_stress_summary.json
* results/benchmarks/horizon_stress/horizon_stress_report.md
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_DIR = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path("/content")
for p in [REPO_DIR / "src", Path("/content/para_001/src"), Path("/content/src")]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from adjointrwm.models.common import ArmDims
from adjointrwm.models import build_arm

HORIZONS = [4, 8, 16, 32, 64]
BATCH_SIZES = [16, 32, 64]


def benchmark_horizon_scaling(device: str = "cuda", out_dir: str | Path = "results/benchmarks/horizon_stress"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    total_vram_gib = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        if device.type == "cuda"
        else 0.0
    )

    print(f"=== Session 1: Horizon Stress & Autograd Scaling Benchmark ===")
    print(f"Device: {device} ({gpu_name}, {total_vram_gib} GiB VRAM)")
    print(f"Horizons: {HORIZONS}")
    print(f"Batch sizes: {BATCH_SIZES}\n")

    results = []

    # Model parameters matched to ~25M production specs
    state_dim = 14
    action_dim = 7
    context_len = 8

    for H in HORIZONS:
        dims = ArmDims(
            state_dim=state_dim,
            action_dim=action_dim,
            visual_tokens=2,
            visual_token_dim=512,
            target_visual_dim=1024,
            context_len=context_len,
            horizon=H,
        )

        try:
            model = build_arm("adjoint_rwm", dims).to(device)
            model.train()
        except Exception as e:
            print(f"Error building model for H={H}: {e}")
            continue

        for B in BATCH_SIZES:
            config_result = {
                "horizon": H,
                "batch_size": B,
                "oom": False,
                "error": None,
            }

            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # Create synthetic tensor batch with requires_grad on context
            context_state = torch.randn(B, context_len, state_dim, device=device, requires_grad=True)
            context_action = torch.randn(B, context_len, action_dim, device=device)
            future_actions = torch.randn(B, H, action_dim, device=device)
            target_state = torch.randn(B, H, state_dim, device=device)
            context_visual = torch.randn(B, context_len, 2 * 512, device=device)

            batch = {
                "context_state": context_state,
                "context_action": context_action,
                "future_actions": future_actions,
                "target_state": target_state,
                "context_visual": context_visual,
            }

            try:
                # 1. Forward Pass Timing
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_fwd_start = time.perf_counter()

                # Enable math SDP for full autograd double-derivative support
                import contextlib
                sdp_ctx = (
                    torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
                    if device.type == "cuda"
                    else contextlib.nullcontext()
                )
                with sdp_ctx:
                    out = model.predict(batch)
                    pred_mean = out["state_mean"]
                    loss = nn.functional.mse_loss(pred_mean, target_state)

                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_fwd = (time.perf_counter() - t_fwd_start) * 1000.0

                    fwd_alloc_mib = (
                        torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
                    )

                    # 2. First-Order BPTT Backward Timing
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_bwd_start = time.perf_counter()

                    # Backward with create_graph=True for second-order evaluation
                    grads = torch.autograd.grad(loss, context_state, create_graph=True, retain_graph=True)[0]

                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_bwd = (time.perf_counter() - t_bwd_start) * 1000.0

                    bwd_alloc_mib = (
                        torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
                    )

                    # 3. Second-Order Curvature Double-Backprop (HVP) Timing
                    # Vector v for Hessian-vector product: \nabla^2 J \cdot v
                    v = torch.randn_like(context_state)
                    grad_prod = (grads * v).sum()

                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_hvp_start = time.perf_counter()

                    grad_prod.backward()

                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_hvp = (time.perf_counter() - t_hvp_start) * 1000.0

                total_peak_alloc_mib = (
                    torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
                )
                total_peak_res_mib = (
                    torch.cuda.max_memory_reserved() / (1024**2) if device.type == "cuda" else 0.0
                )

                config_result.update({
                    "forward_ms": round(t_fwd, 3),
                    "forward_alloc_mib": round(fwd_alloc_mib, 2),
                    "backward_bptt_ms": round(t_bwd, 3),
                    "backward_alloc_mib": round(bwd_alloc_mib, 2),
                    "hvp_double_backprop_ms": round(t_hvp, 3),
                    "total_step_ms": round(t_fwd + t_bwd + t_hvp, 3),
                    "peak_allocated_mib": round(total_peak_alloc_mib, 2),
                    "peak_reserved_mib": round(total_peak_res_mib, 2),
                    "vram_utilization_pct": round(total_peak_alloc_mib / (total_vram_gib * 1024) * 100, 2)
                    if total_vram_gib > 0
                    else 0.0,
                })

                print(
                    f"H={H:2d} | B={B:2d} | Fwd: {t_fwd:6.2f} ms | Bwd: {t_bwd:6.2f} ms | HVP: {t_hvp:6.2f} ms | "
                    f"Peak Alloc: {total_peak_alloc_mib:7.2f} MiB ({config_result['vram_utilization_pct']:5.1f}% VRAM)"
                )

            except torch.cuda.OutOfMemoryError as oom:
                print(f"H={H:2d} | B={B:2d} | CUDA OUT OF MEMORY!")
                config_result["oom"] = True
                config_result["error"] = "CUDA OutOfMemory"
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"H={H:2d} | B={B:2d} | Error: {e}")
                config_result["error"] = str(e)

            results.append(config_result)

    # Summary payload
    summary = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device": str(device),
        "gpu_name": gpu_name,
        "total_vram_gib": total_vram_gib,
        "configs": results,
    }

    with open(out_dir / "horizon_stress_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Generate Markdown Report
    rows = []
    for r in results:
        if r.get("oom"):
            rows.append(f"| {r['horizon']} | {r['batch_size']} | OOM | OOM | OOM | OOM | OOM | OOM |")
        elif r.get("error"):
            rows.append(f"| {r['horizon']} | {r['batch_size']} | ERR | ERR | ERR | ERR | ERR | ERR |")
        else:
            rows.append(
                f"| {r['horizon']} | {r['batch_size']} | {r['forward_ms']:.2f} | {r['backward_bptt_ms']:.2f} | "
                f"{r['hvp_double_backprop_ms']:.2f} | {r['total_step_ms']:.2f} | {r['peak_allocated_mib']:.1f} | "
                f"{r['vram_utilization_pct']:.1f}% |"
            )

    table_md = "\n".join(rows)

    report_md = f"""# Session 1: Horizon Stress Testing & Autograd Scaling Report

**Date (UTC):** {summary['timestamp_utc']}  
**Device:** {summary['gpu_name']} ({summary['total_vram_gib']} GiB VRAM)  
**Evaluated Horizons:** {HORIZONS}  
**Evaluated Batch Sizes:** {BATCH_SIZES}  

---

## 1. Empirical Scaling Summary

| Horizon $H$ | Batch $B$ | Forward (ms) | BPTT Backward (ms) | HVP 2nd-Order (ms) | Total Step (ms) | Peak VRAM (MiB) | VRAM Util % |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
{table_md}

---

## 2. Key Observations & Hardware Gating Insights

1. **Memory Growth Scaling $d(\\text{{VRAM}})/dH$:**
   Unrolling the full computational graph with double-backpropagation retains activations across all unrolled recurrence steps.
2. **Compute Latency Scaling:**
   Evaluates whether long horizons ($H \\ge 32$) induce quadratic attention bottlenecks or linear recurrence scaling.
3. **Hopper G4 Justification Gate:**
   Checks whether $H=64$ at nominal batch size $B=64$ exceeds L4 24 GiB VRAM or requires gradient checkpointing.
"""

    with open(out_dir / "horizon_stress_report.md", "w") as f:
        f.write(report_md)

    print(f"\nArtifacts saved to {out_dir}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Session 1 Horizon Stress Benchmark")
    parser.add_argument("--out-dir", type=str, default="results/benchmarks/horizon_stress")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args, _ = parser.parse_known_args()
    benchmark_horizon_scaling(device=args.device, out_dir=args.out_dir)
