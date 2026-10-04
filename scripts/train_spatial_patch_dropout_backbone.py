#!/usr/bin/env python3
"""scripts/train_spatial_patch_dropout_backbone.py

Session 6A prerequisite: the patch-dropout spatial backbone required by spec B6 of
``docs/plans/2026-10-03-session-6a-spatial-selection-spec.md``.

``scripts/benchmark_spatial_patch_selection.py`` refuses to run in ``real`` mode without a
trained spatial backbone, because every downstream regret number computed from a randomly
initialised dynamics model would be an artefact of that initialisation rather than a
measurement of spatial selection. The existing 12-site checkpoint is a *pooled* ResNet18,
and a backbone trained at ``P = 32`` evaluated at ``k = 4`` is off-distribution. This script
therefore trains the spatial backbone on the multi-site DROID shard with random patch
dropout over the per-camera budgets, so the evaluated budgets are in-distribution.

The training objective is deliberately *the same objective the benchmark minimises*
(:func:`adjointrwm.models.common.prediction_objective` through the same
:func:`adjointrwm.spatial_selection.apply_patch_mask` a deployed selector uses), so the
backbone cannot be good at a surrogate while the benchmark scores a different quantity.

Protocol notes, because they decide whether the result means anything:

* **Splits.** Train and checkpoint selection use the cached manifest's ``train`` and ``val``
  splits only. The ``test`` split is never read here; it belongs to the benchmark.
* **Checkpoint selection.** The saved backbone is the one with the lowest validation
  objective *averaged over the evaluated budgets*, measured against a fixed set of mask
  draws so every checkpoint is scored on identical masks (a paired comparison). Selecting on
  the hardest budget alone would make the easy budgets look worse than they are.
* **The no-dropout endpoint.** ``k_cam = patches_per_camera`` is inside the training budget
  distribution, and its validation objective is tracked separately, so any degradation of the
  full-patch case that Session 2's result depends on is visible in the report rather than
  hidden.

Usage::

    python scripts/train_spatial_patch_dropout_backbone.py \\
        --cache-dir /content/cache_e3_1 \\
        --drive-root /content/drive/MyDrive/Colab\\ Notebooks/AdjointRWM_Production \\
        --run-id spatial_patches_20261003
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import torch

REPO_DIR = Path(__file__).resolve().parent.parent
for candidate in (REPO_DIR / "src", Path("/content/para_001/src"), Path("/content/src")):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

sys.path.insert(0, str(REPO_DIR / "scripts"))

from adjointrwm.io import atomic_write_json  # noqa: E402
from adjointrwm.models.common import prediction_objective, variance_covariance_regularizer  # noqa: E402
from adjointrwm.spatial_selection import (  # noqa: E402
    BUDGETS,
    CAMERAS,
    PATCHES_PER_CAMERA,
    apply_patch_mask,
    camera_of_patch,
    sample_budget_masks,
)

from benchmark_spatial_patch_selection import (  # noqa: E402
    build_spatial_model,
    load_spatial_splits,
    move_to_device,
)

#: Per-camera budgets scored during validation: the three Session 6A budgets plus the
#: no-dropout endpoint. ``P_cam`` is tracked so a regression on the full-patch case shows up.
VALIDATION_BUDGETS: tuple[int, ...] = tuple(sorted({k_cam for _total, k_cam in BUDGETS} | {PATCHES_PER_CAMERA}))

#: ``spatial_adjoint_rwm`` recipe default in ``models/registry.py``.
GRAD_CLIP = 1.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the Session 6A patch-dropout spatial backbone (spec B6)")
    parser.add_argument("--cache-dir", type=str, default="/content/cache_e3_1")
    parser.add_argument("--drive-root", type=str, default="/content/drive/MyDrive/Colab Notebooks/AdjointRWM_Production")
    parser.add_argument("--run-id", type=str, default="spatial_patches_20261003",
                        help="checkpoint lands at <drive-root>/runs/<run-id>/best_spatial.pt")
    parser.add_argument("--output-dir", type=str, default="results/benchmarks/spatial_backbone")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--patches-per-camera", type=int, default=PATCHES_PER_CAMERA)
    parser.add_argument("--var-regularizer", type=float, default=0.01,
                        help="weight on the anti-collapse latent regularizer (pilot stage-1 loss)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers; the shard is .npz-IO-bound (Session 0)")
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=8, help="validation batches per checkpoint evaluation")
    parser.add_argument("--val-mask-draws", type=int, default=4,
                        help="fixed mask draws per budget; identical across checkpoints so selection is paired")
    parser.add_argument("--train-budgets", type=str, default="2,4,8,16",
                        help="per-camera patch-dropout budgets to sample from")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------


def fixed_budget_masks(
    windows: int,
    k_cam: int,
    patches_per_camera: int,
    cameras: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """A fixed ``[B, P]`` mask set at one per-camera budget.

    Used for validation so that every checkpoint is scored on the *same* masks: selection is
    then a paired comparison and cannot be won by a checkpoint that happened to see easier
    masks.
    """
    generator = torch.Generator().manual_seed(seed)
    mask = sample_budget_masks(
        windows, patches_per_camera * cameras, patches_per_camera,
        k_choices=(k_cam,), cameras=cameras, generator=generator,
    )
    return mask.to(device)


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def masked_objective(
    model,
    batch: Mapping[str, torch.Tensor],
    mask: torch.Tensor,
    var_weight: float,
    need_latent: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``J`` under a patch mask, exactly as the benchmark computes it.

    Shares :func:`apply_patch_mask` and :func:`prediction_objective` with
    ``adjointrwm.spatial_selection.objective_at_masks``, so the trainer and the evaluator cannot
    disagree about what a budget means.
    """
    latent = model.encode_context(
        apply_patch_mask(batch["context_visual"], mask),
        batch["context_state"],
        batch["context_action"],
    )
    prediction = model.rollout(latent, batch["future_actions"])
    objective = prediction_objective(prediction, batch["target_state"], batch["target_visual"])
    if var_weight <= 0:
        return objective.mean(), latent if need_latent else None
    total = objective.mean() + var_weight * variance_covariance_regularizer(latent)
    return total, latent if need_latent else None


@torch.no_grad()
def validate(model, batches: Sequence[Mapping[str, torch.Tensor]], patches_per_camera: int,
             cameras: int, mask_draws: int, seed: int, device: torch.device) -> Dict[str, float]:
    """Validation objective per budget, averaged over fixed mask draws.

    Paired across checkpoints: the generators are seeded from ``seed`` and the budget, so the
    same masks are drawn every time.
    """
    model.eval()
    out: Dict[str, float] = {}
    for k_cam in VALIDATION_BUDGETS:
        totals = []
        for draw in range(mask_draws):
            per_batch = []
            for batch in batches:
                mask = fixed_budget_masks(
                    batch["context_state"].shape[0], k_cam, patches_per_camera, cameras,
                    seed + 1000 * k_cam + draw, device,
                )
                objective, _ = masked_objective(model, batch, mask, 0.0, False)
                per_batch.append(float(objective))
            totals.append(sum(per_batch) / len(per_batch))
        out[f"k_cam={k_cam}"] = sum(totals) / len(totals)
    model.train()
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(args, device: torch.device) -> dict:
    patches_per_camera = args.patches_per_camera
    patches = patches_per_camera * CAMERAS
    token_dim = 384
    train_budgets = tuple(int(k) for k in args.train_budgets.split(",") if k.strip())

    torch.manual_seed(args.seed)
    train_set, val_set, _test_set, site_by_episode = load_spatial_splits(
        Path(args.cache_dir), patches_per_camera
    )
    # The test split is loaded by the shared data path but deliberately never iterated here:
    # it belongs to the benchmark, not to this prerequisite.
    del _test_set

    from torch.utils.data import DataLoader

    # Batches stay on the host and are moved per step. Materialising the whole split on the
    # GPU would hold every patch token of every window at once (400 episodes x 32 patches x 384
    # dims x 8 context frames), which is tens of GiB and starves the model of memory. The
    # dataset is IO-bound on .npz reads, so workers are enabled for the same reason Session 0
    # measured it.
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_set, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    def cycle(loader):
        while True:
            for batch in loader:
                yield move_to_device(batch, device)

    train_stream = cycle(train_loader)
    if len(train_loader) == 0:
        raise SystemExit("no training batches: the cache yielded no train windows")
    # A fixed, bounded set of validation batches, so validation cost does not grow with the
    # split and every checkpoint is scored on the same windows.
    val_batches = [move_to_device(b, device) for b in val_loader][: max(1, args.val_batches)]
    if not val_batches:
        raise SystemExit("no validation batches: the cache yielded no val windows")

    model = build_spatial_model(args, patches_per_camera, device, args.width, token_dim)
    model.train()
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(1, args.steps))

    generator = torch.Generator().manual_seed(args.seed + 7)
    history: List[dict] = []
    best = {"score": math.inf, "step": -1, "val": None, "state": None}
    started = time.time()

    for step in range(1, args.steps + 1):
        batch = next(train_stream)
        mask = sample_budget_masks(
            batch["context_state"].shape[0], patches, patches_per_camera,
            k_choices=train_budgets, cameras=CAMERAS, generator=generator, device=device,
        )
        optimiser.zero_grad(set_to_none=True)
        loss, _ = masked_objective(model, batch, mask, args.var_regularizer, False)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimiser.step()
        schedule.step()

        if step % args.val_every == 0 or step == args.steps:
            val = validate(model, val_batches, patches_per_camera, CAMERAS,
                           args.val_mask_draws, args.seed, device)
            # Selection on the mean over the evaluated budgets, not on the hardest one.
            evaluated = [v for k, v in val.items() if k != f"k_cam={PATCHES_PER_CAMERA}"]
            score = sum(evaluated) / len(evaluated)
            row = {
                "step": step,
                "train_loss": float(loss.detach()),
                "grad_norm": float(grad_norm),
                "lr": schedule.get_last_lr()[0],
                "val": val,
                "selection_score": score,
                "elapsed_s": round(time.time() - started, 1),
            }
            history.append(row)
            print(
                f"step {step:5d} | train {float(loss):.4f} | "
                + " ".join(f"{k.split('=')[1]}:{v:.4f}" for k, v in val.items())
                + f" | score {score:.4f}",
                flush=True,
            )
            if score < best["score"]:
                best = {
                    "score": score,
                    "step": step,
                    "val": val,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                }

    if best["state"] is None:
        raise SystemExit("no checkpoint was selected: validation never ran")

    run_dir = Path(args.drive_root) / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best_spatial.pt"
    checkpoint = {
        "model_state_dict": best["state"],
        # Spec B6. The benchmark refuses to treat a checkpoint without this as patch-dropout
        # trained, so it is written by this script and nowhere else.
        "patch_dropout_trained": True,
        "patch_dropout_budgets_per_camera": list(train_budgets),
        "spec": "docs/plans/2026-10-03-session-6a-spatial-selection-spec.md#B6",
        "run_id": args.run_id,
        "selected_step": best["step"],
        "selection_score": best["score"],
        "validation_objective": best["val"],
        "config": {
            "steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
            "width": args.width, "weight_decay": args.weight_decay,
            "var_regularizer": args.var_regularizer, "seed": args.seed,
            "patches_per_camera": patches_per_camera, "grad_clip": GRAD_CLIP,
        },
    }
    torch.save(checkpoint, checkpoint_path)

    summary = {
        "mode": "real",
        "device": str(device),
        "spec": "docs/plans/2026-10-03-session-6a-spatial-selection-spec.md#B6",
        "run_id": args.run_id,
        "checkpoint_path": str(checkpoint_path),
        "patch_dropout_trained": True,
        "train_budgets_per_camera": list(train_budgets),
        "validation_budgets_per_camera": list(VALIDATION_BUDGETS),
        "selection_rule": "lowest mean validation objective over the evaluated budgets "
                          "(k_cam in {2,4,8}), paired over fixed mask draws; the k_cam=16 "
                          "no-dropout endpoint is tracked but not selected on",
        "selected_step": best["step"],
        "selection_score": best["score"],
        "validation_objective_at_best": best["val"],
        "train_windows": len(train_set),
        "val_windows": len(val_set),
        "val_batches_scored": len(val_batches),
        "sites": len({s for s in site_by_episode.values()}),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "history": history,
        "final_validation": history[-1]["val"] if history else None,
        "caveats": [
            "Test split is never read by this script; selection uses the manifest's val split.",
            "The benchmark trains its own heads on top of this frozen backbone per seed.",
            "Regret in Session 6A is measured against a greedy reference, not a proven optimum.",
        ],
    }
    return summary


def main(argv=None) -> int:
    args = parse_args(argv)
    print("=" * 70)
    print("Session 6A prerequisite: patch-dropout spatial backbone (spec B6)")
    print("=" * 70)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    summary = train(args, device)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "spatial_backbone_summary.json", summary)
    print(f"Saved summary to {output_dir / 'spatial_backbone_summary.json'}")
    print(f"Checkpoint: {summary['checkpoint_path']} (step {summary['selected_step']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())