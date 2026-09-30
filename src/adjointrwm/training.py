"""Resumable training and evaluation loop with the operator brief's checkpoint contract.

Lifted and generalised from section 4 of the production pilot (sampler, RNG capture,
warmup-cosine schedule, checkpoint payload). One call trains one (arm, seed) job:

* recovery checkpoints (``latest.pt``) hold model, optimiser, scheduler, sampler position and
  every RNG state, so a job pre-empted by Colab resumes on the exact next batch
  (``tests/test_training.py`` checks resume equivalence, roadmap N0.3);
* files are written locally, then copied atomically to the persistent (Drive) job directory
  with a ``.sha256`` sidecar that is verified before any reload;
* a resumed job must carry the same run id, arm, seed, config, data-manifest and source hashes,
  otherwise it stops instead of mixing two configurations;
* ``DONE.json`` marks a finished job; re-running the notebook skips it.
"""

from __future__ import annotations

import contextlib
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from .eval.prediction import horizon_mean_rmse, per_window_mse
from .io import append_jsonl, atomic_copy, atomic_write_json, sha256_file

IDENTITY_KEYS = ("run_id", "arm", "seed", "config_hash", "data_manifest_hash", "source_hash")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


class StatefulBatchSampler(Sampler):
    """The pilot's resumable batch sampler: epoch-seeded permutation plus a cursor."""

    def __init__(self, length, batch_size, seed):
        self.length = int(length)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.cursor = 0
        self.permutation = None
        self._regenerate()

    def _regenerate(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        self.permutation = torch.randperm(self.length, generator=generator).tolist()
        self.cursor = 0

    def __iter__(self):
        while self.cursor < self.length:
            end = min(self.cursor + self.batch_size, self.length)
            batch = self.permutation[self.cursor : end]
            self.cursor = end
            yield batch
        self.epoch += 1
        self._regenerate()

    def __len__(self):
        return math.ceil((self.length - self.cursor) / self.batch_size)

    def state_dict(self):
        return {
            "length": self.length,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "permutation": self.permutation,
        }

    def load_state_dict(self, state):
        assert state["length"] == self.length
        assert state["batch_size"] == self.batch_size
        self.seed = int(state["seed"])
        self.epoch = int(state["epoch"])
        self.cursor = int(state["cursor"])
        self.permutation = list(state["permutation"])


def warmup_cosine(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Batches and prediction
# ---------------------------------------------------------------------------

def move_batch(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def autocast(device, enabled: bool = True):
    device = torch.device(device)
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


class PermutedActions:
    """Dataset view in which window ``i`` gets the future actions of window ``perm[i]``."""

    def __init__(self, dataset, permutation):
        if sorted(permutation) != list(range(len(dataset))):
            raise ValueError("permutation must cover every window exactly once")
        self.dataset, self.permutation = dataset, list(permutation)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item = dict(self.dataset[i])
        item["future_actions"] = self.dataset[self.permutation[i]]["future_actions"]
        return item


def _fork_rng(device):
    device = torch.device(device)
    devices = [device.index or 0] if device.type == "cuda" else []
    return torch.random.fork_rng(devices=devices)


@torch.no_grad()
def predict_dataset(model, dataset, batch_size: int, device, eval_seed: int = 0, amp: bool = True, with_latents: bool = False) -> dict:
    """Predictions for every window, in dataset order, as float64 NumPy arrays.

    Sampling arms (RSSM) draw from a forked RNG seeded with ``eval_seed``, so evaluation is
    reproducible and does not disturb the training RNG stream.
    """
    was_training = model.training
    model.eval()
    keep = {k: [] for k in ("state_mean", "state_logvar", "visual", "target_state", "target_visual")}
    latents, episode_ids, starts = [], [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    with _fork_rng(device):
        torch.manual_seed(eval_seed)
        for batch in loader:
            batch = move_batch(batch, device)
            with autocast(device, amp):
                out = model.predict(batch)
                latent = model.latent_for_diagnostics(batch) if with_latents else None
            for key in ("state_mean", "state_logvar", "visual"):
                keep[key].append(out[key].float().cpu())
            keep["target_state"].append(batch["target_state"].float().cpu())
            keep["target_visual"].append(batch["target_visual"].float().cpu())
            if latent is not None:
                latents.append(latent.float().cpu())
            episode_ids += list(batch["episode_id"])
            starts += [int(x) for x in batch["window_start"]]
    model.train(was_training)
    result = {k: torch.cat(v).double().numpy() for k, v in keep.items()}
    result["episode_id"] = np.asarray(episode_ids)
    result["window_start"] = np.asarray(starts)
    if latents:
        result["latent"] = torch.cat(latents).double().numpy()
    return result


def validation_score(model, dataset, batch_size, device, amp=True) -> float:
    pred = predict_dataset(model, dataset, batch_size, device, amp=amp)
    return horizon_mean_rmse(per_window_mse(pred["state_mean"], pred["target_state"]))


def measure_latency_ms(function: Callable[[], object], device, warmup: int = 10, iterations: int = 50) -> dict:
    device = torch.device(device)
    for _ in range(warmup):
        function()
    timings = []
    for _ in range(iterations):
        if device.type == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            function()
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            function()
            timings.append(1000 * (time.perf_counter() - t0))
    return {
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
        "p99_ms": float(np.percentile(timings, 99)),
        "mean_ms": float(np.mean(timings)),
        "device": str(device),
    }


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def save_checkpoint(payload: dict, local_path, persist_path, keep_local: bool = False) -> dict:
    """Write locally, copy atomically to ``persist_path`` (+ ``.sha256``), drop the local copy."""
    local_path, persist_path = Path(local_path), Path(persist_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, local_path)
    digest = atomic_copy(local_path, persist_path)
    if not keep_local:
        local_path.unlink()
    return {"path": str(persist_path), "sha256": digest}


def _remove_checkpoint(path: Path) -> None:
    for candidate in (path, path.with_suffix(path.suffix + ".sha256")):
        if candidate.exists():
            candidate.unlink()


def load_checkpoint(path, map_location="cpu") -> dict:
    """Load a checkpoint written by :func:`save_checkpoint` after verifying its SHA-256.

    ``weights_only=False`` is needed for the NumPy/Python RNG states; it is only used on files
    whose hash matches the sidecar written in the same run directory.
    """
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.exists():
        raise FileNotFoundError(f"missing hash sidecar for {path}")
    expected = sidecar.read_text().strip()
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"checkpoint {path} failed its integrity check ({actual} != {expected})")
    return torch.load(path, map_location=map_location, weights_only=False)


def load_best_weights(model, persist_dir, identity: dict | None = None) -> dict:
    """Load ``best.pt`` of a job into ``model`` (hash-verified); returns the payload metadata."""
    payload = load_checkpoint(Path(persist_dir) / "best.pt")
    if identity is not None:
        mismatched = {k: (payload.get(k), identity[k]) for k in IDENTITY_KEYS if payload.get(k) != identity[k]}
        if mismatched:
            raise RuntimeError(f"best.pt belongs to a different job: {mismatched}")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return {k: payload[k] for k in (*IDENTITY_KEYS, "global_step", "best_metrics") if k in payload}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainConfig:
    steps: int = 1500
    batch_size: int = 64
    lr: float = 3.0e-4
    weight_decay: float = 0.05
    warmup_steps: int = 100
    grad_clip: float = 1.0
    eval_interval: int = 100
    checkpoint_interval: int = 250
    log_interval: int = 25
    seed: int = 20260928
    amp: bool = True
    eval_batch_size: int = 128

    def to_dict(self) -> dict:
        return asdict(self)


def train_job(
    model,
    train_dataset,
    val_dataset,
    cfg: TrainConfig,
    *,
    identity: dict,
    local_dir,
    persist_dir,
    device="cpu",
    resume: bool = True,
    max_steps_this_session: int | None = None,
    log: Callable[[dict], None] | None = None,
    score_fn: Callable[[object], float] | None = None,
    score_name: str = "validation_rmse",
    stage: str = "world_model",
    keep_best: bool = True,
) -> dict:
    """Train one job (an arm and seed, or an allocator head); returns its summary.

    ``model`` needs ``training_loss(batch)`` and ``param_groups(lr)``; it may define
    ``clip_gradients(max_norm)`` to clip parameter groups separately. ``identity`` must contain
    :data:`IDENTITY_KEYS`. The model must already be built with the job's seed (so a fresh
    start is reproducible); on resume its weights are replaced. The best checkpoint is chosen
    by ``score_fn(model)`` on validation (lower is better; default: state RMSE).
    ``max_steps_this_session`` stops early with status ``PAUSED`` after a recovery checkpoint,
    which is how a long benchmark is spread over several Colab sessions. The summary is also
    written to ``DONE.json``.

    Storage: the recovery checkpoint ``latest.pt`` is deleted once the job is DONE (operator
    brief: keep the best checkpoints plus a rotating recovery window). ``keep_best=False``
    also deletes ``best.pt`` (for search trials whose weights are never reused); its SHA-256
    stays in ``DONE.json``.
    """
    missing = [k for k in IDENTITY_KEYS if k not in identity]
    if missing:
        raise ValueError(f"identity is missing {missing}")
    local_dir, persist_dir = Path(local_dir), Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    done_path = persist_dir / "DONE.json"
    if resume and done_path.exists():
        import json  # noqa: PLC0415

        return json.loads(done_path.read_text())

    device = torch.device(device)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.param_groups(cfg.lr), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: warmup_cosine(s, cfg.warmup_steps, cfg.steps)
    )
    sampler = StatefulBatchSampler(len(train_dataset), cfg.batch_size, cfg.seed)
    # A dedicated generator: every iter(DataLoader) draws a worker base seed, and drawing it
    # from the global RNG would shift dropout noise after a resume (caught by the N0.3 test).
    loader = DataLoader(
        train_dataset, batch_sampler=sampler, num_workers=0, pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    if score_fn is None:
        score_fn = lambda m: validation_score(m, val_dataset, cfg.eval_batch_size, device, cfg.amp)  # noqa: E731
    step, best = 0, {"score_name": score_name, "validation_score": float("inf"), "step": None, "checkpoint": None}
    latest = persist_dir / "latest.pt"
    if resume and latest.exists():
        payload = load_checkpoint(latest)
        mismatched = {k: (payload.get(k), identity[k]) for k in IDENTITY_KEYS if payload.get(k) != identity[k]}
        if mismatched:
            raise RuntimeError(f"refusing to resume {latest}: identity differs {mismatched}")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        sampler.load_state_dict(payload["sampler_state"])
        restore_rng_state(payload["rng_state"])
        step, best = int(payload["global_step"]), payload["best_metrics"]

    def payload_for(global_step):
        return {
            "schema_version": 3,
            **{k: identity[k] for k in IDENTITY_KEYS},
            "stage": stage,
            "global_step": int(global_step),
            "epoch": sampler.epoch,
            "model_state_dict": model.state_dict(),
            "critic_state_dict": None,  # world-model jobs have no separate critic
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": None,  # BF16 autocast needs no loss scaler
            "sampler_state": sampler.state_dict(),
            "rng_state": capture_rng_state(),
            "train_config": cfg.to_dict(),
            "best_metrics": best,
        }

    iterator = iter(loader)
    session_start, session_steps = time.perf_counter(), 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    status = "RUNNING"
    while step < cfg.steps:
        step += 1
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, cfg.amp):
            loss, parts = model.training_loss(move_batch(batch, device))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step} ({identity['arm']}, seed {identity['seed']})")
        loss.backward()
        if hasattr(model, "clip_gradients"):
            grad_norm = model.clip_gradients(cfg.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient at step {step} ({identity['arm']}, seed {identity['seed']})")
        optimizer.step()
        scheduler.step()
        session_steps += 1

        record = None
        if step == 1 or step % cfg.log_interval == 0:
            record = {"stage": "train", "step": step, "loss": float(loss.detach()), "grad_norm": float(grad_norm),
                      "lr": optimizer.param_groups[-1]["lr"], **{k: float(v) for k, v in parts.items()}}
        if step % cfg.eval_interval == 0 or step == cfg.steps:
            score = float(score_fn(model))
            model.train()
            record = {**(record or {"stage": "train", "step": step}), score_name: score}
            if score < best["validation_score"]:
                ckpt = save_checkpoint(
                    {**payload_for(step), "optimizer_state_dict": None, "scheduler_state_dict": None},
                    local_dir / "best.pt", persist_dir / "best.pt",
                )
                best = {"score_name": score_name, "validation_score": float(score), "step": step, "checkpoint": ckpt}
        if record is not None:
            append_jsonl(local_dir / "train_log.jsonl", record)
            if log:
                log(record)
        stop_now = max_steps_this_session is not None and session_steps >= max_steps_this_session
        if step % cfg.checkpoint_interval == 0 or step == cfg.steps or stop_now:
            save_checkpoint(payload_for(step), local_dir / "latest.pt", latest)
            if (local_dir / "train_log.jsonl").exists():
                atomic_copy(local_dir / "train_log.jsonl", persist_dir / "train_log.jsonl")
        if stop_now and step < cfg.steps:
            status = "PAUSED"
            break

    elapsed = time.perf_counter() - session_start
    summary = {
        **{k: identity[k] for k in IDENTITY_KEYS},
        "status": "DONE" if step >= cfg.steps else status,
        "global_step": step,
        "best": best,
        "session_steps": session_steps,
        "session_seconds": elapsed,
        "session_steps_per_second": session_steps / elapsed if elapsed > 0 else None,
        "peak_vram_gib": (torch.cuda.max_memory_allocated(device) / 1024**3) if device.type == "cuda" else None,
        "train_config": cfg.to_dict(),
    }
    if summary["status"] == "DONE":
        if best["checkpoint"] is None:
            raise RuntimeError("training finished without a validated checkpoint")
        summary["best_checkpoint_kept"] = keep_best
        atomic_write_json(done_path, summary)
        _remove_checkpoint(latest)
        if not keep_best:
            _remove_checkpoint(persist_dir / "best.pt")
    return summary


train_world_model = train_job  # the world-model benchmark's name for the same loop
