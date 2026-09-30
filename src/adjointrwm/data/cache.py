"""Build and verify the per-episode feature cache shared by every arm in a run.

Real DROID episodes are decoded once, embedded with every requested frozen encoder, and saved
as one ``.npz`` per episode. The cache (and its manifest with SHA-256 per file) lives in the
immutable run directory on Drive, so a resumed Colab session copies it to local disk instead of
re-streaming TFDS. Nothing is ever synthesised: a missing or failed episode raises or is
logged as skipped with its reason.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np

from ..io import atomic_write_json, sha256_file
from .droid import IMAGE_KEYS, choose_episode_id, extract_episode

# Feature-array name for each (encoder, camera image key).
CAMERA_SHORT = {"exterior_image_1_left": "exterior", "wrist_image_left": "wrist"}


def feature_key(encoder: str, image_key: str) -> str:
    """The pilot named ResNet-18 features ``exterior_embeddings``; other encoders get a suffix."""
    camera = CAMERA_SHORT.get(image_key, image_key)
    return f"{camera}_embeddings" if encoder == "resnet18" else f"{camera}_{encoder}"


def build_feature_cache(
    episodes: Iterable[tuple[int, Mapping, list]],
    embedders: Mapping[str, Callable],
    out_dir,
    *,
    frame_stride: int = 2,
    min_steps: int = 13,
    image_keys=IMAGE_KEYS,
    log: Callable[[str], None] = print,
    on_episode: Callable[[str, dict], None] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Decode, embed and cache episodes.

    ``episodes`` yields ``(episode_index, episode_mapping, steps)``, where ``steps`` is the list
    produced by ``tfds.as_numpy(episode['steps'])``. Returns ``(records, skipped)``; each record
    has ``episode_id``, ``length``, ``cached_path``, ``cached_sha256``, state contract and
    instruction. ``on_episode(episode_id, parsed)`` sees the decoded images (e.g. to plot the
    eight episodes the operator brief requires someone to inspect by eye).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records, skipped, contract = [], [], None
    for episode_index, episode, steps in episodes:
        if len(steps) < min_steps:
            skipped.append({"episode_index": episode_index, "reason": "EPISODE_TOO_SHORT", "length": len(steps)})
            continue
        parsed = extract_episode(steps, frame_stride=frame_stride, image_keys=image_keys)
        signature = (tuple(parsed["state_keys"]), tuple(parsed["state_dims"]), parsed["action_dim"])
        if contract is None:
            contract = signature
        elif signature != contract:
            raise ValueError(f"state/action contract differs between episodes: {signature} != {contract}")
        episode_id = choose_episode_id(episode, episode_index)
        if on_episode is not None:
            on_episode(episode_id, parsed)
        arrays = {"states": parsed["states"], "actions": parsed["actions"]}
        for encoder, embed in embedders.items():
            for image_key in image_keys:
                arrays[feature_key(encoder, image_key)] = embed(parsed["images"][image_key])
        path = out_dir / f"{episode_id}.npz"
        np.savez_compressed(path, **arrays)
        records.append({
            "episode_index": int(episode_index),
            "episode_id": episode_id,
            "cached_path": str(path),
            "cached_sha256": sha256_file(path),
            "length": int(len(parsed["states"])),
            "state_keys": parsed["state_keys"],
            "state_dims": parsed["state_dims"],
            "state_dim": int(parsed["states"].shape[-1]),
            "action_dim": int(parsed["actions"].shape[-1]),
            "instruction": parsed["instruction"],
            "feature_keys": sorted(k for k in arrays if k not in ("states", "actions")),
        })
        log(f"cached {len(records):4d}: {episode_id} ({len(parsed['states'])} frames)")
    return records, skipped


def write_cache_manifest(records: list[dict], skipped: list[dict], path, extra: Mapping | None = None) -> dict:
    manifest = {"episodes": records, "skipped": skipped, **(extra or {})}
    atomic_write_json(path, manifest)
    return manifest


def persist_cache(records: list[dict], persist_dir) -> list[dict]:
    """Copy cached episodes to the persistent run directory (once per run); returns new records."""
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for record in records:
        target = persist_dir / Path(record["cached_path"]).name
        if not target.exists():
            tmp = target.with_suffix(".tmp")
            shutil.copy2(record["cached_path"], tmp)
            tmp.replace(target)
        if sha256_file(target) != record["cached_sha256"]:
            raise ValueError(f"persisted cache file {target} does not match its hash")
        out.append({**record, "persisted_path": str(target)})
    return out


def restore_cache(manifest: Mapping, local_dir) -> list[dict]:
    """Copy a persisted cache to local disk, verifying every hash; returns records with local paths."""
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for record in manifest["episodes"]:
        source = Path(record["persisted_path"])
        target = local_dir / source.name
        if not target.exists() or sha256_file(target) != record["cached_sha256"]:
            shutil.copy2(source, target)
        if sha256_file(target) != record["cached_sha256"]:
            raise ValueError(f"cache file {source} failed its integrity check")
        records.append({**record, "cached_path": str(target)})
    return records
