"""DROID 500-episode stratified shard streamer and metadata parser (Milestone E3.1).

Codified in docs/plans/roadmap.md §3 (Milestone E3.1).
Stratifies full DROID RLDS (95,658 episodes) by site (lab), scene, and task,
producing a clean 500-episode benchmark shard with strict data contracts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np

from .droid import (
    IMAGE_KEYS,
    STATE_KEYS,
    choose_episode_id,
    decode_text,
    extract_episode,
    extract_state,
    nested_value,
    validate_image,
)


@dataclass(frozen=True)
class EpisodeMetadata:
    episode_id: str
    site: str
    scene: str
    task_instruction: str
    num_steps: int
    split: str  # 'train' | 'val' | 'test'
    recording_folderpath: str
    file_path: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_metadata_from_rlds(episode: Mapping, episode_index: int = 0) -> Tuple[str, str, str, str, str]:
    """Parse site, scene, file_path, recording_folderpath, and instruction from RLDS episode."""
    meta = episode.get("episode_metadata", {})
    rf = decode_text(meta.get("recording_folderpath", b""))
    fp = decode_text(meta.get("file_path", b""))

    # Site extraction from path: e.g. gs://xembodiment_data/r2d2/r2d2-data-full/BVL/success/...
    # Parts: [..., 'r2d2-data-full', '<SITE>', '<STATUS>', '<DATE>', '<SCENE_DIR>', ...]
    site = "UNKNOWN"
    scene = "UNKNOWN"

    parts = rf.replace("\\", "/").split("/")
    if "r2d2-data-full" in parts:
        idx = parts.index("r2d2-data-full")
        if idx + 1 < len(parts):
            site = parts[idx + 1]
        if idx + 4 < len(parts):
            scene = f"{parts[idx+3]}_{parts[idx+4]}"
    elif len(parts) >= 6:
        site = parts[4]
        scene = parts[5] if len(parts) > 5 else "scene_0"

    steps = episode.get("steps", [])
    instruction = ""
    # Extract language instruction
    if hasattr(steps, "__iter__"):
        try:
            for s in steps:
                if "language_instruction" in s:
                    text = decode_text(s["language_instruction"])
                    if text.strip():
                        instruction = text.strip()
                        break
        except Exception:
            pass

    ep_id = choose_episode_id(episode, episode_index)
    return ep_id, site, scene, instruction, rf


def stratify_episodes(
    metadata_list: List[Dict[str, Any]],
    target_count: int = 500,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> List[EpisodeMetadata]:
    """Stratify episodes across sites and scenes into train, val, and test splits.

    Ensures proportional site representation, exact target counts,
    and structured split assignments.
    """
    if len(metadata_list) < target_count:
        target_count = len(metadata_list)

    rng = np.random.default_rng(seed)

    # Group by site
    by_site: Dict[str, List[Dict[str, Any]]] = {}
    for item in metadata_list:
        site = item.get("site", "UNKNOWN")
        by_site.setdefault(site, []).append(item)

    sites = sorted(by_site.keys())
    total_available = len(metadata_list)

    # Proportional quotas per site using largest remainder method
    quotas: Dict[str, int] = {}
    remainders: List[Tuple[float, str]] = []
    allocated = 0
    for site in sites:
        exact_quota = target_count * (len(by_site[site]) / total_available)
        base = int(np.floor(exact_quota))
        if base == 0 and len(by_site[site]) > 0:
            base = 1  # Ensure at least 1 episode per present site
        quotas[site] = base
        allocated += base
        remainders.append((exact_quota - base, site))

    # Adjust to match target_count exactly
    remainders.sort(key=lambda x: x[0], reverse=True)
    while allocated < target_count:
        for _, site in remainders:
            if quotas[site] < len(by_site[site]) and allocated < target_count:
                quotas[site] += 1
                allocated += 1
    while allocated > target_count:
        for _, site in reversed(remainders):
            if quotas[site] > 1 and allocated > target_count:
                quotas[site] -= 1
                allocated -= 1

    selected_records: List[Dict[str, Any]] = []
    for site in sites:
        site_eps = by_site[site]
        perm = rng.permutation(len(site_eps))
        for idx in perm[:quotas[site]]:
            selected_records.append(site_eps[idx])

    # Now assign splits deterministically across selected records
    # Group by scene to keep scenes together where possible
    by_scene: Dict[str, List[Dict[str, Any]]] = {}
    for item in selected_records:
        scene_key = f"{item.get('site', '')}_{item.get('scene', '')}"
        by_scene.setdefault(scene_key, []).append(item)

    scene_keys = sorted(by_scene.keys())
    rng.shuffle(scene_keys)

    target_train = int(round(target_count * train_ratio))
    target_val = int(round(target_count * val_ratio))
    target_test = target_count - target_train - target_val

    train_eps: List[Dict[str, Any]] = []
    val_eps: List[Dict[str, Any]] = []
    test_eps: List[Dict[str, Any]] = []

    for sk in scene_keys:
        eps = by_scene[sk]
        if len(train_eps) + len(eps) <= target_train:
            train_eps.extend(eps)
        elif len(val_eps) + len(eps) <= target_val:
            val_eps.extend(eps)
        elif len(test_eps) + len(eps) <= target_test:
            test_eps.extend(eps)
        else:
            # Distribute individually to fill quotas
            for ep in eps:
                if len(train_eps) < target_train:
                    train_eps.append(ep)
                elif len(val_eps) < target_val:
                    val_eps.append(ep)
                else:
                    test_eps.append(ep)

    # Build final EpisodeMetadata list
    selected: List[EpisodeMetadata] = []
    for split_name, split_list in [("train", train_eps), ("val", val_eps), ("test", test_eps)]:
        for ep_dict in split_list:
            selected.append(
                EpisodeMetadata(
                    episode_id=ep_dict["episode_id"],
                    site=ep_dict["site"],
                    scene=ep_dict["scene"],
                    task_instruction=ep_dict.get("instruction", ""),
                    num_steps=ep_dict.get("num_steps", 0),
                    split=split_name,
                    recording_folderpath=ep_dict.get("recording_folderpath", ""),
                    file_path=ep_dict.get("file_path", ""),
                )
            )

    return selected
