"""Tests for DROID shard streaming and stratification (Milestone E3.1)."""

from __future__ import annotations

import pytest

from adjointrwm.data.droid_shard import (
    EpisodeMetadata,
    parse_metadata_from_rlds,
    stratify_episodes,
)


def test_parse_metadata_from_rlds():
    mock_episode = {
        "episode_metadata": {
            "recording_folderpath": "gs://xembodiment_data/r2d2/r2d2-data-full/BVL/success/2023-08-01/scene_kitchen_table/recordings/MP4",
            "file_path": "droid/trajectory.tfrecord",
        },
        "steps": [
            {"language_instruction": b"pick up the red mug"},
            {"language_instruction": b"pick up the red mug"},
        ],
    }

    ep_id, site, scene, instruction, rf = parse_metadata_from_rlds(mock_episode, 0)
    assert site == "BVL"
    assert "2023-08-01" in scene
    assert "scene_kitchen_table" in scene
    assert instruction == "pick up the red mug"
    assert len(ep_id) == 24
    assert rf == mock_episode["episode_metadata"]["recording_folderpath"]


def test_stratify_episodes():
    # Construct 100 mock episodes across 4 sites
    sites = ["BVL", "AUTOLab", "ILIAD", "TRI"]
    metadata_list = []
    for i in range(100):
        site = sites[i % 4]
        metadata_list.append(
            {
                "episode_id": f"ep_{i:04d}",
                "site": site,
                "scene": f"{site}_scene_{i // 10}",
                "instruction": f"do task {i}",
                "num_steps": 50,
                "recording_folderpath": f"gs://test/{site}/{i}",
                "file_path": f"path/{i}",
            }
        )

    stratified = stratify_episodes(
        metadata_list,
        target_count=40,
        train_ratio=0.80,
        val_ratio=0.10,
        test_ratio=0.10,
        seed=42,
    )

    assert len(stratified) == 40
    assert all(isinstance(x, EpisodeMetadata) for x in stratified)

    # Check site representation
    sites_in_strat = {x.site for x in stratified}
    assert sites_in_strat == set(sites)

    # Check split assignments
    splits = {x.split for x in stratified}
    assert "train" in splits
    assert "val" in splits or "test" in splits


def test_stratify_reproducibility():
    metadata_list = [
        {
            "episode_id": f"ep_{i:04d}",
            "site": "BVL",
            "scene": "scene_1",
            "instruction": "task",
            "num_steps": 20,
        }
        for i in range(50)
    ]

    strat1 = stratify_episodes(metadata_list, target_count=20, seed=123)
    strat2 = stratify_episodes(metadata_list, target_count=20, seed=123)

    assert [x.episode_id for x in strat1] == [x.episode_id for x in strat2]
    assert [x.split for x in strat1] == [x.split for x in strat2]
