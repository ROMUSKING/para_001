#!/usr/bin/env python3
"""scripts/materialise_spatial_normaliser.py

WS4 provenance: write the fitted train-split state/action normaliser to a committed JSON file.

Background (``prereg/deviation_log.yaml`` DEV-20261004-02): on a fresh Colab runtime without
human Drive authorisation, the E3.1 cached normaliser is unreachable, so
``load_spatial_dataset`` fits the normaliser in memory from the manifest's **train** episodes
only (``scripts/benchmark_spatial_patch_selection.py:190``). That fitted object was never
persisted, so no later reader could reproduce the exact normalisation. This script performs
the identical fit and writes it as JSON (lists, float32-rounded) plus a SHA-256, so the
normalisation is pinned without depending on Drive.

Usage::

    python scripts/materialise_spatial_normaliser.py \\
        --cache-dir /content/spatial_cache_e3_1 \\
        --output /content/local_runs/runs/spatial_patches_20261003/spatial_normaliser.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parent.parent
for candidate in (REPO_DIR / "src", Path("/content/para_001/src")):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from adjointrwm.data import fit_normaliser  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Persist the fitted train-split normaliser")
    parser.add_argument("--cache-dir", type=str, default="/content/spatial_cache_e3_1")
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cache_dir = Path(args.cache_dir)
    manifest = json.loads((cache_dir / "cache_manifest.json").read_text())
    states, actions = [], []
    n_train = 0
    for episode in manifest.get("episodes", []):
        if episode.get("split", "train") != "train":
            continue
        path = cache_dir / "episodes" / f"{episode['episode_id']}.npz"
        with np.load(path) as archive:
            states.append(np.asarray(archive["states"], dtype=np.float32))
            actions.append(np.asarray(archive["actions"], dtype=np.float32))
        n_train += 1
    if not states:
        raise SystemExit("no train episodes found; refusing to write an empty normaliser")
    normalisation = fit_normaliser(states, actions)
    payload = {
        "fitted_on": "manifest train split only (DEV-20261004-02)",
        "n_train_episodes": n_train,
        "state_mean": [float(v) for v in normalisation["state_mean"]],
        "state_std": [float(v) for v in normalisation["state_std"]],
        "action_mean": [float(v) for v in normalisation["action_mean"]],
        "action_std": [float(v) for v in normalisation["action_std"]],
    }
    body = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    payload["sha256"] = hashlib.sha256(body.encode()).hexdigest()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out} from {n_train} train episodes sha256={payload['sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
