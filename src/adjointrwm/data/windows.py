"""Episode-level splits, temporal windows and normalisation (NumPy only).

Lifted from sections 2 and 3 of ``notebooks/01-production/AdjointRWM_Production_Pilot.ipynb``.
The rules are reproduced exactly, so a notebook that uses them on the same episodes gets
the same split, windows and normalisation as run ``droid100_adjoint_20260929T070629Z``:

* **Split before windowing.** Episodes are ordered by ``sha256(episode_id)``; the first
  ``int(0.8 n)`` are train, the next ``int(0.1 n)`` validation, the rest test.
* **Window alignment.** ``action[t]`` drives the transition ``state[t] -> state[t+1]``.
  A window starting at ``s`` with context ``T`` and horizon ``H`` has

  - ``context_state  = state[s : s+T]``
  - ``context_action = [0, action[s : s+T-1]]`` (the action that led *into* each context step)
  - ``future_actions = action[s+T-1 : s+T-1+H]`` (actions applied from the last context step on)
  - ``target_state   = state[s+T : s+T+H]`` (strictly after every context step)

* **Normalisation** statistics come from train episodes only (float64 accumulation,
  std floored at 1e-6).

Everything a deployed model may see is in ``context_state``, ``context_action``,
``context_visual`` and ``future_actions``. ``target_*``, ``future_visual`` and
``context_target_visual`` are training/evaluation targets only.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Iterable, Mapping, Sequence

import numpy as np

SPLITS = ("train", "validation", "test")


# ---------------------------------------------------------------------------
# Episode split
# ---------------------------------------------------------------------------

def split_order_key(episode_id: str) -> str:
    return hashlib.sha256(episode_id.encode()).hexdigest()


def episode_split(
    episode_ids: Iterable[str], train_fraction: float = 0.80, val_fraction: float = 0.10
) -> dict[str, str]:
    """Deterministic episode -> split assignment (the pilot's rule, byte for byte)."""
    ids = list(episode_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("episode ids must be unique before splitting")
    if not ids:
        raise ValueError("no episodes to split")
    ordered = sorted(ids, key=split_order_key)
    n_total = len(ordered)
    n_train = max(1, int(train_fraction * n_total))
    n_val = max(1, int(val_fraction * n_total))
    assignment = {}
    for index, episode_id in enumerate(ordered):
        if index < n_train:
            assignment[episode_id] = "train"
        elif index < n_train + n_val:
            assignment[episode_id] = "validation"
        else:
            assignment[episode_id] = "test"
    return assignment


def normalize_split_name(split: str) -> str:
    """Normalize split aliases (e.g. 'val' -> 'validation')."""
    if split == "val":
        return "validation"
    return split


def split_sets(assignment: Mapping[str, str]) -> dict[str, set]:
    sets = {split: set() for split in SPLITS}
    for episode_id, raw_split in assignment.items():
        split = normalize_split_name(raw_split)
        if split not in sets:
            raise ValueError(f"unknown split {raw_split!r} for episode {episode_id}")
        sets[split].add(episode_id)
    return sets


def compare_splits(assignment: Mapping[str, str], reference: Mapping[str, str]) -> dict:
    """Differences between two assignments, e.g. a new run versus the pilot's data manifest."""
    norm_assignment = {e: normalize_split_name(s) for e, s in assignment.items()}
    norm_reference = {e: normalize_split_name(s) for e, s in reference.items()}
    shared = set(norm_assignment) & set(norm_reference)
    return {
        "matches": not (set(norm_assignment) ^ set(norm_reference)) and all(norm_assignment[e] == norm_reference[e] for e in shared),
        "only_in_new": sorted(set(norm_assignment) - set(norm_reference)),
        "only_in_reference": sorted(set(norm_reference) - set(norm_assignment)),
        "different_split": sorted(e for e in shared if norm_assignment[e] != norm_reference[e]),
    }


def assignment_from_manifest(manifest: Mapping) -> dict[str, str]:
    """Read ``{episode_id: split}`` from a pilot-style ``data_manifest.json``."""
    return {row["episode_id"]: normalize_split_name(row["split"]) for row in manifest["episodes"]}


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WindowSpec:
    context_len: int = 8
    horizon: int = 4
    stride: int = 2

    def to_dict(self) -> dict:
        return asdict(self)


def window_starts(length: int, spec: WindowSpec) -> list[int]:
    maximum_start = length - spec.context_len - spec.horizon
    return list(range(0, maximum_start + 1, spec.stride)) if maximum_start >= 0 else []


def fit_normaliser(states: Sequence[np.ndarray], actions: Sequence[np.ndarray]) -> dict:
    """Train-split statistics (pass train episodes only)."""
    s = np.concatenate([np.asarray(x, dtype=np.float64) for x in states], axis=0)
    a = np.concatenate([np.asarray(x, dtype=np.float64) for x in actions], axis=0)
    return {
        "state_mean": s.mean(axis=0).astype(np.float32),
        "state_std": np.maximum(s.std(axis=0), 1e-6).astype(np.float32),
        "action_mean": a.mean(axis=0).astype(np.float32),
        "action_std": np.maximum(a.std(axis=0), 1e-6).astype(np.float32),
    }


def normalise_state(states: np.ndarray, normaliser: Mapping) -> np.ndarray:
    return (np.asarray(states, dtype=np.float32) - normaliser["state_mean"]) / normaliser["state_std"]


def normalise_action(actions: np.ndarray, normaliser: Mapping) -> np.ndarray:
    return (np.asarray(actions, dtype=np.float32) - normaliser["action_mean"]) / normaliser["action_std"]


def denormalise_state(states_norm: np.ndarray, normaliser: Mapping) -> np.ndarray:
    return np.asarray(states_norm, dtype=np.float32) * normaliser["state_std"] + normaliser["state_mean"]


def as_tokens(features: Sequence[np.ndarray]) -> np.ndarray:
    """Stack per-camera features into ``[L, P, D]`` tokens.

    Each input is ``[L, D]`` (one global vector per frame) or ``[L, P_i, D]``. Flattening the
    result over the last two axes gives the pilot's ``concatenate(axis=-1)`` layout, because
    tokens are stacked in key order.
    """
    parts = []
    for array in features:
        array = np.asarray(array)
        parts.append(array[:, None, :] if array.ndim == 2 else array)
    dims = {p.shape[-1] for p in parts}
    if len(dims) != 1:
        raise ValueError(f"token features must share their last dimension, got {sorted(dims)}")
    return np.concatenate(parts, axis=1)


def slice_window(
    states_norm: np.ndarray,
    actions_norm: np.ndarray,
    start: int,
    spec: WindowSpec,
    input_visual: np.ndarray | None = None,
    target_visual: np.ndarray | None = None,
    auxiliary: Mapping[str, np.ndarray] | None = None,
) -> dict:
    """One window with the pilot's alignment (see module docstring).

    ``auxiliary`` maps a name to a per-frame ``[L, ...]`` array that is *not* a model input,
    such as a cached encoder attention map. Each entry is emitted under its own name, sliced to
    the **context** frames only, because such arrays describe what the encoder saw while the
    decision was being made. They are windowed with exactly the same ``start`` as
    ``context_visual``, so a scorer reading them is reading decision-time information about the
    same frames the model conditions on -- never about the future targets.
    """
    T, H = spec.context_len, spec.horizon
    end = start + T
    if start < 0 or end + H > len(states_norm):
        raise IndexError(f"window [{start}, {end + H}) outside episode of length {len(states_norm)}")
    context_action = np.zeros((T, actions_norm.shape[-1]), dtype=np.float32)
    if T > 1:
        context_action[1:] = actions_norm[start : end - 1]
    out = {
        "context_state": np.ascontiguousarray(states_norm[start:end], dtype=np.float32),
        "context_action": context_action,
        "future_actions": np.ascontiguousarray(actions_norm[end - 1 : end - 1 + H], dtype=np.float32),
        "target_state": np.ascontiguousarray(states_norm[end : end + H], dtype=np.float32),
    }
    if input_visual is not None:
        out["context_visual"] = np.ascontiguousarray(input_visual[start:end], dtype=np.float32)
        out["future_visual"] = np.ascontiguousarray(input_visual[end : end + H], dtype=np.float32)
    if target_visual is not None:
        out["target_visual"] = np.ascontiguousarray(target_visual[end : end + H], dtype=np.float32)
        # The shared visual target over the context frames: observable at decision time, but
        # only used as a training target (reconstruction-style arms), never as a model input.
        out["context_target_visual"] = np.ascontiguousarray(target_visual[start:end], dtype=np.float32)
    for name, array in (auxiliary or {}).items():
        array = np.asarray(array)
        if array.shape[0] != len(states_norm):
            raise ValueError(
                f"auxiliary {name!r} has {array.shape[0]} frames but the episode has "
                f"{len(states_norm)}; auxiliary arrays must be per-frame and untrimmed"
            )
        out[name] = np.ascontiguousarray(array[start:end], dtype=np.float32)
    for key in ("future_actions", "target_state"):
        assert len(out[key]) == H, key
    return out


class WindowDataset:
    """Map-style dataset of windows over whole episodes (usable directly with a torch DataLoader).

    ``records`` are dicts with ``episode_id``, ``length`` and either ``arrays`` (a mapping of
    in-memory arrays) or ``cached_path`` (an ``.npz`` written by the extraction cell). Arrays
    must include ``states`` and ``actions`` plus the feature keys named here.

    ``visual_layout='flat'`` returns ``context_visual`` as ``[T, P*D]`` (the pilot layout);
    ``'tokens'`` returns ``[T, P, D]``. ``target_visual`` is always a flat global vector
    ``[H, V]`` built from ``target_visual_keys`` (the shared evaluation target).
    """

    def __init__(
        self,
        records: Sequence[Mapping],
        spec: WindowSpec,
        normaliser: Mapping,
        input_visual_keys: Sequence[str] = ("exterior_embeddings", "wrist_embeddings"),
        target_visual_keys: Sequence[str] | None = None,
        visual_layout: str = "flat",
        auxiliary_keys: Sequence[str] = (),
    ):
        if visual_layout not in ("flat", "tokens"):
            raise ValueError("visual_layout must be 'flat' or 'tokens'")
        self.records = list(records)
        self.spec = spec
        self.normaliser = normaliser
        self.input_visual_keys = tuple(input_visual_keys)
        self.target_visual_keys = tuple(target_visual_keys or input_visual_keys)
        self.visual_layout = visual_layout
        #: Per-frame arrays carried through to the window but never used as a model input,
        #: e.g. a cached encoder attention map that a Tier 0 saliency comparator reads.
        self.auxiliary_keys = tuple(auxiliary_keys)
        self.index = [
            (record_index, start)
            for record_index, record in enumerate(self.records)
            for start in window_starts(int(record["length"]), spec)
        ]
        if not self.index:
            raise RuntimeError("No temporal windows satisfy the contract.")
        self._load = lru_cache(maxsize=16)(self._load_uncached)

    def __len__(self) -> int:
        return len(self.index)

    def episode_ids(self) -> np.ndarray:
        return np.array([self.records[r]["episode_id"] for r, _ in self.index])

    def _load_uncached(self, record_index: int) -> dict:
        record = self.records[record_index]
        if "arrays" in record:
            arrays = record["arrays"]
        else:
            with np.load(record["cached_path"]) as episode:
                arrays = {key: episode[key].copy() for key in episode.files}
        states = normalise_state(arrays["states"], self.normaliser)
        actions = normalise_action(arrays["actions"], self.normaliser)
        tokens = as_tokens([np.asarray(arrays[k], dtype=np.float32) for k in self.input_visual_keys])
        if self.visual_layout == "flat":
            tokens = tokens.reshape(tokens.shape[0], -1)
        target = np.concatenate(
            [np.asarray(arrays[k], dtype=np.float32).reshape(len(states), -1) for k in self.target_visual_keys],
            axis=-1,
        )
        return {"states": states, "actions": actions, "input_visual": tokens, "target_visual": target}

    def _auxiliary(self, arrays: Mapping[str, np.ndarray]) -> dict:
        missing = [k for k in self.auxiliary_keys if k not in arrays]
        if missing:
            raise KeyError(
                f"auxiliary_keys {missing} are not in the cached episode; available: "
                f"{sorted(arrays)}"
            )
        return {k: np.asarray(arrays[k]) for k in self.auxiliary_keys}

    def __getitem__(self, item: int) -> dict:
        record_index, start = self.index[item]
        episode = self._load(record_index)
        record = self.records[record_index]
        if "arrays" in record:
            arrays = record["arrays"]
        else:
            with np.load(record["cached_path"]) as handle:
                arrays = {key: handle[key] for key in handle.files}
        window = slice_window(
            episode["states"], episode["actions"], start, self.spec,
            input_visual=episode["input_visual"], target_visual=episode["target_visual"],
            auxiliary=self._auxiliary(arrays),
        )
        window["episode_id"] = self.records[record_index]["episode_id"]
        window["window_start"] = start
        return window
