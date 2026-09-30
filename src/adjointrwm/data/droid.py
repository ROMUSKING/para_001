"""DROID RLDS episode parsing (lifted from section 2 of the production pilot).

The pure helpers here work on the nested dicts that ``tfds.as_numpy`` yields, so they are
tested without TensorFlow. :func:`load_droid` is the only function that imports
``tensorflow_datasets``; it raises instead of falling back to anything synthetic.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

STATE_KEYS = ("cartesian_position", "gripper_position", "joint_position")
IMAGE_KEYS = ("exterior_image_1_left", "wrist_image_left")


def numpy_value(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def decode_text(value: Any) -> str:
    value = numpy_value(value)
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def nested_value(mapping: Any, path: Sequence[str]) -> Any:
    current = mapping
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def choose_episode_id(episode: Mapping, episode_index: int) -> str:
    """The pilot's episode id: sha256 of the recording/file paths, first 24 hex chars."""
    candidate_paths = [
        ("episode_metadata", "recording_folderpath"),
        ("episode_metadata", "file_path"),
        ("traj_metadata", "episode_metadata", "recording_folderpath"),
        ("traj_metadata", "episode_metadata", "file_path"),
    ]
    pieces = [decode_text(v) for p in candidate_paths if (v := nested_value(episode, p)) is not None]
    if not pieces:
        pieces = [f"droid100_episode_{episode_index:04d}"]
    return hashlib.sha256("|".join(pieces).encode()).hexdigest()[:24]


def extract_state(observation: Mapping, keys: Sequence[str] = STATE_KEYS) -> tuple[np.ndarray, list, list]:
    """Concatenate the proprio fields present; return ``(state, used_keys, dims_per_key)``."""
    parts, used, dims = [], [], []
    for key in keys:
        if key in observation:
            value = np.asarray(numpy_value(observation[key]), dtype=np.float32).reshape(-1)
            parts.append(value)
            used.append(key)
            dims.append(int(value.shape[0]))
    if not parts:
        raise KeyError(f"No supported DROID state fields were present. Observation keys: {list(observation.keys())}")
    return np.concatenate(parts), used, dims


def state_groups(keys: Sequence[str], dims: Sequence[int]) -> dict[str, list[int]]:
    """Column indices of each proprio group in the concatenated state vector."""
    groups, offset = {}, 0
    for key, dim in zip(keys, dims):
        groups[key] = list(range(offset, offset + int(dim)))
        offset += int(dim)
    return groups


def validate_image(image: Any, name: str) -> np.ndarray:
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Real image payload {name} has invalid rank {image.ndim}.")
    if image.var() == 0:
        raise ValueError(f"Real image payload {name} is constant.")
    return image


def extract_episode(
    steps: Sequence[Mapping],
    frame_stride: int = 2,
    image_keys: Sequence[str] = IMAGE_KEYS,
    state_keys: Sequence[str] = STATE_KEYS,
) -> dict:
    """States, actions, images and instruction of one episode, every ``frame_stride``-th step.

    Raises if a required image is missing or constant: no hash-only or synthetic replacement.
    """
    states, actions, instructions = [], [], []
    images = {key: [] for key in image_keys}
    contract = None
    for step_index, step in enumerate(steps):
        if step_index % frame_stride:
            continue
        observation = step["observation"]
        missing = [key for key in image_keys if key not in observation]
        if missing:
            raise KeyError(f"Required real image payloads are missing: {missing}. No hash-only replacement is allowed.")
        for key in image_keys:
            images[key].append(validate_image(observation[key], key))
        state, used, dims = extract_state(observation, state_keys)
        action = np.asarray(numpy_value(step["action"]), dtype=np.float32).reshape(-1)
        signature = (tuple(used), tuple(dims), action.shape[0])
        if contract is None:
            contract = signature
        elif signature != contract:
            raise ValueError(f"state/action contract changed within an episode: {signature} != {contract}")
        states.append(state)
        actions.append(action)
        instructions.append(decode_text(step.get("language_instruction", b"")))
    if not states:
        raise ValueError("episode has no steps")
    return {
        "states": np.stack(states).astype(np.float32),
        "actions": np.stack(actions).astype(np.float32),
        "images": images,
        "instruction": next((text for text in instructions if text.strip()), ""),
        "state_keys": list(contract[0]),
        "state_dims": list(contract[1]),
        "action_dim": int(contract[2]),
    }


def load_droid(dataset_name: str = "droid_100", data_dir: str = "gs://gresearch/robotics"):
    """Open the official DROID RLDS release. Never falls back to other data."""
    import tensorflow as tf  # noqa: PLC0415 - Colab-only dependency
    import tensorflow_datasets as tfds  # noqa: PLC0415

    tf.config.set_visible_devices([], "GPU")  # keep TensorFlow off the training GPU
    if not hasattr(tfds, "load"):
        # tensorflow_datasets wraps its whole import in try/except and only logs the error, so a broken dependency (for example a
        # protobuf runtime older than the generated code of tensorflow_metadata) leaves a module without ``load``. Re-raise the real error.
        import importlib  # noqa: PLC0415

        importlib.import_module("tensorflow_datasets.public_api")
        raise RuntimeError("tensorflow_datasets was imported without its public API (no `load`); see the log for the swallowed import error.")
    try:
        return tfds.load(dataset_name, data_dir=data_dir, split="train", shuffle_files=False, with_info=True)
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(
            f"Unable to load the official DROID {dataset_name} RLDS dataset from {data_dir}. "
            "No synthetic or unrelated fallback is permitted."
        ) from error
