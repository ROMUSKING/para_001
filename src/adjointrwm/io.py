"""Hashing and atomic file writes for immutable run directories (standard library only).

Lifted from section 1 of ``notebooks/01-production/AdjointRWM_Production_Pilot.ipynb``.
The operator brief requires every artefact to be written locally first and then renamed
into place, with its SHA-256 recorded, so a pre-empted Colab session never leaves a
half-written file on Drive that looks complete.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


def stable_json_bytes(value: Any) -> bytes:
    """Canonical JSON encoding used for config, source and data-manifest hashes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(stable_json_bytes(value))


def sha256_file(path: str | os.PathLike, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: str | os.PathLike, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def atomic_write_json(path: str | os.PathLike, value: Any) -> Path:
    return atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, default=str))


def atomic_copy(local_path: str | os.PathLike, final_path: str | os.PathLike) -> str:
    """Copy to ``final_path`` via a temporary name, then write ``<final>.sha256``.

    Returns the SHA-256 of the copied file (computed on the destination).
    """
    local_path, final_path = Path(local_path), Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = final_path.with_suffix(final_path.suffix + ".tmp")
    shutil.copy2(local_path, tmp)
    os.replace(tmp, final_path)
    digest = sha256_file(final_path)
    atomic_write_text(final_path.with_suffix(final_path.suffix + ".sha256"), digest + "\n")
    return digest


def append_jsonl(path: str | os.PathLike, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, default=str) + "\n")


def file_record(path: str | os.PathLike) -> dict:
    """Path, size and hash of a file that must exist (operator brief rule 2)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"expected artefact is missing: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
