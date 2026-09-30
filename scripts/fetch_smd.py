#!/usr/bin/env python3
"""Download Server Machine Dataset machines for D1-0 (tuning and validation splits only).

    python scripts/fetch_smd.py --root ~/adjointrwm_data/smd --splits tuning validation

Files go to ``<root>/{train,test,test_label}/<machine>.txt`` and ``<root>/manifest.json`` records the URL, size,
SHA-256 and retrieval time of every file. The test machines (index mod 4 of 2 or 3) are refused by
``adjointrwm.domains.sensor.download_machine``; the data are kept out of git (MIT licence, register id ``smd``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adjointrwm.domains import sensor as sn  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path.home() / "adjointrwm_data" / "smd")
    parser.add_argument("--splits", nargs="+", default=["tuning"], choices=["tuning", "validation"])
    args = parser.parse_args()
    rows = []
    for split in args.splits:
        for machine in sn.split_machines(split):
            rows += sn.download_machine(args.root, machine)
            print(f"{machine} ({split}): {sum(r['bytes'] for r in rows if r['machine'] == machine):,} bytes")
    manifest = args.root / "manifest.json"
    known = {(r["machine"], r["kind"]): r for r in (json.loads(manifest.read_text()) if manifest.exists() else [])}
    known.update({(r["machine"], r["kind"]): r for r in rows})
    manifest.write_text(json.dumps(list(known.values()), indent=1))
    print(f"manifest: {manifest} ({len(known)} files)")


if __name__ == "__main__":
    main()
