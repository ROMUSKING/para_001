#!/usr/bin/env python3
"""Design-time probe for D4-1 (docs/plans/d4-1-plan.md §10): can a small generic MLP learn the co-state operator ``Lambda(t; c) = Phi(t) c``?

Synthetic rows only: a random right-node index and a random unit goal per row, labelled with the exact propagator. No allocation policy,
no tuning or validation instance is involved, so the numbers are **not evidence about allocation**; they set the estimator's training budget
and the expectation for what a generic estimator can do. Usage: ``python scripts/d4_1_estimator_probe.py m4 m64``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adjointrwm.domains import critics as cr  # noqa: E402
from adjointrwm.domains import highdim as hd  # noqa: E402


def main(cells: list[str], rows: int = 48_000) -> None:
    for cell in cells:
        inst = hd.varying_goal_instances(cell, "train", 1)[0]
        rng = np.random.default_rng(0)
        nodes = rng.integers(1, inst.n_fine + 1, size=rows)
        goal = cr.random_unit_goals(rng, rows, inst.m)
        lam = cr.continuous_costate(inst, nodes, goal)
        data = cr.Rows(np.zeros((rows, hd.FEATURE_DIM)), goal, np.zeros((rows, inst.m)), np.zeros((rows, inst.m)), nodes / inst.n_fine, lam,
                       np.zeros(rows), rng.integers(0, 30, size=rows), np.arange(rows) // 20)
        mask = cr.split_by_group(data.group, 0)
        for hidden, epochs, batch in ((32, 40, 1024), (32, 200, 256), (64, 200, 256)):
            started = time.perf_counter()
            est = cr.train_estimator(data.take(~mask), data.take(mask), hidden=hidden, seed=0, max_epochs=epochs, patience=epochs, batch_size=batch)
            print(f"{cell}: hidden {hidden}, epochs {epochs}, batch {batch}: relative error against the exact co-state, "
                  f"validation {est.report['val_relative_error_against_true_costate']:.3f}, train {est.report['train_relative_error_against_true_costate']:.3f} "
                  f"({time.perf_counter() - started:.0f} s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["m4", "m64"])
