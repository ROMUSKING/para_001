---
name: audit-run
description: Decide whether a training run, checkpoint, notebook or archive counts as scientific evidence for AdjointRWM, and write a dated audit. Use when asked to review, evaluate, audit or "check" a run, a Colab notebook, a checkpoint, or claims of success from another agent or chatbot.
---

# Audit a run or notebook

**Default stance:** nothing counts as evidence until it passes every check below. Past audits in `docs/audits/` show which failures turn up in practice. Read at least `docs/audits/README.md` before starting.

## Checklist

For each item, record PASS / FAIL / NOT ASSESSABLE and give the file and line (or cell) as evidence.

**Data**
- [ ] Real, publisher-sourced data with episode IDs, timestamps and real observation payloads, not hash strings.
- [ ] No seeded-random or synthetic substitutes. Test for this: can the arrays be regenerated from `np.random.seed(k)`? Is the lag-1 autocorrelation about 0?
- [ ] No relabelling of unrelated tabular data as states or actions (the Ailerons case).
- [ ] The split is by episode, scene or task **before** windowing, and split disjointness is verified.

**Objective**
- [ ] Targets are in the future (`t+1…t+H`), not the current step.
- [ ] Target encoders are frozen or stop-gradient, not trained jointly as moving targets.
- [ ] The co-state is supervised by ∂J/∂state or directional gain, not by actions, zeros or constants.
- [ ] The critic is trained on counterfactual marginal gain, and the gate on measured net benefit.

**Metrics**
- [ ] No hard-coded values. Grep for literal assignments to metric names (`regret =`, `success =`, `mse =`).
- [ ] No method-specific multipliers or constants on shared losses (the `×1.15 / ×0.95` case).
- [ ] Baselines are present: persistence and linear for dynamics; random (expected), best-fixed, oracle and direct critic for allocation.
- [ ] Held-out evaluation, with per-horizon and per-episode breakdowns where relevant.
- [ ] Seeds are reported, and there's a CI (episode-cluster bootstrap: `adjointrwm.analysis.episode_bootstrap_ci`).

**Reproducibility**
- [ ] The checkpoint includes model, optimizer, scheduler, scaler, sampler, RNG and hashes, and every trainable module has weights *and* optimizer state.
- [ ] Config, data-manifest and source hashes are recorded. Re-verify the config hash (see the `import-run` skill).
- [ ] A resume-equivalence test was run, not just a reload.

**Claims**
- [ ] Every claim in the run summary maps to a computed artefact. Hardware recommendations follow `docs/production/colab_l4_operator_brief.md`.

## Output

- **Write the audit** to `docs/audits/YYYY-MM-DD_<subject>_audit.md` with:
  - a classification block (e.g. `PIPELINE_SMOKE_TEST_PASSED / REAL_DATA_TRAINING_FAILED`);
  - the checklist table;
  - the hashes of what was audited;
  - an explicit "do not reuse for" list.
- **Register it:** add a row to `docs/audits/README.md`.
- **If the run *does* pass:** say so plainly, then use the `research-note` skill for the findings.

## Tone

- Be specific and cite evidence. "Cell 86 assigns `task_success_rate = 94.1`" beats "metrics look suspicious".
- Separate what failed from what can be salvaged. Pipeline code often can be; weights trained on bad data can't.
