# Documentation index

| Folder | What's inside | Start with |
|---|---|---|
| [`plans/`](plans/) | Roadmap (tracks, milestones, gates, open questions), the rival-model benchmark plan, the cross-domain plan, and the agent work log | [`roadmap.md`](plans/roadmap.md), then [`rival-benchmark-plan.md`](plans/rival-benchmark-plan.md) and [`cross-domain-plan.md`](plans/cross-domain-plan.md) |
| [`research-plan/`](research-plan/) | Preregistration-style research protocols | [`adjoint_guided_comprehensive_research_plan.md`](research-plan/adjoint_guided_comprehensive_research_plan.md) (governing) |
| [`production/`](production/) | Production training plan and the Colab L4 operator rules | [`colab_l4_operator_brief.md`](production/colab_l4_operator_brief.md) (read before running anything) |
| [`research-notes/`](research-notes/) | Dated findings from runs that count as evidence | [`2026-09-29-droid100-pilot-findings.md`](research-notes/2026-09-29-droid100-pilot-findings.md) |
| [`audits/`](audits/) | Checks of runs and notebooks that don't count as evidence, and why | [`README.md`](audits/README.md) |
| [`DRIVE_INVENTORY.csv`](DRIVE_INVENTORY.csv) | Every relevant Google Drive file: ID, size, status, repo path | — |

## Writing conventions

- **Research notes** are named `YYYY-MM-DD-<slug>.md`. Each one names its run ID and config hash and states which gate passed or failed.
- **Audits** are named `YYYY-MM-DD_<subject>_audit.md` and are append-only.
- **Protocol changes** after a confirmatory run starts go in `prereg/deviation_log.yaml` (research plan, "Preregistered numerical margins"), not in edits to the plan.
- **Claims:** a sentence claiming an effect must link to the committed run directory that supports it.
