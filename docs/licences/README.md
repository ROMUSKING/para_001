# Licence register (milestone DL)

What this folder is: a checkable record of the licence and release terms of every dataset and frozen model the cross-domain track might use ([`docs/plans/cross-domain-plan.md`](../plans/cross-domain-plan.md), §8: "Every dataset and every frozen model gets a licence record before the first run"). The findings are in the dated survey notes, the first being [`survey-2026-09-29.md`](survey-2026-09-29.md).

| File | Role |
|---|---|
| `register.csv` | One row per source: the licence **as read**, what it covers, whether it is permissive, the verification level, the evidence URLs, the gaps, and a verdict. Edited by hand. |
| `evidence.jsonl` | One line per evidence URL: HTTP status, bytes, SHA-256, retrieval time and up to four licence-relevant lines. Written by `python scripts/licence_survey.py snapshot`. No licence texts are stored. |
| `probes.jsonl` | Structure-only results of small usability tests on data that is reachable (column names, row counts, label counts, per-article licence classes). Written by `python scripts/licence_survey.py probe`. No data content is stored. |

`tests/test_licences.py` fails when a row claims more than its evidence shows.

## Definitions

- **Permissive** (`yes`): MIT, BSD, Apache-2.0, CC0, CC BY, the Project Gutenberg licence, or an equivalent, for the asset itself.
- **Conditions** (`conditions`): the asset can be used but with obligations that matter here: share-alike (CC BY-SA), no derivatives (CC BY-ND), copyleft (GPL), custom licences with attribution, use-policy or user-count clauses (Llama), or mixed per-item licences.
- **No** (`no`): no licence stated (all rights reserved by default), or non-commercial or research-only terms without a licence.
- **Unknown** (`unknown`): the licence of *this asset* was not read.

**Code licence is not data licence.** A repository's `LICENSE` covers what the repository owner owns. It does not cover a dataset hosted elsewhere, third-party text inside a dataset, or model weights (operator brief: "Do not assume that a code repository license governs the data or assets").

## Verification levels

| Level | Meaning |
|---|---|
| `primary_text_read` | The licence text, or licence strings inside the data itself, were read and apply to the asset. |
| `owner_statement_read` | The owner's own README says which licence applies to the asset; the licence file of the asset itself was not read. |
| `code_only_read` | Only a licence for code was read. The data or weights licence is unread. |
| `no_licence_found` | The README was read and the usual licence file names return 404. |
| `unread` | Nothing was read: the host was unreachable or the source was not yet located. |

## Verdicts (enforced by the tests)

| Verdict | Requires |
|---|---|
| `adopt` | `permissive = yes` and a read level (`primary_text_read` or `owner_statement_read`). Residual unknowns are listed under `gaps`. |
| `adopt_with_conditions` | `permissive` is `yes` or `conditions`, a read level, and the conditions written down. |
| `avoid` | `permissive = no`. Revisit only if the owner clarifies. |
| `unverified` | Everything else, including every `unknown` licence and every row where only code or nothing was read. |

A verdict is about the licence, not about scientific fit. `adopt` means "no licence obstacle found in what was read", not "legal advice" and not "this is the right benchmark".

## Refreshing

```bash
python scripts/licence_survey.py snapshot   # re-fetch every evidence URL, rewrite evidence.jsonl
python scripts/licence_survey.py probe      # re-run the usability probes, rewrite probes.jsonl
pytest -q tests/test_licences.py
```

Run both where the blocked hosts are reachable (Colab, or after widening the sandbox's network access) to fill in the `unread` rows. Then update `register.csv` by hand from what the snapshot records, and never mark a row as read unless its URLs were fetched.
