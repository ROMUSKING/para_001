# Licence survey, pass 2 (milestone DL)

**Date:** 2026-09-29 (after network access was widened) · **Register:** [`register.csv`](register.csv) (59 rows) · **Evidence:** [`evidence.jsonl`](evidence.jsonl) (124 URLs, 114 fetched, 10 expected 404s) · **Probes:** [`probes.jsonl`](probes.jsonl) (14, 13 parsed; the failure is `hf_longbench_v2`, HTTP 501) · **Tools:** [`scripts/licence_survey.py`](../../scripts/licence_survey.py), [`tests/test_licences.py`](../../tests/test_licences.py) (16 tests) · **Follows:** [pass 1](survey-2026-09-29.md) · **Rules:** [`README.md`](README.md)

**Status:** the blocked hosts are reachable now (Hugging Face API, datasets server and pages; UCI; Zenodo; NASA; IMDb; Kaggle pages; Google's Gemma pages; the CIC page). The Hugging Face file CDN still does not answer, so **no dataset file was downloaded from Hugging Face**: model and dataset cards were read, and dataset structure was probed through the datasets server. TPC's terms were not resolved.

**In one paragraph.** The register now has 59 rows: **17 `adopt`, 12 `adopt_with_conditions`, 5 `avoid`, 25 `unverified`**. For D2, several frozen models with Apache-2.0 or MIT licences declared by their owners fit the plan (Qwen3-8B and Phi-4-mini have their `LICENSE` file read; Gemma 4 is Apache-2.0 where Gemma 3 was custom), and MuSiQue, Qasper and LongBench v2 are permissive QA sets. For D1, SMD, NAB, the UCI electricity load and the Monash archive records are permissive. For D3, the query-planning route is closed by licence (JOB's IMDb data is non-commercial; STATS-CEB has no licence), while code-repository context is open: 490 of the 500 SWE-bench Verified instances come from repositories whose licence file is BSD, MIT, Apache or PSF-style, but the dataset states no licence of its own and its issue texts have none. Two owner statements disagree with each other (TriviaQA, LongBench v1) and are recorded as such.

---

## 1. What changed since pass 1

- **Read now:** 11 Hugging Face model cards and their gating, the licence files of Qwen3-8B, Qwen3-4B and Phi-4-mini, 11 Hugging Face dataset records (eight with their README text), UCI's dataset page, 3 pages of Zenodo records, NASA's data portal page, IMDb's dataset terms page, Google's Gemma terms and its Gemma 4 licence page, Kaggle's dataset page, and the CIC-IDS2017 page.
- **New checks:** `snapshot` now records the licence fields inside JSON answers and in HTML catalogue markup, the final URL after redirects, and Hugging Face gating, commit and parameter count. `probe` now reads the structure of Hugging Face datasets through the datasets server and runs a licence census over the repositories behind SWE-bench Verified.
- **Model and dataset IDs** were resolved through the Hugging Face search API rather than from memory. Redirects were recorded: `THUDM/LongBench` is now `zai-org/LongBench`.

## 2. Results

| Domain | `adopt` | `adopt_with_conditions` | `avoid` | `unverified` |
|---|---:|---:|---:|---:|
| D1 sensor streams | 4 (SMD, NAB, UCI electricity, Monash archive) | 2 (ETT, SKAB) | 1 | 4 |
| D2 frozen models and QA data | 12 | 8 | 1 | 9 |
| D3 code-repository context | 0 | 2 (CrossCodeEval, CodeSearchNet) | 0 | 6 |
| D3 alternative: query planning | 0 | 0 | 2 (JOB, STATS-CEB) | 2 |
| Tier 2: cloud traces | 1 (Google) | 0 | 1 (Alibaba) | 1 |
| Tier 2: ranking, security | 0 | 0 | 0 | 3 |

`adopt` means no licence obstacle was found in what was read (README, "Verdicts"), not legal advice and not scientific fit. Of the 25 `unverified` rows, 13 have only a code licence read, 11 have an owner statement that does not settle the licence, and 1 (TPC-H/TPC-DS) is unread.

### 2.1 D2: frozen model for the LLM-context domain

The plan requires a frozen open-weights LLM that fits an L4 in BF16 with a recorded licence. Weights-alone sizes are `2 bytes × safetensors parameter count` from `evidence.jsonl` (an L4 has 24 GB; that is NVIDIA's specification, not a number from this repository, and activations and KV cache come on top; nothing here was run).

| Checkpoint | Declared licence (Hugging Face card) | Licence file read | Parameters | BF16 weights | Verdict |
|---|---|---|---:|---:|---|
| Qwen/Qwen3-8B | apache-2.0 | yes (Apache 2.0 text) | 8,190,735,360 | 16.38 GB | `adopt` |
| Qwen/Qwen3-4B | apache-2.0 | yes | 4,022,468,096 | 8.04 GB | `adopt` |
| microsoft/Phi-4-mini-instruct | mit | yes (MIT text) | 3,836,021,760 | 7.67 GB | `adopt` |
| google/gemma-4-E4B-it | apache-2.0 | no file in the listing; Google's Gemma 4 licence page is titled "Apache License 2.0" | 7,996,156,490 | 15.99 GB | `adopt` |
| ibm-granite/granite-3.3-8b-instruct | apache-2.0 | no | 8,170,864,640 | 16.34 GB | `adopt` |
| allenai/OLMo-2-1124-7B | apache-2.0 | no | 7,298,617,344 | 14.60 GB | `adopt` |
| mistralai/Mistral-7B-v0.3 | apache-2.0 | no | 7,248,023,552 | 14.50 GB | `adopt` |
| HuggingFaceTB/SmolLM3-3B | apache-2.0 | no | 3,075,098,624 | 6.15 GB | `adopt` |
| google/gemma-4-12B-it | apache-2.0 | no | 11,959,730,224 | 23.92 GB | `adopt`, but the weights alone leave almost no room on an L4 |
| google/gemma-3-4b-it | gemma (custom), gated | terms page read | 4,300,079,472 | 8.60 GB | `adopt_with_conditions` |
| meta-llama/Llama-3.1-8B-Instruct | llama3.1 (custom), gated | text read from the `llama-models` repo | 8,030,261,248 | 16.06 GB | `adopt_with_conditions` |

- **Gemma 3 versus Gemma 4.** Gemma 3 is under the custom Gemma Terms of Use: distribution must carry the use restrictions of the Prohibited Use Policy, a copy of the agreement, modification notices and a Notice file (Section 3.1, 3.2 of the page read), and access is gated. Google's page says Gemma 4 has its own licence, and Gemma 4 checkpoints declare Apache 2.0 and are not gated.
- The model card metadata is the owner's declaration. Only the Qwen3 and Phi-4-mini checkpoints had a licence file in their Hugging Face listing that could be read; Llama's licence text was read from Meta's `llama-models` repository.
- Whether a gradient through the model fits in memory (the D2 co-state is a gradient through the frozen LLM) was not measured.

### 2.2 D2: question-answering data

| Dataset | Licence as read | Structure probe (columns of a few rows) | Verdict |
|---|---|---|---|
| MuSiQue | CC BY 4.0 (README and `LICENSE` of the official repo) | `paragraphs`, `question_decomposition`, `answerable`, `answer_aliases`, `question`, `answer` (third-party mirror) | `adopt` |
| Qasper | CC BY 4.0 (`allenai` card, metadata and text) | `full_text`, `qas`, `abstract`, `title`, `figures_and_tables` | `adopt` |
| LongBench v2 | apache-2.0 (card metadata only) | none: the datasets server returned HTTP 501 | `adopt` |
| HotpotQA | CC BY-SA 4.0 (GitHub README and card) | `context`, `supporting_facts`, `question`, `answer`, `level`, `type` | `adopt_with_conditions` (share-alike) |
| NarrativeQA | Apache-2.0 (card); scripts come from imsdb and other repositories | not probed | `adopt_with_conditions` |
| QuALITY | article licences in the data; **annotations unstated** (mirror cards state none) | zip read in pass 1 | `adopt_with_conditions` |
| TriviaQA | GitHub README: Apache 2.0 for code and data; card: licence "unknown" and "The University of Washington does not own the copyright of the questions and documents" | not probed | `unverified` |
| LongBench v1 | card states no licence | not probed | `avoid` |

TriviaQA and LongBench v1 are the two places where owner statements disagree or say nothing; they stay out until the owners clarify.

### 2.3 D1: sensor and time-series data

| Dataset | Licence as read | Verdict |
|---|---|---|
| SMD | MIT (repo `LICENSE`); 38-column files and a separate label file (pass 1 probes) | `adopt` |
| NAB | MIT; univariate in the file probed | `adopt` |
| UCI electricity load diagrams | UCI page: "licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license" | `adopt` |
| Monash forecasting archive (Zenodo) | 65 of the 66 records returned by a creator search carry a licence: 63 `cc-by-4.0` and 2 `cc-by-3.0` | `adopt` (one record has none; the search may include records outside the archive) |
| ETT | CC BY-ND 4.0 | `adopt_with_conditions` (no derivatives) |
| SKAB | GPL-3.0 | `adopt_with_conditions` (copyleft) |
| NASA C-MAPSS | NASA portal page: "License not specified" | `unverified` |
| SMAP/MSL (Kaggle) | Kaggle page: "Data files © Original Authors", which is not a licence | `unverified` |

### 2.4 D3: which graph domain (open question 2)

- **Query planning is closed by licence.** The Join Order Benchmark's data is the IMDb dataset, and IMDb's page says subsets are "available … for personal and non-commercial use" (`avoid`). STATS-CEB has no licence (`avoid`). DSB is MIT code derived from TPC-DS whose terms were not resolved (`unverified`).
- **Code-repository context is open.** A census of the repositories behind SWE-bench Verified (probe `swebench_repos`, 500 instances, 12 repositories) matched their licence files, by text pattern (a heuristic, not SPDX detection), to: BSD-3-Clause 288 instances, MIT 94, BSD-2-Clause 44, PSF-style (matplotlib) 34, Apache-2.0 30 and GPL 10 (pylint). So **490 of 500 instances come from repositories with a permissive-family licence file and 10 from a GPL repository**. But the dataset's Hugging Face cards state no licence, and issue texts (`problem_statement`, `hints_text`) are user contributions with no stated licence, so the dataset stays `unverified`. RepoBench's card says only `cc`. CrossCodeEval (Apache-2.0, third-party code inside) and CodeSearchNet (MIT, per-file licences in a file that was not fetched) remain `adopt_with_conditions`. The Stack v2 is gated (licence "other") and unread.
- **Recommendation:** choose code-repository context (the plan's default) for D3. The decision about SWE-bench Verified rests on whether Roman accepts the repository licence as the effective licence for patches and treats issue text as out of scope; that is a policy call, not a finding.

### 2.5 Tier 2

- **Cloud traces:** Google's cluster traces are CC-BY 4.0 by their README (`adopt`); Alibaba's have no licence (`avoid`); Azure's repository licence is CC BY 4.0 but its README does not say it covers the traces (`unverified`).
- **Security:** the CIC-IDS2017 page says the data is "publicly available for researchers" and names no licence (`unverified`).
- **Ranking:** BEIR and MS MARCO remain `unverified` (per-dataset licences; MS MARCO's terms page was not read).

## 3. What this does and does not support

**Supports:**

- The licence strings and owner statements quoted above, as read on 2026-09-29 and pinned by URL and SHA-256 in `evidence.jsonl` (for Hugging Face, also the repository commit).
- The SWE-bench Verified repository census as a heuristic count (`guess_licence` is unit-tested but is pattern matching).
- Parameter counts and the weights-alone sizes derived from them.
- That the verdicts follow the rubric (`tests/test_licences.py`).

**Does not support:**

- That a model fits, trains or back-propagates on an L4: only weights-alone sizes were computed.
- Legal advice, or any statement about content that the owner statement does not cover (the source documents of QA sets, issue texts, third-party code, source series of time-series sets).
- Scientific fit of any dataset. A structure probe shows columns, not suitability.
- Anything about sources that were not surveyed.

## 4. Recommendation

1. **D2 (LLM context):** Qwen3-8B (or Qwen3-4B for cheaper runs) as the frozen model; MuSiQue and Qasper as first QA sets, with HotpotQA as a share-alike option and QuALITY held until its annotation licence is clarified.
2. **D1 (sensor streams):** SMD first, with the UCI electricity load and the Monash records as forecasting sets, and NAB as a univariate control.
3. **D3:** code-repository context, with SWE-bench Verified subject to the policy decision in §2.4.
4. **These are licence recommendations.** Each still needs a domain card (`docs/plans/cross-domain-plan.md` §2), an opportunity gate and a fit check before any run; none has been built. The decisions to confirm are the model, the datasets and the D3 domain.
5. **Open licence gaps worth closing:** QuALITY's annotation licence (ask the authors), TriviaQA's and LongBench v1's statements, the licence files of the Apache-declared checkpoints that had none in their listing, and TPC's terms.

## 5. Commands

The register counts, evidence and probe files are regenerated by `python scripts/licence_survey.py snapshot` and `probe` (README). The weights-alone sizes:

```bash
python - <<'EOF'
import json
for line in open("docs/licences/evidence.jsonl"):
    h = json.loads(line)["hf"]
    if h and h["safetensors_total_params"]:
        n = h["safetensors_total_params"]
        print(f"{h['id']:40s} {n:>14,d} {2 * n / 1e9:7.2f} GB")
EOF
```
