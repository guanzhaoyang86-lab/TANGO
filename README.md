# TANGO

Time-series anomaly detection with grounded vision-language reasoning.
This repository contains the supplied TANGO software, corrected data adapters,
and an auditable **A / D / F / G** controlled-comparison pipeline.

**Status:** software release, not a release of trained checkpoints or reproduced
benchmark scores. No usable Stage-2 checkpoint was available during this release.
CPU tests include actual randomly initialized tiny Qwen2.5-VL models; those tests
do not establish pretrained-model accuracy, GPU compatibility, or RAGO's benefit.
See [verification scope](VERIFICATION.md).

## Correctness update: 2026-10-20

Full-grounding reward, residual mining and final evaluation now share one strict
output contract. Prose-wrapped JSON, trailing text, duplicate keys, unknown anomaly
types, empty required semantic fields and invalid boundaries cannot receive a
positive training reward while failing final parsing. The alternate legacy reward
function also uses this validator. Valid-output reward formulas are unchanged;
the separate historical `events`-schema compatibility parser is not used by the
four-way controlled launcher.

Checkpoint identity now includes `chat_template.jinja`, named templates under
`additional_chat_templates/` and `chat_templates/`, and relevant tokenizer assets.
Changing the actual chat template changes the fingerprint. Symlink/junction assets
are rejected, while optimizer state and unrelated nested training checkpoints
remain outside the inference identity.

**Migration:** start a fresh experiment directory and rerun prepare, mine,
schedule, train and evaluate with this revision. The `grounding-output-v2`
contract is recorded in experiment identities, mining outputs, prompt schedules,
completion receipts and RL resume signatures; unversioned/older artifacts are
rejected. Do not manually change their version or digest to bypass the check.
Existing source splits and valid Stage-2 weights need not be regenerated merely
for these fixes, but their new identity and downstream experiment artifacts must
be regenerated. No source data or old run is deleted.

## Four controlled variants

| ID | Method | Stage-3 data | Stage-3 update | Main comparison |
|---|---|---|---|---|
| A | Stage-2 SFT / w/o RAGO | None | None; evaluate the frozen Stage-2 checkpoint | A vs G: overall Stage-3 benefit |
| D | Matched Residual-SFT | Exact ordered residual schedule used by G | Plain supervised cross-entropy | D vs G: update objective on matched prompts |
| F | Random-PO | Uniform random training subset, matched prompt count | Same policy objective/reward as G | F vs G: residual selection |
| G | Full RAGO | Residual-balanced training subset | Supplied policy objective | Full method |

These are **training ablations**, not four unrelated model architectures. All
start from the same full local Stage-2 checkpoint for a given dataset and seed.
Only `train` is eligible for residual mining or optimization. D and G consume
exactly the same prompt IDs in the same epoch order. F and G use the same
optimizer, learning rate, reward, generation settings and number of updates.
No semantic auxiliary loss is added to D.

The default samples 50% of eligible training prompts per epoch, without replacement
within that epoch, for two epochs. This is an explicit **new control setting**, not
an inferred paper setting. Taking the complete population would remove the
subset-selection contrast. Set `--prompts-per-epoch` if needed, strictly between
zero and the train count, and keep it fixed for all variants. Bucket exhaustion
can change requested residual ratios; schedules report actual class counts and
whether random/residual membership happened to coincide.

**Equal prompt exposure is not equal compute.** D sees one target per prompt;
F/G generate multiple responses and also score a reference model. Report mining
cost, wall time, generated tokens, hardware and total training cost separately.
The launcher records wall time but never invents GPU-hours.

## Installation and software checks

Python 3.10+ is required; verification used Python 3.12.9. Use a fresh environment
and a PyTorch build compatible with your GPU. The four-way controlled launcher is
currently single-process, full-parameter training; it is not a LoRA or distributed
training recipe. Policy optimization loads both policy and reference models, so
do not assume a 7B run fits in 24 GB.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python scripts/run_submission_checks.py
python scripts/run_ablation.py --action plan
```

On Windows, pass `--bash "C:/Program Files/Git/bin/bash.exe"` to the checker if
that is your installed Git Bash location. Without Bash, shell syntax checks are
reported as skipped; Python checks still run. `requirements-lock.txt` records the
direct versions in the verification environment, not a complete dependency lock.
The optional semantic-summary loss requires `sentence-transformers`; it is off in
the controlled comparisons.

## Data and annotation scope

Obtain the corrected splits and reconstruction material from
[tango-datasets v0.2.0](https://github.com/guanzhaoyang86-lab/tango-datasets/releases/tag/v0.2.0).
This code repository does **not** redistribute TSB/KPI raw data. Reconstruct them
from authorized official sources following that repository. Dataset/model terms
are separate from this code's MIT license.

| Dataset | Final windows | Train / val / test | Training target |
|---|---:|---|---|
| TSB-AD-U, seven subsets | 2,693 | 1,883 / 268 / 542 | Localization only |
| AIOps 2018 KPI | 17,060 | 11,942 / 1,706 / 3,412 | Localization only |
| Synthetic `anomaly_db_v1` | 10,000 | 7,000 / 1,000 / 2,000 | Full synthetic annotations |

For TSB, run each of MSL, NEK, Power, SED, TAO, TODS and YAHOO separately: pass the
individual subset directory, not its parent. The public dataset audit documents
rounding in small groups, exclusions and source limitations. TSB/KPI use
group-isolated splits: overlapping windows, where retained, remain within one
split. This is **not strict future-period testing**. All synthetic
windows contain anomalies by design; 2,398 unknown historical generation-parameter
records remain unknown. Synthetic-only results cannot measure performance on
all-normal windows. KPI coverage excludes 31.51% of original points. Do not merge
different datasets blindly; the dataset audit records a cross-dataset image/label
collision between training samples.

Each materialized root must contain `train.json`, `val.json`, `test.json` and its
referenced images. The v3 adapter includes indexed numeric values (`0..255`, six
decimal places) with the plot. TSB/KPI provide only inclusive event
boundaries: semantic placeholders are not trained as truth, semantic metrics are
`null` with zero eligible denominator, and no arbitrary three-event cap is used.
Stage-2 full-schema predictions are projected to boundaries for fair localization
evaluation; malformed or missing predictions still count, rather than being dropped.

**Bundled miniature warning:** `dataset/anomaly_db_v1` contains 12 software-test
examples, not an independent research dataset. They overlap the full synthetic
corpus, including its heldout splits. Never use this fixture for warm-start or
benchmark training. Launchers reject it for training. Preparation-only smoke
checks are allowed and never load a model.

Review source/model terms before passing `--source-terms-reviewed`. Yahoo requires
special care for LLM training; the launcher blocks training/mining unless you also
pass `--yahoo-training-authorized`. That flag records your assertion of permission;
it does not grant permission. If you lack permission, omit that subset from trained
comparisons and state the exclusion rather than bypassing the guard.

## First create a common Stage-2 checkpoint

No weights are included. Supply a **full local** base VLM with processor/tokenizer
files and weights. These commands are explicit example SFT recipes, not evidence
of a reproduced historical training run. `--action plan` prints their config;
`--action train` actually trains. Defaults use one epoch, learning rate `1e-5`,
Adafactor, gradient accumulation 8 and plain supervised cross-entropy.

```bash
python scripts/train_stage.py --stage 1 --action train \
  --model-path /models/Qwen2.5-VL-7B-Instruct \
  --data-root /data/anomaly_db_v1_repaired_v2 \
  --output-dir outputs/stage1_seed2026 --seed 2026 --source-terms-reviewed

python scripts/train_stage.py --stage 2 --action train \
  --model-path outputs/stage1_seed2026/model \
  --data-root /data/kpi_aiops2018_grouped_v3 \
  --output-dir outputs/stage2_kpi_seed2026 --seed 2026 --source-terms-reviewed
```

Train Stage-2 on the appropriate target-domain **training** split. Never combine
target test labels into either stage or residual mining. For synthetic evaluation,
both stages must respect the same corrected synthetic split. Model selection and
hyperparameter choices must be fixed using training/validation information before
looking at test scores. The bootstrap saves its final epoch, not a test-selected
checkpoint. Repeat the full pipeline for independent seeds if claiming a
multi-seed training result; varying only Stage-3 is a conditional ablation and must
be labeled accordingly. Adapter-only or symlinked checkpoints are rejected by the
controlled launcher; first materialize the exact full merged model if applicable.

## Run A / D / F / G

Linux Bash example; replace paths with your authorized materialized data and
Stage-2 model. Run from this repository in the activated environment.

```bash
COMMON=(--data-root /data/kpi_aiops2018_grouped_v3
        --stage2-checkpoint /models/stage2_kpi_seed2026
        --experiment-dir outputs/kpi_seed2026
        --seed 2026 --source-terms-reviewed)

# Prepare once. Mining reads train only. Both schedules are frozen once.
python scripts/run_ablation.py --action prepare "${COMMON[@]}"
python scripts/run_ablation.py --action mine "${COMMON[@]}"
python scripts/run_ablation.py --action schedule "${COMMON[@]}"

bash scripts/run_A.sh "${COMMON[@]}"  # Evaluate unchanged Stage-2.
bash scripts/run_D.sh "${COMMON[@]}"  # Train matched SFT, then evaluate.
bash scripts/run_F.sh "${COMMON[@]}"  # Train random PO, then evaluate.
bash scripts/run_G.sh "${COMMON[@]}"  # Train full RAGO, then evaluate.
```

For a fresh directory, one command runs all phases:

```bash
python scripts/run_ablation.py --action all --variant all "${COMMON[@]}"
```

Platform-independent equivalents are `python scripts/run_ablation.py --variant A
--action evaluate ...` and `--variant D` (or F/G) `--action run ...`. PowerShell
also has `scripts/run_variant.ps1 -Variant A ...`; use `-Variant D/F/G` for the
other entries. Paths with spaces must be quoted. Bash wrappers accept `PYTHON_BIN`
to select a Python interpreter.

For validation-only work add `--eval-split val`. Use distinct experiment directories
for dataset/seed/config changes. Existing results and incomplete runs are never
silently overwritten. Run phases separately to diagnose a failure. The RL backend
supports explicit exact-state resume; the high-level launcher intentionally does
not certify a manually resumed run without a reviewed completion receipt.

Outputs include:

```text
experiment.json                 # Frozen seed, config and source split hashes.
stage2_identity.json            # Common checkpoint content fingerprint.
prepared_integrity.json         # Prepared input + actual image-content hashes.
mining/residual_pool.jsonl       # TRAIN predictions, error classes and provenance.
schedules/{residual,random}.json # Ordered IDs, budgets and actual error counts.
configs/{D,F,G}.json             # Fully resolved backend configurations.
{D,F,G}/model/                  # Final trained checkpoints, after successful runs.
{D,F,G}_completed.json           # Model and schedule content fingerprints.
evaluation/{A,D,F,G}/            # Predictions and .metrics.json files.
execution_costs.jsonl            # Observed command timings; not assumed GPU-hours.
```

Runtime files can contain licensed source data and absolute local paths. Do not
publish them blindly. They are excluded from this code release. Use the same
point-level metric and aggregation scheme across all variants; do not substitute
event F1 for missing point F1. The current evaluator reports macro averages of
per-window point precision, recall and F1, **not pooled source-point F1**. Pooled
source-series metrics require explicit index reconstruction and duplicate-point
handling. Also state whether a seven-subset mean is weighted or unweighted.
The archive's old `summarize_*` scripts retain their
historical directory conventions and are **not** the new four-way result collector.

## Policy objective and corrected-protocol differences

The supplied `rl_train_grpo.py` name is retained for compatibility. Its implemented
objective is group-relative, sequence-mean policy gradient plus a squared
policy/reference mean-log-probability penalty. It is **not canonical clipped GRPO**,
and that squared penalty is **not tokenwise KL divergence**. CLI `kl_coef` is a
legacy parameter name; the public control config calls it `reference_penalty_coef`.
We preserve the supplied objective instead of silently replacing the algorithm.

The corrected implementation scores the exact sampled token IDs, includes EOS,
handles PAD==EOS, matches scoring temperature to sampling, and disables dropout
in policy scoring. Visual structural tokens are excluded from response generation
and probability normalization consistently; plot tokens in the prompt remain
valid. Top-p is 1 and top-k is 0, avoiding an unmodeled truncated sampling
distribution. Non-finite loss/gradient and incompatible resume state fail loudly.

The new defaults use a 2,048-token generation budget, indexed v3 targets,
train-only fixed subset schedules, no semantic-type reward, and all-parameter
training. These corrections can change results from the original submission;
**rerun experiments and do not attach historical scores to this new protocol**.
Longer/more complex outputs can still hit a generation limit; inspect truncated
outputs and choose a common budget using validation, not test performance.

## Provenance and license

Derived from the author-provided `TANGO_supplementary_code` archive, SHA256:
`954b1745e08fe6cc85f12c24ed4a4480b389e9798cf1aee1a605ef14871d3ca6`.
The original archive is unchanged. This release replaces stale split instructions,
adds the four-way launch protocol and fixes tested implementation issues.

The original [MIT license](LICENSE) is preserved verbatim, including its original
copyright attribution. External datasets, model weights and dependencies retain
their own terms. No API credentials, trained weights, or real TSB/KPI records are
included. The manifest inventories the public files and their hashes.
