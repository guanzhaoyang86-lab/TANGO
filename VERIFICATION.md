# Verification scope

Verified on 2026-10-15. This is a software-verification record, not an experimental
results table or a guarantee that every configuration is free of defects.

## Completed checks

| Check | Result | What it establishes |
|---|---|---|
| Default automated suite | 103 passed | Unit, protocol and bounded integration contracts |
| Python parsing | 38 files passed | Syntax for the released Python sources |
| Bash parsing | 6 launchers passed | Shell syntax; not a Linux GPU run |
| PowerShell launcher | Syntax and plan invocation passed | Windows argument forwarding |
| Miniature prepare-only smoke | Passed | Data preparation and parser consistency, no model accuracy |
| Current full-data adapter compatibility | 29,753 records passed | Input paths, target scopes, inclusive boundaries and label contracts |
| Tiny real Qwen2.5-VL | Passed offline on CPU | Real processor/model forward, gradients, updates, save and reload |
| Four-way CLI integration | A/D/F/G passed, about 60.52 seconds | Prepare, mine, schedule, train D/F/G, save, evaluate A/D/F/G |

The adapter check covered 2,693 TSB windows and 17,060 KPI windows with
localization-only targets, and 10,000 full-semantic synthetic windows. It did not
repeat the entire independent pixel/duplicate audit; see the dataset release for
that audit and its limitations. No third-party records were copied into this code
repository. TSB's longest tested target contains 38 events and is not capped.

The automated tests include actual Torch computation and a real Hugging Face
Trainer two-update test with deliberately small test substitutes, plus separate
tests using the actual tiny Qwen2.5-VL classes and locally built processor. They
check exact D/G order and budget, F/G settings, checkpoint identity, image-content
changes, heldout exclusion, supervision masking, EOS/PAD behavior, exact sampled
tokens, visual-token probability normalization, reward parsing, optimizer updates,
and deterministic RL checkpoint resume. An orchestration test substitutes the
training/prediction subprocesses; it is not represented as a model experiment.

The additional full CLI smoke uses a fresh eight-training/two-validation/two-test
software fixture and a tiny random checkpoint deliberately biased toward EOS.
D, F and G each perform four update steps; all four variants produce predictions
and metrics. Empty generated outputs and zero rewards in this smoke are expected.
This verifies launch plumbing, **not useful learning, convergence, anomaly
detection quality, or a nonzero RAGO improvement**. All fixture checkpoints and
runtime outputs remain outside the public release.

To repeat the optional full CLI smoke (not collected by default pytest):

```bash
python tests/run_tiny_cli_integration.py --work-dir /new/empty/test-directory
```

## Environment and remaining validation

Verification used Windows, Python 3.12.9, Torch 2.5.1+cu121 running on **CPU**,
Transformers 4.57.0, datasets 4.0.0 and accelerate 1.10.1. Other direct versions
are in `requirements-lock.txt`. The inherited local environment emitted one
Requests optional character-detection dependency warning; tests still passed.
This was not a clean-environment dependency reinstall.

No usable pretrained Stage-2 checkpoint was supplied. No full pretrained 2B/7B/8B
training, GPU-kernel compatibility, GPU-memory fit, distributed optimization,
three-seed benchmark, reward effectiveness, or historical paper score reproduction
was verified. A compatible GPU environment and real authorized pretrained weights
are still needed. The original submission's scores must not be attributed to the
changed protocol without new experiments.

Public-file review excludes credentials, local absolute paths, real TSB/KPI data,
full synthetic data, runtime artifacts, caches and model weights. It is a bounded
publication check, not a comprehensive security audit. The twelve original
synthetic fixtures are explicitly software-only because they overlap the complete
dataset, including its current heldout partitions. `PACKAGE_MANIFEST.json` lists
the reviewed public files and hashes, excluding its own self-hash.
