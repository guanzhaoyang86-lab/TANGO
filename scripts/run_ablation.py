"""Prepare, mine, schedule, run and evaluate the A/D/F/G controlled comparison.

All paths/checkpoint hashes/configuration are frozen per experiment directory.
Runtime outputs may contain licensed source data: never publish those outputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ts_grounder.control_protocol import (
    build_prompt_schedules, file_sha256, fingerprint_checkpoint, index_train_records,
    json_sha256, load_prompt_schedule,
)
from ts_grounder.utils import load_yaml
from ts_grounder.vlm_eval import load_jsonl
from ts_grounder.vlm_prompting import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT

VARIANTS = {"A": "Stage-2 SFT / w/o RAGO", "D": "Matched Residual-SFT",
            "F": "Random-PO", "G": "Full RAGO"}


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def validate_control_config(cfg):
    expected = {"epochs", "prompt_fraction", "learning_rate", "optimizer", "num_generations",
                "max_new_tokens", "temperature", "top_p", "reference_penalty_coef", "torch_dtype",
                "gradient_checkpointing", "tau_match", "tau_good", "residual_sampling_ratios", "reward_weights"}
    if not isinstance(cfg, dict) or set(cfg) != expected:
        raise ValueError(f"Control config keys must be exactly {sorted(expected)}")
    for key in ("epochs", "num_generations", "max_new_tokens"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("prompt_fraction", "learning_rate", "temperature", "reference_penalty_coef", "tau_match", "tau_good"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], (int, float)) or not math.isfinite(cfg[key]):
            raise ValueError(f"Invalid {key}")
    if not 0 < cfg["prompt_fraction"] < 1 or cfg["learning_rate"] <= 0 or cfg["temperature"] <= 0:
        raise ValueError("Invalid sampling fraction, learning rate, or temperature")
    if cfg["num_generations"] < 2 or cfg["max_new_tokens"] < 2048:
        raise ValueError("Use at least two candidates and a 2048-token generation budget")
    if isinstance(cfg["top_p"], bool) or cfg["top_p"] != 1 or cfg["reference_penalty_coef"] < 0:
        raise ValueError("Current policy update requires top_p=1 and a nonnegative reference penalty")
    if not 0 <= cfg["tau_match"] <= cfg["tau_good"] <= 1:
        raise ValueError("Require 0 <= tau_match <= tau_good <= 1")
    if cfg["optimizer"] not in {"adafactor", "adamw"} or cfg["torch_dtype"] not in {"bfloat16", "float32"}:
        raise ValueError("Controlled runs support adafactor/adamw and bfloat16/float32 only")
    if not isinstance(cfg["gradient_checkpointing"], bool):
        raise ValueError("gradient_checkpointing must be boolean")
    for key in ("reward_weights", "residual_sampling_ratios"):
        values = cfg[key]
        if not isinstance(values, dict) or not values or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
            for v in values.values()
        ) or sum(values.values()) <= 0:
            raise ValueError(f"Invalid {key}")
    if set(cfg["reward_weights"]) != {"point", "event", "iou", "boundary", "type"}:
        raise ValueError("All five reward weights must be explicit")
    if cfg["reward_weights"]["type"] != 0:
        raise ValueError("The common localization protocol cannot use a semantic type reward")
    if set(cfg["residual_sampling_ratios"]) != {"false_negative", "boundary_error", "false_positive", "correct_abnormal", "correct_normal"}:
        raise ValueError("All five residual sampling classes must be explicit")
    return cfg


def data_config(source_root):
    return {"source_root": str(source_root), "image_subdir": None,
            "include_grounding_records": True, "include_qa_pairs": False,
            "include_indexed_series_text": True, "indexed_series_precision": 6,
            "indexed_series_compact": False, "balanced_by_type": False,
            "max_train_samples": None, "max_val_samples": None, "max_test_samples": None,
            "series_normalization": {"enabled": False}}


def sft_config(source_root, model_path, output_dir, seed, control):
    return {"seed": seed, "output_dir": str(output_dir), "data": data_config(source_root),
            "model": {"model_name_or_path": str(model_path), "trust_remote_code": False,
                      "processor_kwargs": {}, "from_pretrained_kwargs": {"torch_dtype": control["torch_dtype"]}},
            "prompt": {"system_prompt": DEFAULT_SYSTEM_PROMPT, "user_prompt": DEFAULT_USER_PROMPT},
            "training": {"mode": "train", "per_device_train_batch_size": 1,
                         "per_device_eval_batch_size": 1, "gradient_accumulation_steps": 1,
                         "learning_rate": control["learning_rate"], "num_train_epochs": 1,
                         "logging_steps": 10, "save_strategy": "no", "evaluation_strategy": "no",
                         "bf16": control["torch_dtype"] == "bfloat16", "fp16": False,
                         "optim": "adafactor" if control["optimizer"] == "adafactor" else "adamw_torch",
                         "weight_decay": 0.0,
                         "gradient_checkpointing": control["gradient_checkpointing"],
                         "evidence_loss_weight": 0.0, "summary_semantic_loss_weight": 0.0,
                         "generation_eval_after_train": False}}


def validate_source(source, allow_demo=False):
    source = Path(source).resolve()
    manifest = source / "manifest.json"
    if manifest.exists():
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        demo = "software verification" in str(meta.get("purpose", "")).lower() or "mini" in str(meta.get("name", "")).lower()
        if demo and not allow_demo:
            raise ValueError("The bundled miniature is software-only and overlaps released heldout samples. Do not train benchmark checkpoints on it")
    hashes = {}
    ids_by_split = {}
    for split in ("train", "val", "test"):
        path = source / f"{split}.json"
        hashes[split] = file_sha256(path)
        rows = json.loads(path.read_text(encoding="utf-8"))
        ids = [str(row["sample_id"]) for row in rows]
        if not rows or len(set(ids)) != len(ids):
            raise ValueError(f"Empty split or duplicate IDs in {split}")
        ids_by_split[split] = set(ids)
    if any(ids_by_split[a] & ids_by_split[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Cross-split sample ID overlap")
    return hashes


def require_yahoo_training_permission(records, authorized):
    if not authorized and any("yahoo" in str(row.get("metadata", {}).get("source_dataset", "")).lower() for row in records):
        raise ValueError("Yahoo LLM training needs separate permission. Do not set --yahoo-training-authorized unless you actually hold it")


def execute(command, run_root, phase):
    started = time.perf_counter()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", USE_TF="0", USE_FLAX="0", TOKENIZERS_PARALLELISM="false")
    # Old cluster overrides must not silently change a controlled comparison.
    for key in tuple(env):
        if key.startswith("TS_GROUNDER_"):
            env.pop(key)
    print(subprocess.list2cmdline(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, env=env)
    elapsed = time.perf_counter() - started
    with (run_root / "execution_costs.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"phase": phase, "command": command, "returncode": result.returncode,
                                 "wall_seconds": elapsed, "device_count_assumption": None,
                                 "gpu_hours": None, "compute_matched_claim": False,
                                 "utc": datetime.now(timezone.utc).isoformat()}) + "\n")
    if result.returncode:
        raise RuntimeError(f"{phase} failed with exit {result.returncode}; inspect the retained run, do not overwrite it")


def prepared_integrity(cache):
    result = {"jsonl": {}, "images": {}}
    for split in ("train", "val", "test"):
        path = cache / f"{split}.jsonl"
        result["jsonl"][split] = file_sha256(path)
        for row in load_jsonl(path):
            image = Path(row["image_path"])
            if not image.is_file():
                raise ValueError("Prepared image does not exist")
            key = str(image.resolve())
            if key not in result["images"]:
                result["images"][key] = file_sha256(image)
    return result


def run(args):
    cfg = validate_control_config(load_yaml(Path(args.config).resolve()))
    if args.action == "plan":
        print(json.dumps({"variants": VARIANTS, "seed": args.seed, "config": cfg,
                          "flow": ["prepare", "mine TRAIN with Stage-2", "schedule", "train D/F/G", "evaluate A/D/F/G"],
                          "shared_D_G_schedule": "schedules/residual.json", "equal_compute": False,
                          "requires_existing_local_stage2_checkpoint": True}, indent=2))
        return
    if not args.source_terms_reviewed:
        raise ValueError("Review source/model terms and pass --source-terms-reviewed")
    if args.data_root is None or args.experiment_dir is None:
        raise ValueError("--data-root and --experiment-dir are required")
    source, run_root = Path(args.data_root).resolve(), Path(args.experiment_dir).resolve()
    if run_root == source or source.is_relative_to(run_root) or run_root.is_relative_to(source):
        raise ValueError("Keep experiment outputs separate from source data")
    source_hashes = validate_source(source, args.allow_demo_prepare and args.action == "prepare")
    stage2 = Path(args.stage2_checkpoint).resolve() if args.stage2_checkpoint else None
    if args.action != "prepare" and stage2 is None:
        raise ValueError("Supply --stage2-checkpoint; A never creates a substitute checkpoint")
    model_hash = fingerprint_checkpoint(stage2) if stage2 else None
    identity = {"schema_version": 1, "seed": args.seed, "source_root": str(source), "source_split_sha256": source_hashes,
                "control_config": cfg, "prompts_per_epoch_override": args.prompts_per_epoch}
    receipt = run_root / "experiment.json"
    if receipt.exists():
        previous = json.loads(receipt.read_text(encoding="utf-8"))
        if previous != identity:
            raise ValueError("Experiment directory belongs to different data/config/seed. Use a new directory")
    else:
        if run_root.exists() and any(run_root.iterdir()):
            raise ValueError("Refusing an untracked nonempty experiment directory")
        write_new_json(receipt, identity)
    checkpoint_receipt = run_root / "stage2_identity.json"
    if stage2:
        identity2 = {"path": str(stage2), "sha256": model_hash}
        if checkpoint_receipt.exists():
            if json.loads(checkpoint_receipt.read_text(encoding="utf-8")) != identity2:
                raise ValueError("All variants must start from the same frozen Stage-2 checkpoint")
        else:
            write_new_json(checkpoint_receipt, identity2)
    actions = (["prepare", "mine", "schedule", "train", "evaluate"] if args.action == "all"
               else ["train", "evaluate"] if args.action == "run" else [args.action])
    if args.variant == "A" and args.action == "all":
        actions = ["prepare", "evaluate"]
    cache = run_root / "shared" / "dataset_cache"
    integrity_path = run_root / "prepared_integrity.json"
    if "prepare" in actions:
        if cache.exists():
            if not integrity_path.exists():
                raise ValueError("Partial prepared cache exists; use a fresh experiment directory")
        else:
            from ts_grounder.vlm_training import prepare_vlm_dataset
            preparation_cfg = sft_config(source, stage2 or "NOT_LOADED", run_root / "shared", args.seed, cfg)
            prepare_vlm_dataset(preparation_cfg, ROOT, run_root / "shared")
            write_new_json(integrity_path, prepared_integrity(cache))
    if not integrity_path.exists() or json.loads(integrity_path.read_text(encoding="utf-8")) != prepared_integrity(cache):
        raise ValueError("Prepared cache/images changed or are not verified; run prepare first")
    records = load_jsonl(cache / "train.jsonl")
    index_train_records(records)
    if any(action in actions for action in ("mine", "train")):
        require_yahoo_training_permission(records, args.yahoo_training_authorized)
    mining = run_root / "mining" / "residual_pool.jsonl"
    if "mine" in actions:
        if mining.exists():
            raise FileExistsError("Residual mining output already exists; run schedule/train/evaluate separately")
        command = [sys.executable, str(ROOT / "scripts/build_residual_pool.py"),
                   "--model-path", str(stage2), "--checkpoint-fingerprint", model_hash,
                   "--real-train-data", str(cache / "train.jsonl"), "--output", str(mining),
                   "--max-new-tokens", str(cfg["max_new_tokens"]), "--torch-dtype", cfg["torch_dtype"],
                   "--tau-match", str(cfg["tau_match"]), "--tau-good", str(cfg["tau_good"])]
        if args.predictions_jsonl:
            if not args.predictions_manifest:
                raise ValueError("Cached mining predictions require their verified sidecar manifest")
            command.extend(["--predictions-jsonl", str(Path(args.predictions_jsonl).resolve()),
                            "--predictions-manifest", str(Path(args.predictions_manifest).resolve())])
        execute(command, run_root, "train_only_residual_mining")
    schedules = {"residual": run_root / "schedules/residual.json", "random": run_root / "schedules/random.json"}
    if "schedule" in actions:
        summary_path = mining.with_suffix(".summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("input_sha256") != file_sha256(cache / "train.jsonl") or summary.get("residual_sha256") != file_sha256(mining):
            raise ValueError("Mining summary does not match the current prepared input/output")
        observed_checkpoint = summary.get("checkpoint_fingerprint")
        if isinstance(observed_checkpoint, dict):
            observed_checkpoint = observed_checkpoint.get("sha256")
        if observed_checkpoint != model_hash:
            raise ValueError("Residuals were mined with a different Stage-2 checkpoint")
        budget = args.prompts_per_epoch if args.prompts_per_epoch is not None else max(1, math.floor(len(records) * cfg["prompt_fraction"]))
        residual, random_schedule = build_prompt_schedules(
            records, load_jsonl(mining), seed=args.seed, epochs=cfg["epochs"], prompts_per_epoch=budget,
            checkpoint_fingerprint=model_hash, sampling_ratios=cfg["residual_sampling_ratios"],
            expected_input_sha256=file_sha256(cache / "train.jsonl"),
        )
        write_new_json(schedules["residual"], residual)
        write_new_json(schedules["random"], random_schedule)
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    if "train" in actions:
        for variant in variants:
            if variant == "A":
                print("A: no Stage-3 training; evaluate the unchanged Stage-2 checkpoint", flush=True)
                continue
            output = run_root / variant
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(f"Refusing to overwrite variant {variant}; use its backend's explicit resume after reviewing state")
            schedule = schedules["random" if variant == "F" else "residual"]
            epochs = load_prompt_schedule(schedule, records)
            schedule_info = json.loads(schedule.read_text(encoding="utf-8"))
            if schedule_info["stage2_checkpoint_fingerprint"] != model_hash:
                raise ValueError("Schedule checkpoint mismatch")
            if variant == "D":
                resolved = sft_config(source, stage2, output, args.seed, cfg)
                resolved["training"].update(prompt_schedule_file=str(schedule), fixed_order=True,
                                             max_steps=sum(map(len, epochs)), lr_scheduler_type="constant",
                                             warmup_steps=0, max_grad_norm=0.0)
                path = run_root / "configs/D.json"
                write_new_json(path, resolved)
                command = [sys.executable, str(ROOT / "train.py"), "--config", str(path),
                           "--prepared-dataset-root", str(cache)]
            else:
                resolved = {"model_name_or_path": str(stage2), "reference_model_path": str(stage2),
                            "train_file": str(cache / "train.jsonl"), "output_dir": str(output),
                            "prompt_schedule_file": str(schedule), "seed": args.seed,
                            "num_train_epochs": cfg["epochs"], "num_generations": cfg["num_generations"],
                            "learning_rate": cfg["learning_rate"], "optimizer": cfg["optimizer"],
                            "weight_decay": 0.0,
                            "max_new_tokens": cfg["max_new_tokens"], "temperature": cfg["temperature"],
                            "top_p": cfg["top_p"], "kl_coef": cfg["reference_penalty_coef"],
                            "torch_dtype": cfg["torch_dtype"], "disable_gradient_checkpointing": not cfg["gradient_checkpointing"],
                            "use_residual_pool": False, "use_boundary_aware_reward": True,
                            "reward_weights": cfg["reward_weights"], "tau_match": cfg["tau_match"],
                            "tau_good": cfg["tau_good"], "series_length": 256, "max_index": 255,
                            "prediction_schema": "evidence", "save_steps": 0, "logging_steps": 10,
                            "save_optimizer_state": True}
                path = run_root / f"configs/{variant}.json"
                write_new_json(path, {"rl": resolved})
                command = [sys.executable, str(ROOT / "src/ts_grounder/rl_train_grpo.py"), "--config", str(path)]
            execute(command, run_root, f"{variant}_training")
            final_model_hash = fingerprint_checkpoint(output / "model")
            write_new_json(run_root / f"{variant}_completed.json", {
                "name": VARIANTS[variant], "stage2_sha256": model_hash,
                "model_sha256": final_model_hash, "model_path": str(output / "model"),
                "schedule_sha256": file_sha256(schedule), "prompt_exposures": sum(map(len, epochs)),
                "same_prompt_exposure_not_equal_compute": True,
            })
    if "evaluate" in actions:
        for variant in variants:
            checkpoint = stage2 if variant == "A" else run_root / variant / "model"
            if variant != "A" and not (run_root / f"{variant}_completed.json").exists():
                raise ValueError(f"Variant {variant} has no completed training receipt")
            if not checkpoint.is_dir():
                raise FileNotFoundError(checkpoint)
            if variant != "A":
                completed = json.loads((run_root / f"{variant}_completed.json").read_text(encoding="utf-8"))
                if completed.get("model_sha256") != fingerprint_checkpoint(checkpoint):
                    raise ValueError(f"Variant {variant} checkpoint changed after training")
                schedule_path = schedules["random" if variant == "F" else "residual"]
                if completed.get("stage2_sha256") != model_hash or completed.get("schedule_sha256") != file_sha256(schedule_path):
                    raise ValueError(f"Variant {variant} training provenance changed")
            output = run_root / "evaluation" / variant / f"{args.eval_split}_predictions.jsonl"
            if output.exists():
                raise FileExistsError("Evaluation already exists; do not overwrite heldout results")
            execute([sys.executable, str(ROOT / "predict.py"), "--model-path", str(checkpoint),
                     "--dataset-jsonl", str(cache / f"{args.eval_split}.jsonl"), "--output", str(output),
                     "--max-new-tokens", str(cfg["max_new_tokens"]), "--torch-dtype", cfg["torch_dtype"]],
                    run_root, f"{variant}_{args.eval_split}_evaluation")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["plan", "prepare", "mine", "schedule", "train", "evaluate", "run", "all"], default="plan")
    parser.add_argument("--variant", choices=["A", "D", "F", "G", "all"], default="all")
    parser.add_argument("--config", default=str(ROOT / "configs/controls.yaml"))
    parser.add_argument("--data-root")
    parser.add_argument("--stage2-checkpoint")
    parser.add_argument("--experiment-dir")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--prompts-per-epoch", type=int)
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
    parser.add_argument("--source-terms-reviewed", action="store_true")
    parser.add_argument("--yahoo-training-authorized", action="store_true")
    parser.add_argument("--allow-demo-prepare", action="store_true", help="Software-only prepare check; never allows benchmark training on the mini fixture")
    parser.add_argument("--predictions-jsonl")
    parser.add_argument("--predictions-manifest")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
