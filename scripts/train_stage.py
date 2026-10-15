"""Explicit, train-only SFT bootstrap for Stage-1 and the common Stage-2 checkpoint.

This is an example recipe, not a claim of reproduced paper results. Only train
records are optimized. Validation is not used to select the final checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_ablation import (ROOT, data_config, execute, fingerprint_checkpoint, index_train_records,
                          load_jsonl, require_yahoo_training_permission, sft_config,
                          validate_control_config, validate_source, write_new_json)
from ts_grounder.utils import load_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["1", "2"], required=True)
    parser.add_argument("--model-path", required=True, help="Full local base model for Stage-1; Stage-1 model for Stage-2")
    parser.add_argument("--data-root", required=True, help="Synthetic data for Stage-1; target dataset for Stage-2")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--action", choices=["plan", "train"], default="plan")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--source-terms-reviewed", action="store_true")
    parser.add_argument("--yahoo-training-authorized", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0 or args.gradient_accumulation_steps <= 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Epochs, learning rate and accumulation must be positive")
    source, model, output = (Path(value).resolve() for value in (args.data_root, args.model_path, args.output_dir))
    if any(output == path or output.is_relative_to(path) or path.is_relative_to(output) for path in (source, model)):
        raise ValueError("Keep output separate from source data and model directories")
    cfg = validate_control_config(load_yaml(ROOT / "configs/controls.yaml"))
    cfg["torch_dtype"] = args.torch_dtype
    cfg["learning_rate"] = args.learning_rate
    resolved = sft_config(source, model, output, args.seed, cfg)
    resolved["training"].update(num_train_epochs=args.epochs,
                                gradient_accumulation_steps=args.gradient_accumulation_steps,
                                lr_scheduler_type="linear", warmup_steps=0, max_grad_norm=1.0)
    if args.action == "plan":
        print(json.dumps({"stage": args.stage, "example_recipe": True, "config": resolved}, indent=2))
        return
    if not args.source_terms_reviewed:
        raise ValueError("Review data/model terms and pass --source-terms-reviewed")
    source_hashes = validate_source(source)  # Reject the miniature before any training.
    model_hash = fingerprint_checkpoint(model)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty output directory; existing checkpoints are never overwritten")
    from ts_grounder.vlm_training import prepare_vlm_dataset
    prepared = prepare_vlm_dataset(resolved, ROOT, output)
    rows = load_jsonl(prepared.root / "train.jsonl")
    index_train_records(rows)
    require_yahoo_training_permission(rows, args.yahoo_training_authorized)
    if args.stage == "1" and any(row.get("metadata", {}).get("annotation_scope") == "localization_only" for row in rows):
        raise ValueError("Stage-1 expects the semantic synthetic dataset, not localization-only data")
    config_path = output / "bootstrap_config.json"
    write_new_json(config_path, resolved)
    write_new_json(output / "bootstrap_identity.json", {"stage": args.stage, "seed": args.seed,
                   "initial_model_sha256": model_hash, "source_split_sha256": source_hashes,
                   "example_recipe_not_paper_reproduction": True})
    execute([sys.executable, str(ROOT / "train.py"), "--config", str(config_path),
             "--prepared-dataset-root", str(prepared.root)], output, f"stage_{args.stage}_sft")
    write_new_json(output / "bootstrap_completed.json", {"stage": args.stage,
                   "model_path": str(output / "model"), "model_sha256": fingerprint_checkpoint(output / "model")})


if __name__ == "__main__":
    main()
