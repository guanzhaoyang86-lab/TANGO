from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from ts_grounder.utils import distributed_barrier, dump_yaml, ensure_dir, is_main_process, load_yaml, set_seed
from ts_grounder.vlm_training import load_prepared_dataset, prepare_vlm_dataset, run_hf_vlm_sft, run_prepare_only


def _parse_optional_bool(value: str | None) -> bool | None:
    if value in (None, ""):
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean env value: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TS Grounder VLM")
    parser.add_argument("--config", type=str, default="configs/vlm_smoke.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--mode", type=str, default=None, choices=["prepare_only", "train"])
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--prepared-dataset-root", type=str, default=None,
                        help="Use an already verified shared prepared cache (required for matched controls).")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    cfg = load_yaml(repo_root / args.config if not Path(args.config).is_absolute() else args.config)
    env_source_root = os.environ.get("TS_GROUNDER_SOURCE_ROOT")
    if env_source_root:
        cfg.setdefault("data", {})
        cfg["data"]["source_root"] = env_source_root
    env_source_dataset_filter = os.environ.get("TS_GROUNDER_SOURCE_DATASET_FILTER")
    if env_source_dataset_filter:
        cfg.setdefault("data", {})
        cfg["data"]["source_dataset_filter"] = [
            item.strip()
            for item in env_source_dataset_filter.split(",")
            if item.strip()
        ]
    env_model_path = os.environ.get("TS_GROUNDER_MODEL_PATH")
    if env_model_path:
        cfg.setdefault("model", {})
        cfg["model"]["model_name_or_path"] = env_model_path
    env_checkpoint_output_dir = os.environ.get("TS_GROUNDER_CHECKPOINT_OUTPUT_DIR")
    if env_checkpoint_output_dir:
        cfg.setdefault("training", {})
        cfg["training"]["checkpoint_output_dir"] = env_checkpoint_output_dir
    env_num_train_epochs = os.environ.get("TS_GROUNDER_NUM_TRAIN_EPOCHS")
    if env_num_train_epochs:
        num_train_epochs = float(env_num_train_epochs)
        if num_train_epochs <= 0:
            raise ValueError("TS_GROUNDER_NUM_TRAIN_EPOCHS must be greater than zero.")
        cfg.setdefault("training", {})
        cfg["training"]["num_train_epochs"] = num_train_epochs
    for env_name, cfg_key in (
        ("TS_GROUNDER_BF16", "bf16"),
        ("TS_GROUNDER_FP16", "fp16"),
    ):
        env_value = _parse_optional_bool(os.environ.get(env_name))
        if env_value is not None:
            cfg.setdefault("training", {})
            cfg["training"][cfg_key] = env_value
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.mode:
        cfg.setdefault("training", {})
        cfg["training"]["mode"] = args.mode
    if args.resume_from_checkpoint:
        cfg.setdefault("training", {})
        cfg["training"]["resume_from_checkpoint"] = args.resume_from_checkpoint

    if str(cfg.get("training", {}).get("mode", "prepare_only")) == "train":
        source_path = Path(cfg.get("data", {}).get("source_root", ""))
        if not source_path.is_absolute():
            source_path = repo_root / source_path
        manifest = source_path / "manifest.json"
        if manifest.is_file():
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            if "software verification" in str(metadata.get("purpose", "")).lower() or "mini" in str(metadata.get("name", "")).lower():
                raise ValueError("Bundled mini data is software-only and overlaps benchmark heldout samples; never use it to train a checkpoint")

    set_seed(int(cfg.get("seed", 2026)))
    output_dir = ensure_dir(repo_root / cfg["output_dir"] if not Path(cfg["output_dir"]).is_absolute() else cfg["output_dir"])
    if is_main_process():
        dump_yaml(cfg, output_dir / "resolved_config.yaml")
    distributed_barrier()

    dataset_root = Path(args.prepared_dataset_root).resolve() if args.prepared_dataset_root else output_dir / "dataset_cache"
    if args.prepared_dataset_root:
        prepared = load_prepared_dataset(dataset_root)
    elif is_main_process():
        prepared = prepare_vlm_dataset(cfg, repo_root=repo_root, output_dir=output_dir)
    distributed_barrier()
    if not is_main_process() and not args.prepared_dataset_root:
        prepared = load_prepared_dataset(dataset_root)
    distributed_barrier()

    mode = str(cfg["training"].get("mode", "prepare_only"))
    if mode == "prepare_only":
        if is_main_process():
            summary = run_prepare_only(prepared, output_dir=output_dir)
        else:
            summary = {"mode": "prepare_only"}
    elif mode == "train":
        summary = run_hf_vlm_sft(cfg, prepared, output_dir=output_dir)
    else:
        raise ValueError(f"Unsupported training.mode: {mode}")
    distributed_barrier()

    if is_main_process():
        print(f"Mode: {mode}")
        print(f"Output dir: {output_dir}")
        print(f"Dataset cache: {prepared.root}")
        if "checkpoint_dir" in summary:
            print(f"Checkpoint dir: {summary['checkpoint_dir']}")
        if summary.get("resumed_from_checkpoint"):
            print(f"Resumed from: {summary['resumed_from_checkpoint']}")
        if summary.get("best_model_checkpoint"):
            print(f"Best checkpoint: {summary['best_model_checkpoint']}")
        if summary.get("best_metric") is not None:
            print(f"Best metric: {summary['best_metric']}")
        print(f"Summary written under: {output_dir}")
        if "oracle_metrics" in summary:
            print(f"Oracle point F1: {summary['oracle_metrics']['metrics']['point_f1']:.4f}")


if __name__ == "__main__":
    main()
