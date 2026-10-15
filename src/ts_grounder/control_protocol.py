"""Auditable, train-only fixed schedules for the A/D/F/G controlled comparison.

This module imports no model framework and performs no downloads. A schedule
fixes prompt exposure/order, NOT equal GPU compute between SFT and policy updates.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def fingerprint_checkpoint(path: str | Path) -> str:
    """Content identity of a local full checkpoint, independent of its location."""
    root = Path(path).resolve()
    if not root.is_dir() or not (root / "config.json").is_file():
        raise ValueError("A local full Stage-2 checkpoint with config.json is required")
    if (root / "adapter_config.json").exists():
        raise ValueError("Controlled runs require a full checkpoint, not an adapter; merge with its exact base first")
    weights = list(root.glob("*.safetensors")) + list(root.glob("pytorch_model*.bin"))
    if not weights:
        raise ValueError("Stage-2 checkpoint has no full model weights")
    excluded = {"optimizer.bin", "optimizer.pt", "training_args.bin", "trainer_state.json",
                "scheduler.pt", "rng_state.pth", "all_results.json", "train_results.json", "eval_results.json"}
    inventory = {}
    for item in sorted(root.iterdir()):
        if item.is_symlink():
            raise ValueError("Use a materialized local checkpoint, not symlinked files")
        if item.is_file() and item.name not in excluded and item.suffix in {
            ".json", ".model", ".txt", ".tiktoken", ".safetensors", ".bin"
        }:
            inventory[item.name] = {"bytes": item.stat().st_size, "sha256": file_sha256(item)}
    return json_sha256(inventory)


def record_id(record: dict[str, Any]) -> str:
    value = record.get("id") or record.get("sample_id")
    if not isinstance(value, str) or not value:
        raise ValueError("Every prepared record must have a nonempty string ID")
    return value


def record_fingerprint(record: dict[str, Any]) -> str:
    # Mining may add predictions, metrics and provenance, but must not alter any
    # model input, actual target, source sample or supervision scope.
    return json_sha256({
        **{key: record.get(key) for key in (
            "id", "split", "system_prompt", "user_prompt", "assistant_text", "target", "image_path", "raw_sample")},
        "annotation_scope": record.get("metadata", {}).get("annotation_scope", record.get("annotation_scope", "full_grounding")),
        "image_sha256": file_sha256(record["image_path"]),
    })


def index_train_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not records:
        raise ValueError("Empty training population")
    indexed = {}
    for record in records:
        ident = record_id(record)
        if ident in indexed:
            raise ValueError(f"Duplicate training record ID: {ident}")
        if record.get("split") != "train":
            raise ValueError(f"Only explicitly marked train records are eligible: {ident}")
        containers = [record, record.get("metadata", {}), record.get("raw_sample", {})]
        containers += [item.get(key, {}) for item in list(containers)
                       for key in ("context", "provenance") if isinstance(item, dict)]
        for item in containers:
            if isinstance(item, dict) and any(item.get(key) not in (None, "", "train")
                                             for key in ("split", "materialized_subset_split")):
                raise ValueError(f"Conflicting heldout split provenance: {ident}")
        if record.get("metadata", {}).get("task_type", "grounding") != "grounding":
            raise ValueError("Controlled runs support grounding records only, not QA mixtures")
        for key in ("system_prompt", "user_prompt", "assistant_text", "target", "image_path"):
            if key not in record:
                raise ValueError(f"Prepared training record lacks {key}: {ident}")
        indexed[ident] = record
    return indexed


def build_prompt_schedules(
    train_records: list[dict[str, Any]], residual_records: list[dict[str, Any]], *,
    seed: int, epochs: int, prompts_per_epoch: int, checkpoint_fingerprint: str,
    sampling_ratios: dict[str, float] | None = None,
    expected_input_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from .event_metrics import ERROR_TYPES, build_balanced_residual_epoch, parse_sampling_ratios

    indexed = index_train_records(train_records)
    residual_index = index_train_records(residual_records)
    if set(indexed) != set(residual_index):
        raise ValueError("Residual mining must cover every eligible training ID exactly once")
    fingerprints = {ident: record_fingerprint(record) for ident, record in indexed.items()}
    mining_inputs = set()
    for ident, record in residual_index.items():
        if record_fingerprint(record) != fingerprints[ident]:
            raise ValueError(f"Residual mining changed input/target fields: {ident}")
        if record.get("error_type") not in ERROR_TYPES:
            raise ValueError(f"Unknown residual class: {ident}")
        provenance = record.get("residual_mining", {})
        if (provenance.get("checkpoint_fingerprint") != checkpoint_fingerprint
                or provenance.get("source_split") != "train"):
            raise ValueError(f"Residual checkpoint/split provenance mismatch: {ident}")
        input_hash = provenance.get("input_sha256")
        if not isinstance(input_hash, str) or len(input_hash) != 64 or any(c not in "0123456789abcdef" for c in input_hash):
            raise ValueError(f"Missing residual input hash: {ident}")
        mining_inputs.add(input_hash)
    if len(mining_inputs) != 1 or (expected_input_sha256 is not None and mining_inputs != {expected_input_sha256}):
        raise ValueError("Residual rows refer to inconsistent or different training input files")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if (isinstance(prompts_per_epoch, bool) or not isinstance(prompts_per_epoch, int)
            or not 0 < prompts_per_epoch < len(indexed)):
        raise ValueError("Use 0 < prompts_per_epoch < train count; taking the full population removes subset selection")
    if len(checkpoint_fingerprint) != 64 or any(c not in "0123456789abcdef" for c in checkpoint_fingerprint):
        raise ValueError("A SHA256 identity of the common Stage-2 checkpoint is required")
    ratios = parse_sampling_ratios(sampling_ratios)
    if any(not math.isfinite(float(value)) or value < 0 for value in ratios.values()):
        raise ValueError("Sampling ratios must be finite and nonnegative")
    residual_epochs, random_epochs = [], []
    source_ids = sorted(indexed)
    for epoch in range(epochs):
        selected = build_balanced_residual_epoch(
            [residual_index[ident] for ident in source_ids], epoch_size=prompts_per_epoch,
            sampling_ratios=ratios, seed=seed + epoch,
        )
        residual_epochs.append([record_id(record) for record in selected])
        random_epochs.append(random.Random(seed + epoch).sample(source_ids, prompts_per_epoch))
    common = {"schema_version": 1, "seed": seed, "num_epochs": epochs,
              "prompts_per_epoch": prompts_per_epoch, "total_prompt_exposures": epochs * prompts_per_epoch,
              "stage2_checkpoint_fingerprint": checkpoint_fingerprint,
              "mining_input_sha256": next(iter(mining_inputs)),
              "record_fingerprints": fingerprints,
              "train_population_sha256": json_sha256(fingerprints),
              "sampling_without_replacement_within_epoch": True,
              "equal_compute_claim": False}
    residual = {**common, "selection": "residual", "epochs": residual_epochs,
                "requested_sampling_ratios": ratios,
                "actual_error_counts_by_epoch": [dict(Counter(residual_index[ident]["error_type"] for ident in row))
                                                  for row in residual_epochs],
                "fallback_policy": "Original bucket-exhaustion fallback; actual quotas can differ from requested ratios"}
    random_schedule = {**common, "selection": "random", "epochs": random_epochs,
                       "actual_error_counts_by_epoch": [dict(Counter(residual_index[ident]["error_type"] for ident in row))
                                                         for row in random_epochs],
                       "matching_residual_membership_by_epoch": [set(a) == set(b) for a, b in zip(residual_epochs, random_epochs)]}
    return residual, random_schedule


def load_prompt_schedule(path: str | Path, records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    indexed = index_train_records(records)
    if payload.get("schema_version") != 1 or payload.get("selection") not in {"residual", "random"}:
        raise ValueError("Unsupported prompt schedule")
    fingerprints = {ident: record_fingerprint(record) for ident, record in indexed.items()}
    if payload.get("record_fingerprints") != fingerprints or payload.get("train_population_sha256") != json_sha256(fingerprints):
        raise ValueError("Schedule was built from different model inputs, targets, IDs, or supervision")
    epochs = payload.get("epochs")
    if not isinstance(epochs, list) or not epochs or len(epochs) != payload.get("num_epochs"):
        raise ValueError("Invalid schedule epoch count")
    result = []
    for ids in epochs:
        if (not isinstance(ids, list) or not ids or len(ids) != payload.get("prompts_per_epoch")
                or any(not isinstance(ident, str) or ident not in indexed for ident in ids)):
            raise ValueError("Invalid schedule IDs or length")
        if payload.get("sampling_without_replacement_within_epoch") and len(set(ids)) != len(ids):
            raise ValueError("Duplicate ID within a without-replacement epoch")
        result.append([indexed[ident] for ident in ids])
    if sum(map(len, result)) != payload.get("total_prompt_exposures"):
        raise ValueError("Total prompt exposure mismatch")
    return result
