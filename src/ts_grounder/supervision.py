"""Annotation-aware contracts: point labels never imply semantic ground truth."""
from __future__ import annotations

import json
from typing import Any

TARGET_ADAPTER_VERSION = "annotation-aware-v3"
LOCALIZATION_SYSTEM_PROMPT = (
    "You are a time-series anomaly localization model. Read the indexed plot and "
    "indexed numeric values. Return exactly one JSON object with only the key `evidence`. "
    "Each evidence item has only integer `start` and `end`, using zero-based inclusive "
    "local indices. If no anomaly is detected, return {\"evidence\": []}. "
    "Do not output type, strength, direction, summary, markdown, or extra text."
)
LOCALIZATION_USER_PROMPT = (
    "Locate all anomalous intervals in this indexed time series. Return only the "
    "localization JSON specified above. Boundaries are inclusive."
)


def annotation_scope(sample: dict[str, Any]) -> str:
    contexts = [sample, sample.get("context") or {}, sample.get("metadata") or {}]
    if any(not isinstance(obj, dict) for obj in contexts):
        raise ValueError("Annotation contexts must be JSON objects")
    scopes = {"full_grounding" if obj["annotation_scope"] == "full_semantic" else obj["annotation_scope"]
              for obj in contexts if obj.get("annotation_scope")}
    flags = [obj["semantic_label_available"] for obj in contexts if "semantic_label_available" in obj]
    if any(type(flag) is not bool for flag in flags):
        raise ValueError("semantic_label_available must be a JSON boolean")
    if len(scopes) > 1 or len(set(flags)) > 1:
        raise ValueError("Conflicting annotation metadata")
    scope = next(iter(scopes), None)
    if scope is not None and scope not in {"localization_only", "full_grounding"}:
        raise ValueError(f"Unknown annotation_scope: {scope}")
    if scope == "full_grounding" and False in flags:
        raise ValueError("Full grounding cannot use unavailable semantic labels")
    if scope == "localization_only" and True in flags:
        raise ValueError("Inconsistent localization-only metadata")
    if scope is not None:
        return scope
    if False in flags:
        return "localization_only"
    sources = {str(obj.get("source_dataset", "")).casefold() for obj in contexts}
    if sources & {"msl", "nek", "power", "sed", "tao", "tods", "yahoo", "kpi", "aiops2018", "aiops2018_kpi"}:
        return "localization_only"
    return "full_grounding"


def validate_localization_payload(payload: dict[str, Any], series_length: int | None = None) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {"evidence"}:
        raise ValueError("Localization target must contain only evidence")
    if not isinstance(payload["evidence"], list):
        raise ValueError("evidence must be a list")
    if series_length is not None and series_length <= 0:
        raise ValueError("series_length must be positive")
    evidence = []
    for item in payload["evidence"]:
        if not isinstance(item, dict) or set(item) != {"start", "end"}:
            raise ValueError("Localization evidence must contain only start/end")
        start, end = item["start"], item["end"]
        if type(start) is not int or type(end) is not int or start < 0 or end < start:
            raise ValueError("Invalid inclusive interval")
        if series_length is not None and end >= series_length:
            raise ValueError("Interval outside the input window")
        evidence.append({"start": start, "end": end})
    return {"evidence": evidence}


def project_localization_prediction(payload: dict[str, Any], series_length: int | None = None) -> dict[str, Any]:
    """Score Stage-2 full-schema output without inventing semantic references."""
    if not isinstance(payload, dict) or not isinstance(payload.get("evidence"), list):
        raise ValueError("Prediction must contain an evidence list")
    evidence = []
    for item in payload["evidence"]:
        if not isinstance(item, dict) or not {"start", "end"} <= set(item):
            raise ValueError("Prediction evidence lacks boundaries")
        evidence.append({"start": item["start"], "end": item["end"]})
    return validate_localization_payload({"evidence": evidence}, series_length)


def localization_target(sample: dict[str, Any]) -> dict[str, Any]:
    convention = (sample.get("context") or {}).get("interval_convention", sample.get("interval_convention"))
    if convention is not None and convention not in {"inclusive", "closed", "zero_based_inclusive", "[start,end]"}:
        raise ValueError(f"Model data must use inclusive intervals; got {convention!r}")
    payload = {"evidence": [{"start": event["start"], "end": event["end"]} for event in sample.get("events", [])]}
    series_length = len(sample.get("series", []))
    result = validate_localization_payload(payload, series_length)
    labels = sample.get("point_labels")
    if labels is not None:
        if len(labels) != series_length or any(value not in (0, 1) for value in labels):
            raise ValueError("Invalid point_labels")
        reconstructed = [0] * series_length
        for item in result["evidence"]:
            reconstructed[item["start"]:item["end"] + 1] = [1] * (item["end"] - item["start"] + 1)
        if list(labels) != reconstructed:
            raise ValueError("Inclusive evidence disagrees with point_labels")
    return result


def validate_prepared_record(record: dict[str, Any]) -> str:
    metadata = record.get("metadata", {})
    if metadata.get("target_adapter_version") != TARGET_ADAPTER_VERSION:
        raise ValueError(f"Old/unversioned dataset cache: regenerate with {TARGET_ADAPTER_VERSION}")
    scope = annotation_scope(record)
    if metadata.get("task_type") == "qa":
        if scope == "localization_only":
            raise ValueError("Point-label-only data cannot train semantic QA")
        if record.get("assistant_text") != record.get("target", {}).get("answer"):
            raise ValueError("QA assistant text disagrees with answer")
        return scope
    payload = json.loads(record["assistant_text"])
    if payload != record["target"]:
        raise ValueError("assistant_text disagrees with target")
    if scope == "localization_only":
        validate_localization_payload(payload, int(metadata["series_length"]))
        if not metadata.get("indexed_series_text_enabled") or metadata.get("indexed_series_precision") != 6:
            raise ValueError("Localization-only data requires indexed numeric values at six decimals")
        if record["system_prompt"] != LOCALIZATION_SYSTEM_PROMPT:
            raise ValueError("Localization-only record has an incompatible system prompt")
    else:
        from .schema import GroundingResult
        GroundingResult.from_dict(payload)
    return scope


def validate_generation_budget(records: list[dict[str, Any]], max_new_tokens: int) -> None:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if any(annotation_scope(record) == "localization_only" for record in records) and max_new_tokens < 2048:
        raise ValueError("Localization evaluation requires max_new_tokens >= 2048; never truncate multi-event outputs")


def forbidden_action_token_ids(model) -> list[int]:
    """Input-only visual structure must not be generated as a textual JSON action."""
    config = getattr(model, "config", None)
    forbidden = set()
    for container in (config, getattr(config, "text_config", None)):
        for name in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
            value = getattr(container, name, None)
            if value is not None:
                values = value if isinstance(value, (list, tuple)) else [value]
                forbidden.update(int(item) for item in values)
    return sorted(forbidden)


def disable_dropout(model) -> int:
    """Use the same deterministic stochastic-layer setting for matched D/F/G."""
    import torch
    count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.modules.dropout._DropoutNd) and module.p:
            module.p = 0.0
            count += 1
        for name in ("attention_dropout", "hidden_dropout", "dropout"):
            value = getattr(module, name, None)
            if isinstance(value, float) and value:
                setattr(module, name, 0.0)
                count += 1
    return count
