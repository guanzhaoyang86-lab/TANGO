from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from .schema import parse_grounding_output, parse_json_object
from .taxonomy import canonicalize_type_name


DEFAULT_TAU_MATCH = 0.1
DEFAULT_TAU_GOOD = 0.5
DEFAULT_SERIES_LENGTH = 256
DEFAULT_MAX_INDEX = 255
DEFAULT_PREDICTION_SCHEMA = "evidence"
ERROR_TYPES = (
    "invalid_json",
    "false_negative",
    "false_positive",
    "boundary_error",
    "type_error",
    "correct_abnormal",
    "correct_normal",
)
LEGACY_RESIDUAL_SAMPLING_RATIOS = {
    "false_negative": 0.30,
    "boundary_error": 0.30,
    "false_positive": 0.20,
    "correct_abnormal": 0.10,
    "correct_normal": 0.10,
}
LEGACY_BOUNDARY_AWARE_REWARD_WEIGHTS = {
    "point": 0.0,
    "event": 0.45,
    "iou": 0.35,
    "boundary": 0.15,
    "type": 0.05,
}
DEFAULT_RESIDUAL_SAMPLING_RATIOS = {
    "false_negative": 0.60,
    "boundary_error": 0.20,
    "false_positive": 0.10,
    "correct_abnormal": 0.10,
    "correct_normal": 0.00,
}
DEFAULT_BOUNDARY_AWARE_REWARD_WEIGHTS = {
    "point": 0.60,
    "event": 0.00,
    "iou": 0.25,
    "boundary": 0.15,
    "type": 0.00,
}
ERROR_TYPE_FALLBACKS = {
    "false_negative": ("boundary_error", "correct_abnormal", "false_positive", "correct_normal"),
    "boundary_error": ("false_negative", "correct_abnormal", "type_error", "false_positive"),
    "false_positive": ("correct_normal", "boundary_error", "false_negative", "correct_abnormal"),
    "correct_abnormal": ("boundary_error", "type_error", "false_negative", "correct_normal"),
    "correct_normal": ("false_positive", "correct_abnormal", "boundary_error", "false_negative"),
    "type_error": ("boundary_error", "correct_abnormal", "false_negative", "false_positive"),
    "invalid_json": ("false_negative", "boundary_error", "false_positive", "correct_normal"),
}


@dataclass(frozen=True)
class EventParseResult:
    json_valid: bool
    events: list[dict[str, Any]]
    payload: dict[str, Any] | None = None
    errors: tuple[str, ...] = ()
    schema_compliant: bool = False


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, str):
        try:
            result = float(value.strip())
        except ValueError:
            return None
        return result if math.isfinite(result) else None
    return None


def _coerce_strict_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def extract_first_json_object(output_text: Any) -> dict[str, Any]:
    if isinstance(output_text, dict):
        return output_text
    if not isinstance(output_text, str):
        raise ValueError("model output must be a string or dict")

    text = output_text.strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("no valid JSON object found in model output")


def _event_key_for_schema(payload: dict[str, Any], prediction_schema: str) -> str | None:
    schema = prediction_schema.strip().lower()
    if schema == "events":
        return "events"
    if schema == "evidence":
        return "evidence"
    if schema == "auto":
        if "events" in payload:
            return "events"
        if "evidence" in payload:
            return "evidence"
        return "events"
    raise ValueError(f"Unsupported prediction schema: {prediction_schema}")


def _normalize_event_type(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return canonicalize_type_name(value)


def _normalize_event_item(item: dict[str, Any], *, event_key: str) -> dict[str, Any]:
    event = {
        "start": int(item["start"]),
        "end": int(item["end"]),
        "type": _normalize_event_type(item.get("type")),
    }
    if "strength" in item:
        event["strength"] = str(item["strength"])
    if "direction" in item:
        event["direction"] = str(item["direction"])
    if "explanation" in item:
        event["explanation"] = str(item["explanation"])
    elif "description" in item:
        event["explanation"] = str(item["description"])
    elif event_key == "evidence" and "direction" in item:
        event["explanation"] = str(item["direction"])
    return event


def _validate_schema_fields(
    payload: dict[str, Any],
    item: dict[str, Any],
    *,
    event_key: str,
    index: int,
) -> list[str]:
    errors: list[str] = []
    prefix = f"{event_key}[{index}]"
    if event_key == "evidence":
        summary = payload.get("summary")
        if "summary" not in payload:
            errors.append("missing_summary")
        elif not isinstance(summary, str):
            errors.append("summary_not_string")
        required_keys = ("start", "end", "type", "strength", "direction")
    else:
        required_keys = ("start", "end", "type")

    for key in required_keys:
        if key not in item:
            errors.append(f"{prefix}_missing_{key}")
    for key in ("type", "strength", "direction"):
        if key in item and not isinstance(item[key], str):
            errors.append(f"{prefix}_{key}_not_string")
    return errors


def parse_event_output(
    output_text: Any,
    *,
    max_index: int = DEFAULT_MAX_INDEX,
    prediction_schema: str = DEFAULT_PREDICTION_SCHEMA,
    annotation_scope: str = "full_grounding",
) -> EventParseResult:
    if annotation_scope == "full_semantic":
        annotation_scope = "full_grounding"
    if annotation_scope not in {"localization_only", "full_grounding"}:
        raise ValueError(f"Unsupported annotation scope: {annotation_scope}")
    if annotation_scope == "localization_only":
        from .supervision import project_localization_prediction, validate_localization_payload

        try:
            # Unlike the historical full-schema parser, do not recover JSON from
            # prose or ignore trailing output. Stage-2 semantic fields may be
            # projected for localization metrics, but are reported separately.
            payload = json.loads(output_text) if isinstance(output_text, str) else output_text
            projected = project_localization_prediction(payload, max_index + 1)
            try:
                validate_localization_payload(payload, max_index + 1)
                compliant = True
            except ValueError:
                compliant = False
            return EventParseResult(True, projected["evidence"], payload, (), compliant)
        except (ValueError, TypeError, KeyError) as exc:
            return EventParseResult(False, [], None, (str(exc),), False)
    schema = prediction_schema.strip().lower()
    try:
        # Only the explicitly requested historical events format retains JSON
        # recovery. The evidence contract used by all four controls must match
        # final evaluation: no prose, trailing text, or partial JSON recovery.
        payload = (
            extract_first_json_object(output_text)
            if schema == "events"
            else parse_json_object(output_text)
        )
    except Exception as exc:
        return EventParseResult(False, [], None, (str(exc),))

    errors: list[str] = []
    try:
        event_key = _event_key_for_schema(payload, prediction_schema)
    except ValueError as exc:
        return EventParseResult(False, [], payload, (str(exc),))
    if event_key is None or event_key not in payload:
        return EventParseResult(False, [], payload, (f"missing_{event_key or 'events'}",))
    if event_key == "evidence":
        try:
            result = parse_grounding_output(payload, series_length=max_index + 1)
        except (ValueError, TypeError, KeyError) as exc:
            return EventParseResult(False, [], payload, (str(exc),))
        events = [_normalize_event_item(item.to_dict(), event_key="evidence") for item in result.evidence]
        return EventParseResult(True, events, payload, (), True)
    raw_events = payload.get(event_key)
    if not isinstance(raw_events, list):
        return EventParseResult(False, [], payload, (f"{event_key}_not_list",))
    events: list[dict[str, Any]] = []
    for index, item in enumerate(raw_events):
        prefix = f"{event_key}[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix}_not_object")
            continue

        errors.extend(_validate_schema_fields(payload, item, event_key=event_key, index=index))

        start = _coerce_strict_int(item.get("start"))
        end = _coerce_strict_int(item.get("end"))
        if start is None:
            errors.append(f"{prefix}_start_not_int")
        if end is None:
            errors.append(f"{prefix}_end_not_int")
        if start is None or end is None:
            continue
        if start < 0 or start > max_index:
            errors.append(f"{prefix}_start_out_of_range")
        if end < 0 or end > max_index:
            errors.append(f"{prefix}_end_out_of_range")
        if start > end:
            errors.append(f"{prefix}_start_after_end")
        if errors:
            continue
        events.append(_normalize_event_item(item, event_key=event_key))

    if errors:
        return EventParseResult(False, [], payload, tuple(errors))
    return EventParseResult(True, events, payload, (), True)


def _scoped_gt_events(value: Any, annotation_scope: str, series_length: int) -> list[dict[str, Any]]:
    if annotation_scope != "localization_only":
        return events_from_ground_truth(value)
    from .supervision import validate_localization_payload

    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        for key in ("ground_truth", "target"):
            if key in value:
                value = value[key]
                break
    return validate_localization_payload(value, series_length)["evidence"]


def _normalize_gt_event(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    start = item.get("start")
    end = item.get("end")
    if isinstance(start, bool) or isinstance(end, bool):
        return None
    try:
        start_int = int(start)
        end_int = int(end)
    except (TypeError, ValueError):
        return None
    if start_int > end_int:
        return None
    event = {
        "start": start_int,
        "end": end_int,
        "type": _normalize_event_type(item.get("type")),
    }
    if "strength" in item:
        event["strength"] = str(item["strength"])
    if "direction" in item:
        event["direction"] = str(item["direction"])
    if "explanation" in item:
        event["explanation"] = str(item["explanation"])
    elif "description" in item:
        event["explanation"] = str(item["description"])
    return event


def point_labels_to_events(labels: Any) -> list[dict[str, Any]]:
    if not isinstance(labels, list):
        return []
    events: list[dict[str, Any]] = []
    start: int | None = None
    for index, value in enumerate(labels):
        numeric = _coerce_float(value)
        active = bool(value) if numeric is None else numeric > 0
        if active and start is None:
            start = index
        elif not active and start is not None:
            events.append({"start": start, "end": index - 1, "type": ""})
            start = None
    if start is not None:
        events.append({"start": start, "end": len(labels) - 1, "type": ""})
    return events


def events_from_ground_truth(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text or text[0] not in "{[":
            return []
        try:
            return events_from_ground_truth(json.loads(text))
        except json.JSONDecodeError:
            return []
    if isinstance(value, list):
        if all(not isinstance(item, (dict, list, tuple)) for item in value):
            return point_labels_to_events(value)
        events = []
        for item in value:
            normalized = _normalize_gt_event(item)
            if normalized is not None:
                events.append(normalized)
        return events
    if not isinstance(value, dict):
        return []

    for key in ("ground_truth", "target"):
        if key in value:
            return events_from_ground_truth(value[key])
    for key in ("events", "evidence", "intervals", "anomaly_intervals"):
        if key in value:
            return events_from_ground_truth(value[key])
    for key in ("point_labels", "labels"):
        if key in value:
            return point_labels_to_events(value[key])
    if "raw_sample" in value:
        return events_from_ground_truth(value["raw_sample"])
    return []


def interval_iou(interval_a: dict[str, Any], interval_b: dict[str, Any]) -> float:
    start_a = int(interval_a["start"])
    end_a = int(interval_a["end"])
    start_b = int(interval_b["start"])
    end_b = int(interval_b["end"])
    intersection = max(0, min(end_a, end_b) - max(start_a, start_b) + 1)
    union = max(end_a, end_b) - min(start_a, start_b) + 1
    return 0.0 if union <= 0 else float(intersection / union)


def match_events(
    pred_events: list[dict[str, Any]],
    gt_events: list[dict[str, Any]],
    *,
    tau_match: float = DEFAULT_TAU_MATCH,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for pred_index, pred_event in enumerate(pred_events):
        for gt_index, gt_event in enumerate(gt_events):
            iou = interval_iou(pred_event, gt_event)
            if iou >= tau_match:
                candidates.append(
                    {
                        "pred_index": pred_index,
                        "gt_index": gt_index,
                        "pred_event": pred_event,
                        "gt_event": gt_event,
                        "iou": float(iou),
                    }
                )
    candidates.sort(key=lambda item: item["iou"], reverse=True)

    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches: list[dict[str, Any]] = []
    for item in candidates:
        if item["pred_index"] in used_pred or item["gt_index"] in used_gt:
            continue
        used_pred.add(int(item["pred_index"]))
        used_gt.add(int(item["gt_index"]))
        matches.append(item)
    matches.sort(key=lambda item: (item["gt_index"], item["pred_index"]))
    return matches


def _event_f1_from_counts(tp: int, fp: int, fn: int, *, empty_is_perfect: bool = True) -> dict[str, float]:
    if tp == 0 and fp == 0 and fn == 0 and empty_is_perfect:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def boundary_mae_for_matches(
    matches: list[dict[str, Any]],
    *,
    has_events: bool,
    series_length: int = DEFAULT_SERIES_LENGTH,
) -> float:
    if not matches:
        return 0.0 if not has_events else float(series_length)
    total = 0.0
    for item in matches:
        pred = item["pred_event"]
        gt = item["gt_event"]
        total += (abs(int(pred["start"]) - int(gt["start"])) + abs(int(pred["end"]) - int(gt["end"]))) / 2.0
    return float(total / len(matches))


def evaluate_event_prediction(
    *,
    pred_events: list[dict[str, Any]],
    gt_events: list[dict[str, Any]],
    tau_match: float = DEFAULT_TAU_MATCH,
    series_length: int = DEFAULT_SERIES_LENGTH,
) -> dict[str, Any]:
    matches = match_events(pred_events, gt_events, tau_match=tau_match)
    matched_pred = {int(item["pred_index"]) for item in matches}
    matched_gt = {int(item["gt_index"]) for item in matches}
    tp = len(matches)
    fp = max(0, len(pred_events) - tp)
    fn = max(0, len(gt_events) - tp)
    prf = _event_f1_from_counts(tp, fp, fn)
    mean_iou = (
        1.0
        if not pred_events and not gt_events
        else float(sum(float(item["iou"]) for item in matches) / len(matches))
        if matches
        else 0.0
    )
    boundary_mae = boundary_mae_for_matches(
        matches,
        has_events=bool(pred_events or gt_events),
        series_length=series_length,
    )
    return {
        "event_precision": prf["precision"],
        "event_recall": prf["recall"],
        "event_f1": prf["f1"],
        "mean_iou": float(mean_iou),
        "boundary_mae": float(boundary_mae),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matches": matches,
        "unmatched_pred_indices": [
            index for index in range(len(pred_events)) if index not in matched_pred
        ],
        "unmatched_gt_indices": [
            index for index in range(len(gt_events)) if index not in matched_gt
        ],
    }


def _types_match(pred_event: dict[str, Any], gt_event: dict[str, Any]) -> bool:
    return _normalize_event_type(pred_event.get("type")) == _normalize_event_type(gt_event.get("type"))


def classify_error_type(
    *,
    json_valid: bool,
    pred_events: list[dict[str, Any]],
    gt_events: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    tau_good: float = DEFAULT_TAU_GOOD,
) -> str:
    if not json_valid:
        return "invalid_json"
    unmatched_pred = len(pred_events) - len(matches)
    unmatched_gt = len(gt_events) - len(matches)
    if gt_events and unmatched_gt > 0:
        return "false_negative"
    if (not gt_events and pred_events) or (gt_events and unmatched_pred > 0):
        return "false_positive"
    if gt_events and any(float(item["iou"]) < tau_good for item in matches):
        return "boundary_error"
    if gt_events and any(not _types_match(item["pred_event"], item["gt_event"]) for item in matches):
        return "type_error"
    if gt_events:
        return "correct_abnormal"
    return "correct_normal"


def score_prediction_output(
    *,
    pred_output: Any,
    gt: Any,
    tau_match: float = DEFAULT_TAU_MATCH,
    tau_good: float = DEFAULT_TAU_GOOD,
    series_length: int = DEFAULT_SERIES_LENGTH,
    max_index: int = DEFAULT_MAX_INDEX,
    prediction_schema: str = DEFAULT_PREDICTION_SCHEMA,
    annotation_scope: str = "full_grounding",
) -> dict[str, Any]:
    if type(series_length) is not int or series_length <= 0:
        raise ValueError("series_length must be a positive integer.")
    parsed = parse_event_output(
        pred_output,
        max_index=min(max_index, series_length - 1),
        prediction_schema=prediction_schema,
        annotation_scope=annotation_scope,
    )
    gt_events = _scoped_gt_events(gt, annotation_scope, series_length)
    event_scores = evaluate_event_prediction(
        pred_events=parsed.events,
        gt_events=gt_events,
        tau_match=tau_match,
        series_length=series_length,
    )
    if not parsed.json_valid:
        event_scores = dict(event_scores)
        event_scores["event_precision"] = 0.0
        event_scores["event_recall"] = 0.0
        event_scores["event_f1"] = 0.0
        event_scores["mean_iou"] = 0.0
    error_type = classify_error_type(
        json_valid=parsed.json_valid,
        pred_events=parsed.events,
        gt_events=gt_events,
        matches=event_scores["matches"],
        tau_good=tau_good,
    )
    return {
        "json_valid": bool(parsed.json_valid),
        "schema_compliant": bool(parsed.schema_compliant),
        "annotation_scope": annotation_scope,
        "parse_errors": list(parsed.errors),
        "gt_evidence": gt_events,
        "pred_evidence": parsed.events,
        "error_type": error_type,
        **event_scores,
    }


def _union_coverage_length(events: list[dict[str, Any]], *, series_length: int) -> int:
    if not events:
        return 0
    clipped = []
    for event in events:
        start = max(0, min(series_length - 1, int(event["start"])))
        end = max(0, min(series_length - 1, int(event["end"])))
        if start <= end:
            clipped.append((start, end))
    if not clipped:
        return 0
    clipped.sort()
    merged: list[list[int]] = []
    for start, end in clipped:
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return int(sum(end - start + 1 for start, end in merged))


def _events_to_point_mask(events: list[dict[str, Any]], *, series_length: int) -> list[int]:
    mask = [0] * int(series_length)
    for event in events:
        try:
            start = max(0, min(int(series_length) - 1, int(event["start"])))
            end = max(0, min(int(series_length) - 1, int(event["end"])))
        except (KeyError, TypeError, ValueError):
            continue
        if start > end:
            continue
        for idx in range(start, end + 1):
            mask[idx] = 1
    return mask


def point_prf_for_events(
    pred_events: list[dict[str, Any]],
    gt_events: list[dict[str, Any]],
    *,
    series_length: int,
) -> dict[str, float]:
    pred_mask = _events_to_point_mask(pred_events, series_length=series_length)
    gt_mask = _events_to_point_mask(gt_events, series_length=series_length)
    tp = sum(1 for pred, gt in zip(pred_mask, gt_mask) if pred and gt)
    fp = sum(1 for pred, gt in zip(pred_mask, gt_mask) if pred and not gt)
    fn = sum(1 for pred, gt in zip(pred_mask, gt_mask) if gt and not pred)
    if tp == 0 and fp == 0 and fn == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def resolve_boundary_reward_weights(weights: dict[str, float] | None = None) -> dict[str, float]:
    resolved = dict(DEFAULT_BOUNDARY_AWARE_REWARD_WEIGHTS)
    if weights:
        for key, value in weights.items():
            if key not in resolved:
                raise ValueError(f"Unsupported boundary-aware reward weight key: {key}")
            resolved[key] = float(value)
    return resolved


def compute_boundary_aware_reward_details(
    pred_output: Any,
    gt: Any,
    *,
    tau_match: float = DEFAULT_TAU_MATCH,
    series_length: int = DEFAULT_SERIES_LENGTH,
    max_index: int = DEFAULT_MAX_INDEX,
    reward_weights: dict[str, float] | None = None,
    prediction_schema: str = DEFAULT_PREDICTION_SCHEMA,
    annotation_scope: str = "full_grounding",
) -> dict[str, Any]:
    if type(series_length) is not int or series_length <= 0:
        raise ValueError("series_length must be a positive integer.")
    weights = resolve_boundary_reward_weights(reward_weights)
    localization_only = annotation_scope == "localization_only"
    if localization_only and weights["type"] != 0:
        raise ValueError("Localization-only labels cannot support a nonzero type reward")
    parsed = parse_event_output(
        pred_output,
        max_index=min(max_index, series_length - 1),
        prediction_schema=prediction_schema,
        annotation_scope=annotation_scope,
    )
    gt_events = _scoped_gt_events(gt, annotation_scope, series_length)
    if not parsed.json_valid or (localization_only and not parsed.schema_compliant):
        return {
            "reward": 0.0,
            "g_valid": 0.0,
            "json_valid": float(parsed.json_valid),
            "schema_valid": 0.0,
            "reward_weights": weights,
            "reward_terms": {},
            "errors": list(parsed.errors) or ["incompatible_localization_schema"],
            "pred_intervals": [],
            "gt_intervals": gt_events,
            "matches": [],
            "event_precision": 0.0,
            "event_recall": 0.0,
            "event_f1": 0.0,
            "mean_iou": 0.0,
            "boundary_iou": 0.0,
            "boundary_reward": 0.0,
            "point_precision": 0.0,
            "point_recall": 0.0,
            "point_f1": 0.0,
            "type_score": None if localization_only else 0.0,
            "normal_pred_coverage": 0.0,
            "normal_num_pred_events": 0,
        }

    event_scores = evaluate_event_prediction(
        pred_events=parsed.events,
        gt_events=gt_events,
        tau_match=tau_match,
        series_length=series_length,
    )
    matches = event_scores["matches"]
    point_scores = point_prf_for_events(parsed.events, gt_events, series_length=series_length)
    if gt_events:
        point_f1 = float(point_scores["f1"])
        event_f1 = float(event_scores["event_f1"])
        mean_iou = float(sum(float(item["iou"]) for item in matches) / len(matches)) if matches else 0.0
        if matches:
            boundary_terms = []
            type_terms = []
            for item in matches:
                pred = item["pred_event"]
                gt_event = item["gt_event"]
                boundary_error = abs(int(pred["start"]) - int(gt_event["start"])) + abs(
                    int(pred["end"]) - int(gt_event["end"])
                )
                boundary_terms.append(max(0.0, 1.0 - (boundary_error / (2.0 * series_length))))
                type_terms.append(1.0 if _types_match(pred, gt_event) else 0.0)
            boundary_reward = float(sum(boundary_terms) / len(boundary_terms))
            type_score = float(sum(type_terms) / len(type_terms))
        else:
            boundary_reward = 0.0
            type_score = 0.0
        reward_terms = {
            "point": point_f1,
            "event": event_f1,
            "iou": mean_iou,
            "boundary": boundary_reward,
            "type": type_score,
        }
        r_task = sum(float(weights[key]) * float(reward_terms[key]) for key in weights)
    elif not parsed.events:
        reward_terms = {
            "point": 1.0,
            "event": 1.0,
            "iou": 1.0,
            "boundary": 1.0,
            "type": 1.0,
            "normal": 1.0,
        }
        r_task = 1.0
        mean_iou = 1.0
        boundary_reward = 1.0
        type_score = 1.0
    else:
        coverage = _union_coverage_length(parsed.events, series_length=series_length) / float(series_length)
        r_task = max(0.0, 1.0 - coverage - (0.2 * len(parsed.events)))
        reward_terms = {
            "point": 0.0,
            "normal_false_positive": float(r_task),
            "pred_coverage": float(coverage),
            "num_pred_events": float(len(parsed.events)),
        }
        mean_iou = 0.0
        boundary_reward = 0.0
        type_score = 0.0

    if localization_only:
        reward_terms.pop("type", None)
    return {
        "reward": float(r_task),
        "g_valid": 1.0,
        "json_valid": 1.0,
        "schema_valid": 1.0,
        "reward_weights": weights,
        "reward_terms": reward_terms,
        "errors": [],
        "pred_intervals": parsed.events,
        "gt_intervals": gt_events,
        "matches": matches,
        "event_precision": float(event_scores["event_precision"]),
        "event_recall": float(event_scores["event_recall"]),
        "event_f1": float(event_scores["event_f1"]),
        "mean_iou": float(mean_iou),
        "boundary_iou": float(mean_iou),
        "boundary_reward": float(boundary_reward),
        "boundary_mae": float(event_scores["boundary_mae"]),
        "point_precision": float(point_scores["precision"]),
        "point_recall": float(point_scores["recall"]),
        "point_f1": float(point_scores["f1"]),
        "type_score": None if localization_only else float(type_score),
        "annotation_scope": annotation_scope,
        "normal_pred_coverage": (
            float(_union_coverage_length(parsed.events, series_length=series_length) / float(series_length))
            if not gt_events and parsed.events
            else 0.0
        ),
        "normal_num_pred_events": len(parsed.events) if not gt_events else 0,
    }


def parse_sampling_ratios(value: str | dict[str, float] | None) -> dict[str, float]:
    if value in (None, ""):
        return dict(DEFAULT_RESIDUAL_SAMPLING_RATIOS)
    if isinstance(value, dict):
        ratios = {str(key): float(val) for key, val in value.items()}
    else:
        ratios = {}
        for part in str(value).split(","):
            if not part.strip():
                continue
            key, sep, raw_val = part.partition("=")
            if not sep:
                raise ValueError(f"Invalid sampling ratio item: {part}")
            ratios[key.strip()] = float(raw_val.strip())
    if not ratios:
        raise ValueError("sampling ratios must not be empty")
    for key, value_float in ratios.items():
        if key not in ERROR_TYPES:
            raise ValueError(f"Unsupported error_type in sampling ratios: {key}")
        if value_float < 0:
            raise ValueError(f"Sampling ratio must be non-negative for {key}")
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("sampling ratios must sum to a positive value")
    return {key: value_float / total for key, value_float in ratios.items()}


def _allocate_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {key: float(ratio) * total for key, ratio in ratios.items()}
    counts = {key: int(math.floor(value)) for key, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(raw, key=lambda key: (raw[key] - counts[key], raw[key]), reverse=True)
    for key in order[:remaining]:
        counts[key] += 1
    return counts


def _dataset_for_record(record: dict[str, Any]) -> str:
    for container_key in ("metadata", "context"):
        container = record.get(container_key)
        if isinstance(container, dict) and container.get("source_dataset") not in (None, ""):
            return str(container["source_dataset"])
    raw_sample = record.get("raw_sample")
    if isinstance(raw_sample, dict):
        context = raw_sample.get("context")
        if isinstance(context, dict) and context.get("source_dataset") not in (None, ""):
            return str(context["source_dataset"])
    if record.get("dataset") not in (None, ""):
        return str(record["dataset"])
    return "unknown"


def build_balanced_residual_epoch(
    records: list[dict[str, Any]],
    *,
    epoch_size: int | None = None,
    sampling_ratios: str | dict[str, float] | None = None,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    if not records:
        return []
    size = len(records) if epoch_size is None else int(epoch_size)
    if size <= 0:
        return []
    ratios = parse_sampling_ratios(sampling_ratios)
    rng = random.Random(seed)

    buckets: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        error_type = str(record.get("error_type", "unknown"))
        dataset = _dataset_for_record(record)
        buckets[error_type][dataset].append(record)
    for dataset_buckets in buckets.values():
        for values in dataset_buckets.values():
            rng.shuffle(values)

    def available_count(error_type: str) -> int:
        return sum(len(values) for values in buckets.get(error_type, {}).values())

    picked_by_dataset: Counter[str] = Counter()
    picked: list[dict[str, Any]] = []

    def pick_one(error_type: str) -> dict[str, Any] | None:
        dataset_buckets = buckets.get(error_type, {})
        available_datasets = [dataset for dataset, values in dataset_buckets.items() if values]
        if not available_datasets:
            return None
        min_count = min(picked_by_dataset[dataset] for dataset in available_datasets)
        candidates = [dataset for dataset in available_datasets if picked_by_dataset[dataset] == min_count]
        dataset = rng.choice(candidates)
        picked_by_dataset[dataset] += 1
        return dataset_buckets[dataset].pop()

    def pick_many(error_type: str, count: int) -> int:
        added = 0
        while added < count:
            item = pick_one(error_type)
            if item is None:
                break
            picked.append(item)
            added += 1
        return added

    target_counts = _allocate_counts(size, ratios)
    for target_type, target_count in target_counts.items():
        needed = target_count
        for candidate_type in (target_type, *ERROR_TYPE_FALLBACKS.get(target_type, ())):
            if needed <= 0:
                break
            if available_count(candidate_type) <= 0:
                continue
            needed -= pick_many(candidate_type, needed)
        if needed > 0:
            for candidate_type in ERROR_TYPES:
                if needed <= 0:
                    break
                if available_count(candidate_type) <= 0:
                    continue
                needed -= pick_many(candidate_type, needed)

    if len(picked) < size:
        by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_dataset[_dataset_for_record(record)].append(record)
        datasets = sorted(by_dataset)
        while len(picked) < size:
            min_count = min(picked_by_dataset[dataset] for dataset in datasets)
            candidates = [dataset for dataset in datasets if picked_by_dataset[dataset] == min_count]
            dataset = rng.choice(candidates)
            picked_by_dataset[dataset] += 1
            picked.append(rng.choice(by_dataset[dataset]))

    rng.shuffle(picked)
    return picked[:size]


def residual_pool_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    dataset_counts = Counter(str(record.get("dataset") or _dataset_for_record(record)) for record in records)
    error_counts = Counter(str(record.get("error_type", "unknown")) for record in records)
    total = len(records)
    json_valid_count = sum(1 for record in records if bool(record.get("json_valid")))
    def gt_evidence(record: dict[str, Any]) -> list[dict[str, Any]]:
        value = record.get("gt_evidence", record.get("gt_events", []))
        return value if isinstance(value, list) else []

    def pred_evidence(record: dict[str, Any]) -> list[dict[str, Any]]:
        value = record.get("sft_pred_evidence", record.get("sft_pred_events", []))
        return value if isinstance(value, list) else []

    abnormal_records = [record for record in records if gt_evidence(record)]
    normal_records = [record for record in records if not gt_evidence(record)]
    abnormal_empty = sum(
        1
        for record in abnormal_records
        if bool(record.get("json_valid")) and bool(gt_evidence(record)) and not pred_evidence(record)
    )
    normal_false_positive = sum(
        1
        for record in normal_records
        if bool(record.get("json_valid")) and not gt_evidence(record) and bool(pred_evidence(record))
    )

    def mean(key: str) -> float:
        if not records:
            return 0.0
        return float(sum(float(record.get(key, 0.0)) for record in records) / len(records))

    return {
        "num_samples": total,
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "error_type_counts": {
            key: {
                "count": int(error_counts.get(key, 0)),
                "ratio": 0.0 if total == 0 else float(error_counts.get(key, 0) / total),
            }
            for key in ERROR_TYPES
        },
        "json_parse_success_rate": 0.0 if total == 0 else float(json_valid_count / total),
        "abnormal_empty_rate": (
            0.0 if not abnormal_records else float(abnormal_empty / len(abnormal_records))
        ),
        "normal_false_positive_rate": (
            0.0 if not normal_records else float(normal_false_positive / len(normal_records))
        ),
        "mean_event_f1": mean("event_f1"),
        "mean_iou": mean("mean_iou"),
        "mean_boundary_mae": mean("boundary_mae"),
    }
