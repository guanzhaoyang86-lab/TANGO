from __future__ import annotations

import json
import math
import re
from typing import Any

from .event_metrics import (
    DEFAULT_BOUNDARY_AWARE_REWARD_WEIGHTS,
    DEFAULT_MAX_INDEX,
    DEFAULT_PREDICTION_SCHEMA,
    DEFAULT_SERIES_LENGTH,
    DEFAULT_TAU_MATCH,
    compute_boundary_aware_reward_details as _compute_boundary_aware_reward_details,
    resolve_boundary_reward_weights,
)
from .schema import parse_grounding_output, parse_json_object
from .taxonomy import TYPE_NAMES, canonicalize_type_name


ALLOWED_TYPES = set(TYPE_NAMES)
DEFAULT_REWARD_WEIGHTS = {
    "event_f1": 0.45,
    "boundary_iou": 0.35,
    "hallucination_penalty": 0.0,
}
ACTIVE_REWARD_TERMS = ("event_f1", "boundary_iou", "hallucination_penalty")


def resolve_reward_weights(weights: dict[str, float] | None = None) -> dict[str, float]:
    resolved = dict(DEFAULT_REWARD_WEIGHTS)
    if weights:
        for key, value in weights.items():
            if key not in resolved:
                raise ValueError(f"Unsupported reward weight key: {key}")
            resolved[key] = float(value)
    return resolved


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"[-+]?\d+", text):
            return int(text)
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
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


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def parse_model_output(output_text: Any) -> dict[str, Any]:
    """Legacy extraction utility; never use it to score model predictions.

    Kept for callers that explicitly need to recover an object from old logs.
    Active reward paths use the same strict JSON contract as final evaluation.
    """

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


def schema_errors(pred_json: Any) -> list[str]:
    """Compatibility validator backed by the shared full-grounding contract."""
    try:
        parse_grounding_output(pred_json)
    except (ValueError, TypeError) as exc:
        return [str(exc)]
    return []


def validate_schema(pred_json: Any) -> bool:
    return not schema_errors(pred_json)


def extract_intervals(pred_json: Any) -> list[dict[str, Any]]:
    if not isinstance(pred_json, dict):
        return []
    evidence = pred_json.get("evidence", [])
    if not isinstance(evidence, list):
        return []

    intervals: list[dict[str, Any]] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        start = _coerce_int(item.get("start"))
        end = _coerce_int(item.get("end"))
        if start is None or end is None:
            continue
        interval = dict(item)
        interval["start"] = start
        interval["end"] = end
        if "type" in interval:
            interval["type"] = canonicalize_type_name(str(interval["type"]))
        intervals.append(interval)
    return intervals


def merge_point_labels_to_intervals(labels: Any) -> list[dict[str, Any]]:
    if labels is None:
        return []
    intervals: list[dict[str, Any]] = []
    start: int | None = None
    for index, value in enumerate(labels):
        numeric = _coerce_float(value)
        active = bool(value) if numeric is None else numeric > 0
        if active and start is None:
            start = index
        elif not active and start is not None:
            intervals.append({"start": start, "end": index - 1})
            start = None
    if start is not None:
        intervals.append({"start": start, "end": len(labels) - 1})
    return intervals


def _interval_bounds(interval: Any, *, allow_reversed: bool = False) -> tuple[int, int] | None:
    if isinstance(interval, dict):
        start = _coerce_int(interval.get("start"))
        end = _coerce_int(interval.get("end"))
    elif isinstance(interval, (list, tuple)) and len(interval) >= 2:
        start = _coerce_int(interval[0])
        end = _coerce_int(interval[1])
    else:
        return None
    if start is None or end is None:
        return None
    if start > end and not allow_reversed:
        return None
    return start, end


def interval_iou(pred_interval: Any, gt_interval: Any) -> float:
    pred_bounds = _interval_bounds(pred_interval)
    gt_bounds = _interval_bounds(gt_interval)
    if pred_bounds is None or gt_bounds is None:
        return 0.0

    pred_start, pred_end = pred_bounds
    gt_start, gt_end = gt_bounds
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    if inter_start > inter_end:
        return 0.0
    intersection = inter_end - inter_start + 1
    pred_length = pred_end - pred_start + 1
    gt_length = gt_end - gt_start + 1
    union = pred_length + gt_length - intersection
    return 0.0 if union <= 0 else intersection / union


def _valid_intervals(intervals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = []
    for interval in intervals:
        bounds = _interval_bounds(interval)
        if bounds is None:
            continue
        start, end = bounds
        copied = dict(interval)
        copied["start"] = start
        copied["end"] = end
        valid.append(copied)
    return valid


def match_intervals(
    pred_intervals: list[dict[str, Any]],
    gt_intervals: list[dict[str, Any]],
    *,
    iou_threshold: float = 1e-12,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for pred_index, pred in enumerate(pred_intervals):
        for gt_index, gt in enumerate(gt_intervals):
            iou = interval_iou(pred, gt)
            if iou > iou_threshold:
                candidates.append(
                    {
                        "pred_index": pred_index,
                        "gt_index": gt_index,
                        "pred_interval": pred,
                        "gt_interval": gt,
                        "iou": iou,
                    }
                )

    candidates.sort(key=lambda item: item["iou"], reverse=True)
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches: list[dict[str, Any]] = []
    for item in candidates:
        if item["pred_index"] in used_pred or item["gt_index"] in used_gt:
            continue
        used_pred.add(item["pred_index"])
        used_gt.add(item["gt_index"])
        matches.append(item)
    matches.sort(key=lambda item: (item["gt_index"], item["pred_index"]))
    return matches


def compute_event_f1(
    pred_intervals: list[dict[str, Any]],
    gt_intervals: list[dict[str, Any]],
) -> dict[str, Any]:
    pred_intervals = _valid_intervals(pred_intervals)
    gt_intervals = _valid_intervals(gt_intervals)
    if not pred_intervals and not gt_intervals:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "matches": [],
        }

    matches = match_intervals(pred_intervals, gt_intervals)
    tp = len(matches)
    fp = max(0, len(pred_intervals) - tp)
    fn = max(0, len(gt_intervals) - tp)
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matches": matches,
    }


def compute_boundary_iou(matches: list[dict[str, Any]]) -> float:
    if not matches:
        return 0.0
    return float(sum(float(item.get("iou", 0.0)) for item in matches) / len(matches))


def _intervals_from_event_list(events: Any) -> list[dict[str, Any]]:
    if not isinstance(events, list):
        return []
    intervals: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        start = _coerce_int(event.get("start"))
        end = _coerce_int(event.get("end"))
        if start is None or end is None:
            continue
        interval: dict[str, Any] = {"start": start, "end": end}
        if "type" in event:
            interval["type"] = canonicalize_type_name(str(event["type"]))
        intervals.append(interval)
    return intervals


def _intervals_from_interval_list(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    intervals: list[dict[str, Any]] = []
    for item in items:
        bounds = _interval_bounds(item, allow_reversed=True)
        if bounds is None:
            continue
        start, end = bounds
        interval = dict(item) if isinstance(item, dict) else {"start": start, "end": end}
        interval["start"] = start
        interval["end"] = end
        if "type" in interval:
            interval["type"] = canonicalize_type_name(str(interval["type"]))
        intervals.append(interval)
    return intervals


def _looks_like_point_labels(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    return all(not isinstance(item, (dict, list, tuple)) for item in value)


def ground_truth_to_intervals(gt_json_or_labels: Any) -> list[dict[str, Any]]:
    if gt_json_or_labels is None:
        return []

    if isinstance(gt_json_or_labels, str):
        text = gt_json_or_labels.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                return ground_truth_to_intervals(json.loads(text))
            except json.JSONDecodeError:
                return []
        return []

    if _looks_like_point_labels(gt_json_or_labels):
        return merge_point_labels_to_intervals(gt_json_or_labels)

    if isinstance(gt_json_or_labels, list):
        return _intervals_from_interval_list(gt_json_or_labels)

    if not isinstance(gt_json_or_labels, dict):
        return []

    for key in ("ground_truth", "target"):
        if key in gt_json_or_labels:
            return ground_truth_to_intervals(gt_json_or_labels[key])
    if "evidence" in gt_json_or_labels:
        return extract_intervals(gt_json_or_labels)
    if "events" in gt_json_or_labels:
        return _intervals_from_event_list(gt_json_or_labels["events"])
    for key in ("point_labels", "labels"):
        if key in gt_json_or_labels:
            return merge_point_labels_to_intervals(gt_json_or_labels[key])
    for key in ("intervals", "anomaly_intervals"):
        if key in gt_json_or_labels:
            return _intervals_from_interval_list(gt_json_or_labels[key])
    if "raw_sample" in gt_json_or_labels:
        return ground_truth_to_intervals(gt_json_or_labels["raw_sample"])
    return []


def _confidence_score_for_intervals(
    pred_intervals: list[dict[str, Any]],
    matches: list[dict[str, Any]],
) -> float:
    if not pred_intervals:
        return 1.0
    matched_pred = {int(item["pred_index"]) for item in matches}
    scores: list[float] = []
    for index, interval in enumerate(pred_intervals):
        confidence = _coerce_float(interval.get("confidence"))
        if confidence is None:
            confidence = 0.5
        confidence = _clip01(confidence)
        target = 1.0 if index in matched_pred else 0.0
        scores.append(1.0 - abs(confidence - target))
    return float(sum(scores) / len(scores))


def compute_confidence_score(pred_json: Any, matches: list[dict[str, Any]]) -> float:
    return _confidence_score_for_intervals(extract_intervals(pred_json), matches)


def compute_explanation_consistency(pred_json: Any) -> float:
    if not isinstance(pred_json, dict):
        return 0.0
    summary = pred_json.get("summary", "")
    if not isinstance(summary, str) or not summary.strip():
        return 0.0
    summary_text = summary.lower()
    intervals = extract_intervals(pred_json)
    if not intervals:
        no_anomaly_markers = ("no anomaly", "normal", "no anomalous", "nothing anomal")
        return 1.0 if any(marker in summary_text for marker in no_anomaly_markers) else 0.5

    scores: list[float] = []
    for interval in intervals:
        checks = 0.0
        total = 3.0
        type_name = interval.get("type")
        if isinstance(type_name, str):
            canonical_type = canonicalize_type_name(type_name)
            type_markers = {canonical_type}
            if canonical_type == "freq":
                type_markers.add("frequency")
            if canonical_type == "range":
                type_markers.add("level")
            checks += float(any(marker in summary_text for marker in type_markers))
        start = _coerce_int(interval.get("start"))
        end = _coerce_int(interval.get("end"))
        if start is not None:
            checks += float(str(start) in summary_text)
        if end is not None:
            checks += float(str(end) in summary_text)
        scores.append(checks / total)
    return float(sum(scores) / len(scores))


def _clip_intervals_to_seq_len(
    intervals: list[dict[str, Any]],
    seq_len: int,
) -> tuple[list[dict[str, Any]], int, int]:
    seq_len = max(1, int(seq_len))
    clipped: list[dict[str, Any]] = []
    clipped_count = 0
    invalid_order_count = 0
    for interval in intervals:
        bounds = _interval_bounds(interval, allow_reversed=True)
        if bounds is None:
            continue
        start, end = bounds
        if start > end:
            invalid_order_count += 1
            continue
        clipped_start = max(0, min(seq_len - 1, start))
        clipped_end = max(0, min(seq_len - 1, end))
        if clipped_start != start or clipped_end != end:
            clipped_count += 1
        copied = dict(interval)
        copied["start"] = clipped_start
        copied["end"] = clipped_end
        clipped.append(copied)
    return clipped, clipped_count, invalid_order_count


def _length_penalty(output_text: Any) -> float:
    if isinstance(output_text, str):
        text = output_text
    else:
        try:
            text = json.dumps(output_text, ensure_ascii=False)
        except (TypeError, ValueError):
            # Invalid programmatic actions still need to receive the invalid
            # reward, rather than failing while constructing diagnostic fields.
            text = str(output_text)
    token_estimate = max(1, len(text.split()))
    return min(2.0, max(0.0, (token_estimate - 120) / 120))


def compute_grounder_reward_details(
    pred_output: Any,
    gt_json_or_labels: Any,
    seq_len: int,
    *,
    reward_weights: dict[str, float] | None = None,
    annotation_scope: str = "full_grounding",
) -> dict[str, Any]:
    if annotation_scope not in {"localization_only", "full_grounding", "full_semantic"}:
        raise ValueError(f"Unknown annotation scope: {annotation_scope}")
    if type(seq_len) is not int or seq_len <= 0:
        raise ValueError("seq_len must be a positive integer.")
    resolved_reward_weights = resolve_reward_weights(reward_weights)
    pred_json: dict[str, Any] | None = None
    json_valid = False
    try:
        pred_json = parse_json_object(pred_output)
        json_valid = True
        if annotation_scope == "localization_only":
            from .supervision import validate_localization_payload

            # Metric projection can evaluate Stage-2 full-schema predictions,
            # but reward must require the requested localization-only schema.
            validate_localization_payload(pred_json, seq_len)
        else:
            parse_grounding_output(pred_json, series_length=seq_len)
    except (ValueError, TypeError) as exc:
        # Diagnostics never salvage an invalid action into a positive reward.
        # In particular, do not clip out-of-window predictions into correct ones.
        raw_intervals = extract_intervals(pred_json) if pred_json is not None else []
        _, clipped_count, invalid_order_count = _clip_intervals_to_seq_len(raw_intervals, seq_len)
        return {
            "reward": -1.0,
            "reward_weights": resolved_reward_weights,
            "reward_terms": {},
            "json_valid": float(json_valid),
            "schema_valid": 0.0,
            "schema_reward": 0.0,
            "event_precision": 0.0,
            "event_recall": 0.0,
            "event_f1": 0.0,
            "boundary_iou": 0.0,
            "confidence_score": 0.0,
            "explanation_consistency": 0.0,
            "hallucination_penalty": 1.0,
            "length_penalty": _length_penalty(pred_output),
            "errors": [str(exc)],
            "pred_intervals": [],
            "gt_intervals": [],
            "matches": [],
            "clipped_interval_count": clipped_count,
            "invalid_order_count": invalid_order_count,
        }

    errors: list[str] = []
    schema_valid = True
    evidence_usable = True
    schema_reward = 1.0

    raw_pred_intervals = extract_intervals(pred_json) if evidence_usable else []
    pred_intervals, clipped_count, invalid_order_count = _clip_intervals_to_seq_len(raw_pred_intervals, seq_len)
    raw_gt_intervals = ground_truth_to_intervals(gt_json_or_labels)
    gt_intervals, _, _ = _clip_intervals_to_seq_len(raw_gt_intervals, seq_len)

    if evidence_usable:
        event_scores = compute_event_f1(pred_intervals, gt_intervals)
        matches = event_scores["matches"]
        boundary_iou = (
            1.0
            if not pred_intervals and not gt_intervals
            else compute_boundary_iou(matches)
        )
        confidence_score = (
            1.0
            if not pred_intervals and not gt_intervals
            else 0.0
            if not pred_intervals and gt_intervals
            else _confidence_score_for_intervals(pred_intervals, matches)
        )
        explanation_consistency = (
            0.0
            if annotation_scope == "localization_only" or (not pred_intervals and gt_intervals)
            else compute_explanation_consistency(pred_json)
        )
    else:
        event_scores = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": len(gt_intervals),
            "matches": [],
        }
        matches = []
        boundary_iou = 0.0
        confidence_score = 0.0
        explanation_consistency = 0.0

    false_positive_count = max(0, len(pred_intervals) - len(matches))
    false_positive_rate = 0.0 if not pred_intervals else false_positive_count / len(pred_intervals)
    hallucination_penalty = min(
        2.0,
        false_positive_rate + (0.10 * clipped_count) + (0.25 * invalid_order_count),
    )
    length_penalty = _length_penalty(pred_output)

    reward_terms = {
        "event_f1": float(event_scores["f1"]),
        "boundary_iou": float(boundary_iou),
        "hallucination_penalty": float(hallucination_penalty),
    }
    reward = round(math.fsum(
        float(resolved_reward_weights[key]) * float(reward_terms[key])
        for key in ACTIVE_REWARD_TERMS
    ), 12)
    return {
        "reward": float(reward),
        "reward_weights": resolved_reward_weights,
        "reward_terms": reward_terms,
        "json_valid": 1.0,
        "schema_valid": float(schema_valid),
        "schema_reward": schema_reward,
        "event_precision": float(event_scores["precision"]),
        "event_recall": float(event_scores["recall"]),
        "event_f1": float(event_scores["f1"]),
        "boundary_iou": float(boundary_iou),
        "confidence_score": float(confidence_score),
        "explanation_consistency": float(explanation_consistency),
        "hallucination_penalty": float(hallucination_penalty),
        "length_penalty": float(length_penalty),
        "errors": errors,
        "pred_intervals": pred_intervals,
        "gt_intervals": gt_intervals,
        "matches": matches,
        "clipped_interval_count": clipped_count,
        "invalid_order_count": invalid_order_count,
    }


def compute_grounder_reward(
    pred_output: Any,
    gt_json_or_labels: Any,
    seq_len: int,
    *,
    reward_weights: dict[str, float] | None = None,
) -> float:
    return float(
        compute_grounder_reward_details(
            pred_output,
            gt_json_or_labels,
            seq_len,
            reward_weights=reward_weights,
        )["reward"]
    )


def resolve_boundary_aware_reward_weights(weights: dict[str, float] | None = None) -> dict[str, float]:
    return resolve_boundary_reward_weights(weights)


def compute_boundary_aware_grounder_reward_details(
    pred_output: Any,
    gt_json_or_labels: Any,
    seq_len: int = DEFAULT_SERIES_LENGTH,
    *,
    tau_match: float = DEFAULT_TAU_MATCH,
    max_index: int = DEFAULT_MAX_INDEX,
    reward_weights: dict[str, float] | None = None,
    prediction_schema: str = DEFAULT_PREDICTION_SCHEMA,
    annotation_scope: str = "full_grounding",
) -> dict[str, Any]:
    return _compute_boundary_aware_reward_details(
        pred_output,
        gt_json_or_labels,
        tau_match=tau_match,
        series_length=seq_len,
        max_index=max_index,
        reward_weights=reward_weights,
        prediction_schema=prediction_schema,
        annotation_scope=annotation_scope,
    )


def compute_boundary_aware_grounder_reward(
    pred_output: Any,
    gt_json_or_labels: Any,
    seq_len: int = DEFAULT_SERIES_LENGTH,
    *,
    tau_match: float = DEFAULT_TAU_MATCH,
    max_index: int = DEFAULT_MAX_INDEX,
    reward_weights: dict[str, float] | None = None,
    prediction_schema: str = DEFAULT_PREDICTION_SCHEMA,
    annotation_scope: str = "full_grounding",
) -> float:
    return float(
        compute_boundary_aware_grounder_reward_details(
            pred_output,
            gt_json_or_labels,
            seq_len,
            tau_match=tau_match,
            max_index=max_index,
            reward_weights=reward_weights,
            prediction_schema=prediction_schema,
            annotation_scope=annotation_scope,
        )["reward"]
    )
