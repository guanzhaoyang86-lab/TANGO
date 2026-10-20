from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ts_grounder.rl_reward import (  # noqa: E402
    compute_grounder_reward,
    compute_grounder_reward_details,
    interval_iou,
    merge_point_labels_to_intervals,
    parse_model_output,
    validate_schema,
)


def _json(payload: dict) -> str:
    return json.dumps(payload)


def _event(start: int, end: int, type_name: str, **extra) -> dict:
    return {
        "start": start,
        "end": end,
        "type": type_name,
        "strength": "moderate",
        "direction": "upward",
        **extra,
    }


def _valid_prediction() -> dict:
    return {
        "evidence": [_event(5, 8, "point", confidence=0.9)],
        "summary": "A point anomaly occurs from index 5 to 8.",
    }


def test_correct_json_and_interval_gets_high_reward() -> None:
    pred = {
        "evidence": [
            {
                "start": 10,
                "end": 20,
                "type": "trend",
                "confidence": 0.9,
                "strength": "moderate",
                "direction": "upward shift",
            }
        ],
        "summary": "A trend anomaly occurs from index 10 to 20.",
    }
    gt = {"evidence": [{"start": 10, "end": 20, "type": "trend"}], "summary": "target"}

    details = compute_grounder_reward_details(_json(pred), gt, seq_len=64)

    assert validate_schema(parse_model_output(_json(pred)))
    assert details["event_f1"] == 1.0
    assert details["boundary_iou"] == 1.0
    assert details["reward"] == 0.8


def test_invalid_json_gets_negative_reward() -> None:
    details = compute_grounder_reward_details("not a json object", {"evidence": []}, seq_len=32)

    assert details["json_valid"] == 0.0
    assert details["reward"] == -1.0


def test_start_after_end_is_schema_invalid_and_penalized() -> None:
    pred = {
        "evidence": [_event(20, 10, "point", confidence=0.8)],
        "summary": "A point anomaly occurs from index 20 to 10.",
    }
    gt = {"evidence": [{"start": 10, "end": 20, "type": "point"}]}

    details = compute_grounder_reward_details(_json(pred), gt, seq_len=64)

    assert details["schema_valid"] == 0.0
    assert details["invalid_order_count"] == 1
    assert details["event_f1"] == 0.0
    assert details["reward"] < 0.2


def test_partial_overlap_has_fractional_boundary_iou() -> None:
    pred_interval = {"start": 10, "end": 20}
    gt_interval = {"start": 15, "end": 25}
    gt = {"evidence": [{"start": 15, "end": 25, "type": "range"}]}
    pred = {
        "evidence": [_event(10, 20, "range", confidence=0.8)],
        "summary": "A range anomaly occurs from index 10 to 20.",
    }

    details = compute_grounder_reward_details(_json(pred), gt, seq_len=64)

    assert interval_iou(pred_interval, gt_interval) == 6 / 16
    assert details["event_f1"] == 1.0
    assert details["boundary_iou"] == 6 / 16


def test_multiple_predictions_with_one_hallucinated_false_positive() -> None:
    pred = {
        "evidence": [
            _event(10, 20, "trend", confidence=0.9),
            _event(40, 45, "trend", confidence=0.9),
        ],
        "summary": "Trend anomalies occur from index 10 to 20 and 40 to 45.",
    }
    gt = {"evidence": [{"start": 10, "end": 20, "type": "trend"}]}

    details = compute_grounder_reward_details(_json(pred), gt, seq_len=64)

    assert details["event_precision"] == 0.5
    assert details["event_recall"] == 1.0
    assert details["hallucination_penalty"] > 0.0
    assert details["reward"] < compute_grounder_reward(_json({"evidence": pred["evidence"][:1], "summary": "A trend anomaly occurs from index 10 to 20."}), gt, 64)


def test_empty_ground_truth_and_empty_prediction_get_positive_reward() -> None:
    pred = {"evidence": [], "summary": "No anomaly is detected."}
    labels = [0, 0, 0, 0]

    assert merge_point_labels_to_intervals(labels) == []
    details = compute_grounder_reward_details(_json(pred), labels, seq_len=4)

    assert details["event_f1"] == 1.0
    assert details["reward"] == 0.8


def test_empty_ground_truth_with_false_positive_is_penalized() -> None:
    pred = {
        "evidence": [_event(1, 2, "freq", confidence=0.9)],
        "summary": "A freq anomaly occurs from index 1 to 2.",
    }

    details = compute_grounder_reward_details(_json(pred), [0, 0, 0, 0], seq_len=4)

    assert details["event_f1"] == 0.0
    assert details["hallucination_penalty"] == 1.0
    assert details["reward"] < 0.5


def test_type_score_is_not_part_of_reward_details() -> None:
    gt = {"evidence": [{"start": 5, "end": 8, "type": "point"}]}
    pred_correct = {
        "evidence": [_event(5, 8, "point")],
        "summary": "A point anomaly occurs from index 5 to 8.",
    }

    details = compute_grounder_reward_details(_json(pred_correct), gt, seq_len=32)

    assert "type_score" not in details
    assert "type_score" not in details["reward_terms"]
    assert "type_score" not in details["reward_weights"]


def test_type_score_reward_weight_is_rejected() -> None:
    try:
        compute_grounder_reward_details(
            _json({"evidence": [], "summary": "No anomaly is detected."}),
            {"evidence": []},
            seq_len=32,
            reward_weights={"type_score": 1.0},
        )
    except ValueError as exc:
        assert "Unsupported reward weight key: type_score" in str(exc)
    else:
        raise AssertionError("type_score reward weight must not be accepted")


def test_negative_hallucination_penalty_weight_reduces_reward() -> None:
    gt = {"evidence": [{"start": 5, "end": 8, "type": "point"}]}
    pred = {
        "evidence": [
            _event(5, 8, "point"),
            _event(20, 22, "point"),
        ],
        "summary": "Point anomalies occur from index 5 to 8 and 20 to 22.",
    }

    without_penalty = compute_grounder_reward_details(
        _json(pred),
        gt,
        seq_len=32,
        reward_weights={"hallucination_penalty": 0.0},
    )
    with_penalty = compute_grounder_reward_details(
        _json(pred),
        gt,
        seq_len=32,
        reward_weights={"hallucination_penalty": -0.10},
    )

    assert with_penalty["hallucination_penalty"] == 0.5
    assert with_penalty["reward"] == without_penalty["reward"] - 0.05


@pytest.mark.parametrize("scope", ["full_grounding", "full_semantic"])
@pytest.mark.parametrize("wrapper", [
    "Explanation: {}",
    "{}\nThe above is my answer.",
    "```json\n{}\n```",
    "{} {}",
])
def test_prose_wrapped_or_concatenated_json_cannot_receive_reward(scope, wrapper) -> None:
    prediction = _valid_prediction()
    text = wrapper.replace("{}", _json(prediction))
    details = compute_grounder_reward_details(text, prediction, 32, annotation_scope=scope)

    assert details["json_valid"] == 0.0
    assert details["schema_valid"] == 0.0
    assert details["reward"] == -1.0
    assert details["event_f1"] == 0.0
    assert details["boundary_iou"] == 0.0


@pytest.mark.parametrize("scope", ["full_grounding", "full_semantic"])
@pytest.mark.parametrize(("field", "value"), [
    ("type", "invented-type"),
    ("strength", ""),
    ("direction", "  "),
    ("start", -1),
    ("end", 32),
    ("start", True),
    ("start", "5"),
    ("end", 8.0),
    ("confidence", float("nan")),
    ("confidence", float("inf")),
])
@pytest.mark.parametrize("as_text", [True, False])
def test_invalid_grounding_actions_are_gated_before_scoring(scope, field, value, as_text) -> None:
    prediction = _valid_prediction()
    prediction["evidence"][0][field] = value
    output = _json(prediction) if as_text else prediction
    details = compute_grounder_reward_details(
        output, _valid_prediction(), 32, annotation_scope=scope,
    )

    assert details["schema_valid"] == 0.0
    assert details["reward"] == -1.0
    assert details["event_f1"] == 0.0
    assert details["boundary_iou"] == 0.0
    assert details["reward_terms"] == {}


@pytest.mark.parametrize("summary", ["", "   ", None])
def test_missing_or_empty_summary_cannot_receive_positive_reward(summary) -> None:
    prediction = _valid_prediction()
    if summary is None:
        del prediction["summary"]
    else:
        prediction["summary"] = summary
    details = compute_grounder_reward_details(prediction, _valid_prediction(), 32)
    assert details["json_valid"] == 1.0
    assert details["reward"] == -1.0
    assert details["schema_valid"] == 0.0


@pytest.mark.parametrize("field", ["strength", "direction"])
def test_missing_required_semantic_fields_cannot_receive_positive_reward(field) -> None:
    prediction = _valid_prediction()
    del prediction["evidence"][0][field]
    assert compute_grounder_reward_details(prediction, _valid_prediction(), 32)["reward"] == -1.0


def test_duplicate_json_keys_cannot_receive_reward() -> None:
    text = '{"evidence": [], "evidence": [], "summary": "Normal."}'
    details = compute_grounder_reward_details(text, {"evidence": []}, 32)
    assert details["json_valid"] == 0.0
    assert details["reward"] == -1.0


def test_invalid_empty_action_does_not_receive_normal_window_reward() -> None:
    details = compute_grounder_reward_details({"evidence": []}, [0, 0, 0, 0], 4)
    assert details["reward"] == -1.0
    assert details["event_f1"] == 0.0


def test_localization_reward_still_requires_exact_localization_schema() -> None:
    target = {"evidence": [{"start": 5, "end": 8}]}
    valid = compute_grounder_reward_details(target, target, 32, annotation_scope="localization_only")
    projected_only = compute_grounder_reward_details(
        _valid_prediction(), target, 32, annotation_scope="localization_only",
    )
    prose = compute_grounder_reward_details(
        "Answer: " + _json(target), target, 32, annotation_scope="localization_only",
    )
    assert valid["reward"] == 0.8
    assert projected_only["reward"] == -1.0
    assert prose["reward"] == -1.0


def test_valid_whitespace_wrapped_json_preserves_legacy_reward() -> None:
    prediction = _valid_prediction()
    details = compute_grounder_reward_details(" \n" + _json(prediction) + "\t ", prediction, 32)
    assert details["schema_valid"] == 1.0
    assert details["reward"] == 0.8
    assert details["confidence_score"] == 0.9


@pytest.mark.parametrize("prediction", [None, [], {"evidence": {1, 2}}, 42])
def test_non_json_programmatic_actions_receive_invalid_reward(prediction) -> None:
    details = compute_grounder_reward_details(prediction, _valid_prediction(), 32)
    assert details["json_valid"] == 0.0
    assert details["reward"] == -1.0
