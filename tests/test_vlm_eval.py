from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ts_grounder.schema import GroundingResult  # noqa: E402
from ts_grounder.vlm_eval import evaluate_predictions, parse_generated_result, score_prediction  # noqa: E402


def _item(start: int, end: int, type_name: str = "trend") -> dict:
    return {
        "start": start,
        "end": end,
        "type": type_name,
        "strength": "moderate",
        "direction": "upward",
    }


def test_empty_object_is_not_valid_grounding_output() -> None:
    try:
        parse_generated_result("{}")
    except ValueError:
        pass
    else:
        raise AssertionError("An empty object must not satisfy the output schema.")


def test_extra_text_is_not_accepted_as_exact_json() -> None:
    payload = json.dumps({"evidence": [], "summary": "No anomaly is detected."})
    try:
        parse_generated_result(f"Result: {payload}")
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("Extra text must invalidate the exact-JSON output contract.")


def test_correct_normal_prediction_has_perfect_point_scores() -> None:
    normal = GroundingResult.no_anomaly()
    scored = score_prediction(target=normal, prediction=normal, series_length=256)
    assert scored["point_precision"] == 1.0
    assert scored["point_recall"] == 1.0
    assert scored["point_f1"] == 1.0


def test_reordered_evidence_is_scored_semantically() -> None:
    target = GroundingResult.from_dict(
        {
            "evidence": [_item(10, 20, "trend"), _item(100, 110, "range")],
            "summary": "Two anomalies are detected.",
        }
    )
    prediction = GroundingResult.from_dict(
        {
            "evidence": [_item(100, 110, "range"), _item(10, 20, "trend")],
            "summary": "Two anomalies are detected.",
        }
    )
    scored = score_prediction(target=target, prediction=prediction, series_length=256)
    assert scored["exact_match"] == 1.0
    assert scored["type_accuracy"] == 1.0
    assert scored["start_mae"] == 0.0
    assert scored["end_mae"] == 0.0


def test_out_of_range_interval_is_rejected() -> None:
    payload = json.dumps(
        {
            "evidence": [_item(250, 256)],
            "summary": "An anomaly is detected.",
        }
    )
    try:
        parse_generated_result(payload, series_length=256)
    except ValueError:
        pass
    else:
        raise AssertionError("Intervals outside the series must be rejected.")


def test_missing_predictions_count_as_zero_for_grounding_metrics() -> None:
    records = [
        {
            "id": "sample-1",
            "target": {"evidence": [_item(10, 20)], "summary": "An anomaly is detected."},
            "metadata": {"series_length": 256, "task_type": "grounding", "anomaly_type": "trend"},
        }
    ]
    result = evaluate_predictions(records, [])
    metrics = result["metrics"]
    assert metrics["parse_success_rate"] == 0.0
    assert metrics["point_f1"] == 0.0
    assert metrics["start_mae"] == 256.0
