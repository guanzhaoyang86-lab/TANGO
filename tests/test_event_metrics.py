from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ts_grounder.event_metrics import (  # noqa: E402
    compute_boundary_aware_reward_details,
    interval_iou,
    score_prediction_output,
)


def _evidence_item(start: int, end: int, type_name: str = "trend") -> dict:
    return {
        "start": start,
        "end": end,
        "type": type_name,
        "strength": "moderate",
        "direction": "upward shift",
    }


def _pred(evidence: list[dict]) -> str:
    return json.dumps({"evidence": evidence, "summary": "A trend anomaly occurs."})


def _gt(evidence: list[dict]) -> dict:
    return {"evidence": evidence, "summary": "target"}


def _error_type(gt_evidence: list[dict], pred_evidence: list[dict]) -> str:
    return str(
        score_prediction_output(
            pred_output=_pred(pred_evidence),
            gt=_gt(gt_evidence),
            tau_match=0.1,
            tau_good=0.5,
            series_length=256,
            max_index=255,
        )["error_type"]
    )


def test_closed_interval_iou() -> None:
    assert interval_iou({"start": 80, "end": 105}, {"start": 70, "end": 130}) == 26 / 61


def test_false_negative_sanity_case() -> None:
    assert _error_type([_evidence_item(80, 105)], []) == "false_negative"


def test_false_positive_sanity_case() -> None:
    assert _error_type([], [_evidence_item(40, 60)]) == "false_positive"


def test_boundary_error_sanity_case() -> None:
    assert (
        _error_type(
            [_evidence_item(80, 105)],
            [_evidence_item(70, 130)],
        )
        == "boundary_error"
    )


def test_type_error_sanity_case() -> None:
    assert (
        _error_type(
            [_evidence_item(80, 105, "trend")],
            [_evidence_item(82, 106, "range")],
        )
        == "type_error"
    )


def test_correct_abnormal_sanity_case() -> None:
    assert (
        _error_type(
            [_evidence_item(80, 105, "trend")],
            [_evidence_item(82, 106, "trend")],
        )
        == "correct_abnormal"
    )


def test_correct_normal_sanity_case() -> None:
    assert _error_type([], []) == "correct_normal"


def test_evidence_schema_is_default() -> None:
    scored = score_prediction_output(
        pred_output=json.dumps(
            {
                "evidence": [_evidence_item(80, 105, "trend")],
                "summary": "A trend anomaly occurs from index 80 to 105.",
            }
        ),
        gt={"evidence": [{"start": 80, "end": 105, "type": "trend"}]},
    )

    assert scored["json_valid"] is True
    assert scored["error_type"] == "correct_abnormal"


def test_invalid_json_gets_zero_boundary_aware_reward() -> None:
    details = compute_boundary_aware_reward_details(
        "not json",
        {"evidence": [{"start": 80, "end": 105, "type": "trend"}]},
    )

    assert details["json_valid"] == 0.0
    assert details["reward"] == 0.0


def test_missing_evidence_is_invalid_json() -> None:
    scored = score_prediction_output(
        pred_output=json.dumps({"summary": "No anomaly detected."}),
        gt={"evidence": []},
    )

    assert scored["json_valid"] is False
    assert scored["error_type"] == "invalid_json"
    assert scored["event_f1"] == 0.0
    assert scored["mean_iou"] == 0.0


def test_missing_summary_is_invalid_json() -> None:
    scored = score_prediction_output(
        pred_output=json.dumps({"evidence": []}),
        gt={"evidence": []},
    )

    assert scored["json_valid"] is False
    assert scored["error_type"] == "invalid_json"


def test_missing_strength_direction_is_invalid_json() -> None:
    scored = score_prediction_output(
        pred_output=json.dumps(
            {
                "evidence": [{"start": 80, "end": 105, "type": "trend"}],
                "summary": "A trend anomaly occurs.",
            }
        ),
        gt={"evidence": [_evidence_item(80, 105)]},
    )

    assert scored["json_valid"] is False
    assert scored["error_type"] == "invalid_json"


def test_start_after_end_is_invalid_json() -> None:
    scored = score_prediction_output(
        pred_output=_pred([_evidence_item(105, 80)]),
        gt={"evidence": [_evidence_item(80, 105)]},
    )

    assert scored["json_valid"] is False
    assert scored["error_type"] == "invalid_json"


def test_out_of_range_index_is_invalid_json() -> None:
    scored = score_prediction_output(
        pred_output=_pred([_evidence_item(40, 256)]),
        gt={"evidence": []},
    )

    assert scored["json_valid"] is False
    assert scored["error_type"] == "invalid_json"
