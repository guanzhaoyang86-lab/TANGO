#!/usr/bin/env python3
"""Verify appendix metric/reward formulas against the production helpers.

This script intentionally imports the residual-mining and boundary-aware
reward implementation.  It does not provide an independent replacement for
those functions.  The small amount of local counting below only exposes the
point-level TP/FP/FN values underlying the imported point-mask construction.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ts_grounder.event_metrics import (  # noqa: E402
    DEFAULT_MAX_INDEX,
    DEFAULT_SERIES_LENGTH,
    DEFAULT_TAU_GOOD,
    DEFAULT_TAU_MATCH,
    _events_to_point_mask,
    compute_boundary_aware_reward_details,
    events_from_ground_truth,
    point_prf_for_events,
    score_prediction_output,
)


PAPER_REWARD_WEIGHTS = {
    "point": 0.45,
    "event": 0.0,
    "iou": 0.40,
    "boundary": 0.15,
    "type": 0.0,
}


def evidence(start: int, end: int, type_name: str = "trend") -> dict[str, Any]:
    return {
        "start": start,
        "end": end,
        "type": type_name,
        "strength": "moderate",
        "direction": "up",
    }


def prediction(items: list[dict[str, Any]]) -> str:
    return json.dumps(
        {
            "evidence": items,
            "summary": "No anomaly is detected." if not items else "An anomaly is detected.",
        }
    )


def point_counts(
    pred_events: list[dict[str, Any]],
    gt_events: list[dict[str, Any]],
) -> tuple[int, int, int]:
    pred_mask = _events_to_point_mask(pred_events, series_length=DEFAULT_SERIES_LENGTH)
    gt_mask = _events_to_point_mask(gt_events, series_length=DEFAULT_SERIES_LENGTH)
    tp = sum(bool(pred) and bool(gt) for pred, gt in zip(pred_mask, gt_mask))
    fp = sum(bool(pred) and not bool(gt) for pred, gt in zip(pred_mask, gt_mask))
    fn = sum(not bool(pred) and bool(gt) for pred, gt in zip(pred_mask, gt_mask))
    return int(tp), int(fp), int(fn)


CASES: list[dict[str, Any]] = [
    {
        "name": "perfect_prediction",
        "gt": {"evidence": [evidence(10, 19)]},
        "pred": prediction([evidence(10, 19)]),
        "category": "correct_abnormal",
        "reward": 1.0,
    },
    {
        "name": "partial_overlap",
        "gt": {"evidence": [evidence(10, 19)]},
        "pred": prediction([evidence(15, 24)]),
        "category": "boundary_error",
    },
    {
        "name": "disjoint_intervals",
        "gt": {"evidence": [evidence(10, 19)]},
        "pred": prediction([evidence(30, 39)]),
        "category": "false_negative",
        "reward": 0.0,
    },
    {
        "name": "missed_event",
        "gt": {"evidence": [evidence(10, 19)]},
        "pred": prediction([]),
        "category": "false_negative",
        "reward": 0.0,
    },
    {
        "name": "false_positive_on_normal",
        "gt": {"evidence": []},
        "pred": prediction([evidence(10, 19)]),
        "category": "false_positive",
        "reward": 1.0 - (10.0 / DEFAULT_SERIES_LENGTH) - 0.2,
    },
    {
        "name": "empty_gt_and_prediction",
        "gt": {"evidence": []},
        "pred": prediction([]),
        "category": "correct_normal",
        "reward": 1.0,
    },
    {
        "name": "invalid_json",
        "gt": {"evidence": [evidence(10, 19)]},
        "pred": "not a JSON object",
        "category": "invalid_json",
        "reward": 0.0,
    },
    {
        "name": "multiple_gt_and_predictions",
        "gt": {"evidence": [evidence(10, 19), evidence(40, 49, "range")]},
        "pred": prediction(
            [
                evidence(41, 50, "range"),
                evidence(9, 18),
                evidence(80, 85, "point"),
            ]
        ),
        "category": "false_positive",
    },
]


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def main() -> None:
    print(
        "Protocol: inclusive intervals; greedy one-to-one matching; "
        f"tau_match={DEFAULT_TAU_MATCH}; tau_good={DEFAULT_TAU_GOOD}; "
        f"T={DEFAULT_SERIES_LENGTH}; weights={PAPER_REWARD_WEIGHTS}"
    )
    for case in CASES:
        scored = score_prediction_output(
            pred_output=case["pred"],
            gt=case["gt"],
            tau_match=DEFAULT_TAU_MATCH,
            tau_good=DEFAULT_TAU_GOOD,
            series_length=DEFAULT_SERIES_LENGTH,
            max_index=DEFAULT_MAX_INDEX,
            prediction_schema="evidence",
        )
        reward = compute_boundary_aware_reward_details(
            case["pred"],
            case["gt"],
            tau_match=DEFAULT_TAU_MATCH,
            series_length=DEFAULT_SERIES_LENGTH,
            max_index=DEFAULT_MAX_INDEX,
            reward_weights=PAPER_REWARD_WEIGHTS,
            prediction_schema="evidence",
        )
        gt_events = events_from_ground_truth(case["gt"])
        pred_events = list(scored["pred_evidence"])
        tp, fp, fn = point_counts(pred_events, gt_events)
        point = point_prf_for_events(
            pred_events,
            gt_events,
            series_length=DEFAULT_SERIES_LENGTH,
        )
        matches = [
            (int(item["pred_index"]), int(item["gt_index"]), float(item["iou"]))
            for item in scored["matches"]
        ]

        assert scored["error_type"] == case["category"], case["name"]
        assert close(float(point["precision"]), float(reward["point_precision"])), case["name"]
        assert close(float(point["recall"]), float(reward["point_recall"])), case["name"]
        assert close(float(point["f1"]), float(reward["point_f1"])), case["name"]
        if "reward" in case:
            assert close(float(reward["reward"]), float(case["reward"])), case["name"]

        print(f"\n[{case['name']}]")
        print(
            f"point TP={tp} FP={fp} FN={fn}; "
            f"P={point['precision']:.6f} R={point['recall']:.6f} F1={point['f1']:.6f}"
        )
        print(
            f"matches={matches}; matched_mean_IoU={reward['mean_iou']:.6f}; "
            f"boundary_MAE={reward.get('boundary_mae', 0.0):.6f}; "
            f"boundary_score={reward['boundary_reward']:.6f}"
        )
        print(
            f"residual={scored['error_type']}; valid_gate={reward['g_valid']:.0f}; "
            f"final_reward={reward['reward']:.6f}"
        )

    # The paper-facing three-field names are not accepted by the current
    # residual/reward parser.  Keep this check explicit so schema drift is
    # detected instead of silently changing the reported metric protocol.
    paper_schema_only = json.dumps(
        {
            "anomaly_label": [0] * DEFAULT_SERIES_LENGTH,
            "grounded_evidence": [],
            "summary": "No anomaly is detected.",
        }
    )
    schema_check = score_prediction_output(pred_output=paper_schema_only, gt={"evidence": []})
    assert schema_check["json_valid"] is False
    assert schema_check["error_type"] == "invalid_json"
    ignored_label_check = score_prediction_output(
        pred_output=json.dumps(
            {
                "anomaly_label": [1],
                "evidence": [],
                "summary": "No anomaly is detected.",
            }
        ),
        gt={"evidence": []},
    )
    assert ignored_label_check["json_valid"] is True
    assert ignored_label_check["error_type"] == "correct_normal"
    free_string_attributes = score_prediction_output(
        pred_output=prediction(
            [
                {
                    "start": 10,
                    "end": 19,
                    "type": "not-in-taxonomy",
                    "strength": "not-in-taxonomy",
                    "direction": "not-in-taxonomy",
                }
            ]
        ),
        gt={"evidence": [evidence(10, 19, "not-in-taxonomy")]},
    )
    assert free_string_attributes["json_valid"] is True
    print(
        "\n[schema_consistency_check]\n"
        "anomaly_label + grounded_evidence + summary => invalid_json "
        "(current code expects evidence + summary)\n"
        "anomaly_label length 1 + evidence + summary => valid "
        "(anomaly_label is ignored)\n"
        "out-of-vocabulary string attributes => valid "
        "(active parser checks string type, not categorical membership)"
    )
    print(f"\nAll {len(CASES)} metric/reward cases passed.")


if __name__ == "__main__":
    main()
