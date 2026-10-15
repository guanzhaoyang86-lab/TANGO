from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ts_grounder.event_metrics import compute_boundary_aware_reward_details, score_prediction_output


def test_localization_scoring_never_invents_semantic_labels():
    target = {"evidence": [{"start": 10, "end": 20}]}
    scored = score_prediction_output(pred_output=json.dumps(target), gt=target, annotation_scope="localization_only")
    assert scored["event_f1"] == 1.0 and scored["schema_compliant"]
    assert scored["gt_evidence"] == target["evidence"] == scored["pred_evidence"]
    assert scored["error_type"] == "correct_abnormal"


def test_stage2_semantic_output_projects_for_metrics_but_not_reward():
    target = {"evidence": [{"start": 10, "end": 20}]}
    prediction = {"evidence": [{"start": 10, "end": 20, "type": "trend", "strength": "strong", "direction": "up"}], "summary": "Some anomaly."}
    scored = score_prediction_output(pred_output=json.dumps(prediction), gt=target, annotation_scope="localization_only")
    assert scored["event_f1"] == 1.0 and scored["json_valid"] and not scored["schema_compliant"]
    assert scored["error_type"] == "correct_abnormal"
    reward = compute_boundary_aware_reward_details(json.dumps(prediction), target, annotation_scope="localization_only")
    assert reward["reward"] == 0 and reward["schema_valid"] == 0 and reward["json_valid"] == 1
    assert reward["type_score"] is None


def test_localization_reward_valid_correct_output_and_strict_prose_rejection():
    target = {"evidence": [{"start": 10, "end": 20}]}
    reward = compute_boundary_aware_reward_details(json.dumps(target), target, annotation_scope="localization_only")
    assert reward["reward"] == 1.0 and reward["type_score"] is None
    for text in ("prefix " + json.dumps(target), json.dumps(target) + " trailing", '{"evidence":[{"start":true,"end":20}]}'):
        scored = score_prediction_output(pred_output=text, gt=target, annotation_scope="localization_only")
        assert not scored["json_valid"] and scored["error_type"] == "invalid_json"


def test_localization_groundtruth_cannot_include_placeholder_semantics():
    target = {"evidence": [{"start": 10, "end": 20, "type": "unknown"}]}
    try:
        score_prediction_output(pred_output='{"evidence":[]}', gt=target, annotation_scope="localization_only")
    except ValueError:
        return
    raise AssertionError("Ground truth with placeholder semantic type must be rejected")


def test_localization_nonzero_semantic_reward_is_rejected():
    try:
        compute_boundary_aware_reward_details('{"evidence":[]}', {"evidence": []}, annotation_scope="localization_only", reward_weights={"type": 0.1})
    except ValueError:
        return
    raise AssertionError("Cannot optimize semantic labels that do not exist")
