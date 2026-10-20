"""One exact-JSON full-grounding contract for optimization, mining and eval."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ts_grounder.event_metrics import parse_event_output, score_prediction_output
from ts_grounder.rl_reward import compute_boundary_aware_grounder_reward_details
from ts_grounder.schema import GROUNDING_CONTRACT_VERSION, parse_grounding_output
from ts_grounder.vlm_eval import evaluate_predictions, parse_generated_result


def _target():
    return {
        "evidence": [{"start": 10, "end": 20, "type": "trend", "strength": "moderate", "direction": "upward"}],
        "summary": "A trend anomaly occurs.",
    }


def _evaluate(text, target=None, length=256, scope="full_grounding"):
    target = _target() if target is None else target
    record = {"id": "sample", "target": target,
              "metadata": {"task_type": "grounding", "series_length": length, "annotation_scope": scope}}
    return evaluate_predictions([record], [{"id": "sample", "generated_text": text}])["metrics"]


def _bad_outputs():
    text = json.dumps(_target())
    outputs = [
        ("prefix", "Result: " + text),
        ("suffix", text + " explanation"),
        ("markdown", "```json\n" + text + "\n```"),
        ("multiple_objects", text + text),
        ("invalid_json", "{broken"),
        ("array", "[" + text + "]"),
        ("null", "null"),
        ("duplicate_summary", text[:-1] + ', "summary": "different"}'),
        ("duplicate_boundary", text.replace('"start": 10', '"start": 9, "start": 10')),
        ("nan", text[:-1] + ', "extra": NaN}'),
        ("infinity", text[:-1] + ', "extra": Infinity}'),
    ]
    for name, field, value in [
        ("empty_summary", "summary", ""),
        ("whitespace_summary", "summary", "  \t"),
        ("nonstring_summary", "summary", 7),
        ("empty_strength", "strength", ""),
        ("whitespace_direction", "direction", "\n"),
        ("nonstring_strength", "strength", True),
        ("nonstring_direction", "direction", 4),
        ("unknown_type", "type", "mystery"),
        ("empty_type", "type", ""),
        ("nonstring_type", "type", False),
        ("boolean_start", "start", True),
        ("float_start", "start", 10.0),
        ("string_end", "end", "20"),
        ("negative_start", "start", -1),
        ("reversed_bounds", "start", 21),
        ("out_of_range", "end", 256),
    ]:
        payload = _target()
        container = payload if field == "summary" else payload["evidence"][0]
        container[field] = value
        outputs.append((name, json.dumps(payload)))
    for field in ("start", "end", "type", "strength", "direction"):
        payload = _target()
        del payload["evidence"][0][field]
        outputs.append(("missing_" + field, json.dumps(payload)))
    outputs.extend([
        ("missing_summary", json.dumps({"evidence": _target()["evidence"]})),
        ("missing_evidence", '{"summary":"Missing evidence."}'),
        ("wrong_evidence_type", '{"evidence":{},"summary":"Wrong evidence."}'),
        ("wrong_item_type", '{"evidence":[1],"summary":"Wrong evidence item."}'),
    ])
    return outputs


@pytest.mark.parametrize("name,text", _bad_outputs(), ids=[case[0] for case in _bad_outputs()])
def test_invalid_predictions_are_rejected_by_reward_mining_and_evaluation(name, text):
    with pytest.raises(ValueError):
        parse_grounding_output(text, series_length=256)
    with pytest.raises(ValueError):
        parse_generated_result(text, series_length=256)
    parsed = parse_event_output(text)
    assert not parsed.json_valid and not parsed.schema_compliant
    for scope in ("full_grounding", "full_semantic"):
        reward = compute_boundary_aware_grounder_reward_details(text, _target(), annotation_scope=scope)
        mining = score_prediction_output(pred_output=text, gt=_target(), annotation_scope=scope)
        metrics = _evaluate(text, scope=scope)
        assert reward["reward"] == 0.0 and reward["schema_valid"] == 0.0
        assert mining["error_type"] == "invalid_json" and mining["event_f1"] == 0.0
        assert metrics["parse_success_rate"] == 0.0 and metrics["point_f1"] == 0.0


@pytest.mark.parametrize("type_name,canonical", [("point", "point"), ("trend", "trend"), ("freq", "freq"),
                                                ("range", "range"), ("frequency", "freq"), ("level", "range")])
def test_valid_outputs_have_identical_canonical_types_across_paths(type_name, canonical):
    target = _target()
    target["evidence"][0]["type"] = canonical
    prediction = copy.deepcopy(target)
    prediction["evidence"][0]["type"] = type_name
    text = " \n" + json.dumps(prediction) + "\t"
    parsed = parse_event_output(text)
    assert parsed.json_valid and parsed.schema_compliant
    assert parsed.events[0]["type"] == canonical
    assert parse_generated_result(text).evidence[0].type == canonical
    assert compute_boundary_aware_grounder_reward_details(text, target)["reward"] == 1.0
    assert score_prediction_output(pred_output=text, gt=target)["error_type"] == "correct_abnormal"
    assert _evaluate(text, target=target)["exact_match"] == 1.0


def test_short_window_bounds_do_not_depend_on_default_max_index():
    target = _target()
    target["evidence"][0].update(start=1, end=3)
    prediction = copy.deepcopy(target)
    prediction["evidence"][0]["end"] = 8
    text = json.dumps(prediction)
    assert compute_boundary_aware_grounder_reward_details(text, target, seq_len=8)["reward"] == 0.0
    assert score_prediction_output(pred_output=text, gt=target, series_length=8)["error_type"] == "invalid_json"
    assert _evaluate(text, target=target, length=8)["parse_success_rate"] == 0.0


def test_valid_reward_formula_is_unchanged():
    target = _target()
    prediction = copy.deepcopy(target)
    prediction["evidence"][0]["start"] = 11
    reward = compute_boundary_aware_grounder_reward_details(json.dumps(prediction), target)
    expected = 0.60 * (20 / 21) + 0.25 * (10 / 11) + 0.15 * (1 - 1 / 512)
    assert reward["reward"] == pytest.approx(expected)


@pytest.mark.parametrize("text", ['{"evidence":[],"summary":""}', 'Explanation: {"evidence":[],"summary":"Normal."}'])
def test_invalid_normal_predictions_cannot_receive_perfect_reward(text):
    target = {"evidence": [], "summary": "No anomaly is detected."}
    assert compute_boundary_aware_grounder_reward_details(text, target)["reward"] == 0.0
    assert score_prediction_output(pred_output=text, gt=target)["error_type"] == "invalid_json"
    assert _evaluate(text, target=target)["point_f1"] == 0.0


def test_programmatic_payload_and_valid_normal_output_are_supported():
    target = {"evidence": [], "summary": "No anomaly is detected."}
    assert parse_grounding_output(target).to_dict() == target
    assert compute_boundary_aware_grounder_reward_details(target, target)["reward"] == 1.0
    assert _evaluate(json.dumps(target), target=target)["point_f1"] == 1.0
    assert GROUNDING_CONTRACT_VERSION == "grounding-output-v2"


def test_legacy_events_recovery_requires_explicit_schema():
    text = 'Explanation: {"events":[{"start":10,"end":20,"type":"trend"}]}'
    assert not parse_event_output(text).json_valid
    assert not parse_event_output(text, prediction_schema="auto").json_valid
    assert parse_event_output(text, prediction_schema="events").json_valid


def test_actual_residual_miner_does_not_mark_prose_wrapped_prediction_correct(tmp_path):
    from PIL import Image
    from ts_grounder.vlm_data import _record_from_sample
    from ts_grounder.vlm_prompting import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT

    spec = importlib.util.spec_from_file_location("grounding_contract_miner", ROOT / "scripts/build_residual_pool.py")
    miner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(miner)
    Image.new("RGB", (64, 32), "white").save(tmp_path / "sample.png")
    sample = {
        "sample_id": "sample", "series": [float(i) for i in range(256)],
        "events": [{"start": 10, "end": 20, "type": "trend",
                    "verbal_tags": {"strength": "moderate", "direction": "upward"}}],
        "image_path": "sample.png", "context": {"series_length": 256},
    }
    record = _record_from_sample(sample, source_root=tmp_path, output_root=tmp_path / "cache", split="train",
                                system_prompt=DEFAULT_SYSTEM_PROMPT, user_prompt=DEFAULT_USER_PROMPT)
    (tmp_path / "train.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    args = miner.build_parser().parse_args([
        "--model-path", str(tmp_path / "checkpoint"),
        "--real-train-data", str(tmp_path / "train.jsonl"), "--output", str(tmp_path / "pool.jsonl"),
    ])
    text = "Here is the answer: " + record["assistant_text"]
    with patch.object(miner, "checkpoint_provenance", return_value={"kind": "test_double", "sha256": "abc"}), \
            patch.object(miner, "_generate_outputs", return_value=[text]):
        miner.build_residual_pool(args)
    result = miner.load_jsonl(tmp_path / "pool.jsonl")[0]
    assert result["error_type"] == "invalid_json"
    assert not result["json_valid"]
