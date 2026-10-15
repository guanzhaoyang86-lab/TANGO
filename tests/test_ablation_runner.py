"""Orchestration unit tests; subprocess doubles do not represent model training."""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("tango_ablation_runner_tests", ROOT / "scripts/run_ablation.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _reject(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except (ValueError, FileExistsError):
        return
    raise AssertionError("Expected a rejected experimental configuration")


def _config():
    return runner.load_yaml(ROOT / "configs/controls.yaml")


def test_plan_has_four_models_and_is_read_only_without_checkpoint_or_terms():
    args = runner.build_parser().parse_args(["--action", "plan"])
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), patch.object(runner, "execute", side_effect=AssertionError("plan launched subprocess")), patch.object(runner, "fingerprint_checkpoint", side_effect=AssertionError("plan accessed weights")):
        runner.run(args)
    plan = json.loads(stream.getvalue())
    assert set(plan["variants"]) == {"A", "D", "F", "G"}
    assert plan["equal_compute"] is False
    assert plan["shared_D_G_schedule"] == "schedules/residual.json"


def test_config_rejects_unmatched_policy_or_unavailable_semantic_reward():
    good = _config()
    assert runner.validate_control_config(good) == good
    for key, value in (("top_p", 0.9), ("num_generations", 1), ("max_new_tokens", 256),
                       ("epochs", True), ("prompt_fraction", 1.0), ("learning_rate", float("nan")),
                       ("reference_penalty_coef", -0.1), ("tau_good", -1)):
        candidate = copy.deepcopy(good)
        candidate[key] = value
        _reject(runner.validate_control_config, candidate)
    candidate = copy.deepcopy(good)
    candidate["reward_weights"]["type"] = 0.1
    _reject(runner.validate_control_config, candidate)
    _reject(runner.validate_control_config, {**good, "unknown_option": True})


def test_json_config_preserves_scientific_notation_as_number():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "controls.json"
        cfg = _config()
        cfg["learning_rate"] = 1e-6
        path.write_text(json.dumps(cfg), encoding="utf-8")
        loaded = runner.load_yaml(path)
        assert loaded["learning_rate"] == 1e-6 and isinstance(loaded["learning_rate"], float)
        runner.validate_control_config(loaded)


def test_nonplan_requires_source_terms_and_never_silently_starts_training():
    args = runner.build_parser().parse_args(["--action", "train"])
    with patch.object(runner, "execute", side_effect=AssertionError("unapproved run launched")):
        _reject(runner.run, args)


def test_miniature_is_blocked_and_split_id_overlap_is_rejected():
    _reject(runner.validate_source, ROOT / "dataset/anomaly_db_v1")
    assert set(runner.validate_source(ROOT / "dataset/anomaly_db_v1", allow_demo=True)) == {"train", "val", "test"}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for split in ("train", "val", "test"):
            (root / f"{split}.json").write_text(json.dumps([{"sample_id": split}]), encoding="utf-8")
        assert len(runner.validate_source(root)) == 3
        (root / "test.json").write_text('[{"sample_id":"train"}]', encoding="utf-8")
        _reject(runner.validate_source, root)


def test_execute_strips_legacy_overrides_and_records_failure_cost_without_claiming_gpu_hours():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with patch.dict(os.environ, {"TS_GROUNDER_MODEL_PATH": "wrong-checkpoint", "TS_GROUNDER_SOURCE_ROOT": "heldout"}), patch.object(runner.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as subprocess_run:
            runner.execute([sys.executable, "--version"], root, "unit-test")
        env = subprocess_run.call_args.kwargs["env"]
        assert not any(key.startswith("TS_GROUNDER_") for key in env)
        receipt = json.loads((root / "execution_costs.jsonl").read_text(encoding="utf-8"))
        assert receipt["returncode"] == 0 and receipt["gpu_hours"] is None
        assert receipt["compute_matched_claim"] is False
        with patch.object(runner.subprocess, "run", return_value=SimpleNamespace(returncode=2)):
            try:
                runner.execute([sys.executable, "--version"], root, "failure-unit-test")
            except RuntimeError:
                pass
            else:
                raise AssertionError("Failed subprocess must stop the experiment")
        assert len((root / "execution_costs.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_sft_config_uses_train_cache_no_random_truncation_or_semantic_auxiliary_loss():
    cfg = runner.sft_config(Path("dataset"), Path("stage2"), Path("run/D"), 2026, _config())
    assert cfg["data"]["include_indexed_series_text"] and cfg["data"]["indexed_series_precision"] == 6
    assert cfg["data"]["max_train_samples"] is None
    assert not cfg["data"]["balanced_by_type"] and not cfg["data"]["include_qa_pairs"]
    assert cfg["training"]["evidence_loss_weight"] == cfg["training"]["summary_semantic_loss_weight"] == 0
    assert cfg["training"]["gradient_accumulation_steps"] == cfg["training"]["per_device_train_batch_size"] == 1
    assert cfg["training"]["evaluation_strategy"] == "no"
    assert cfg["model"]["model_name_or_path"] == str(Path("stage2"))
