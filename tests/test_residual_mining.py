"""CPU-only checks for leakage guards, immutable targets and provenance."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location("tango_residual_miner_tests", ROOT / "scripts/build_residual_pool.py")
miner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(miner)


def _raises(exception, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except exception:
        return
    raise AssertionError(f"Expected {exception.__name__}")


def test_train_source_rejects_heldout_and_duplicate_ids():
    miner._validate_train_source([{"id": "one", "split": "train"}], Path("train.jsonl"), None)
    _raises(ValueError, miner._validate_train_source, [{"id": "one"}], Path("val.json"), "train")
    _raises(ValueError, miner._validate_train_source, [{"id": "one", "split": "test"}], Path("train.jsonl"), None)
    _raises(ValueError, miner._validate_train_source, [{"id": "one"}, {"id": "one"}], Path("train.json"), None)
    _raises(ValueError, miner._validate_train_source, [{"id": "one"}], Path("renamed.json"), "train")
    miner._validate_train_source([{"id": "one", "split": "train"}], Path("renamed.jsonl"), "train")
    _raises(ValueError, miner._validate_train_source, [{"id": "one", "split": "train", "raw_sample": {"context": {"materialized_subset_split": "test"}}}], Path("train.jsonl"), None)
    _raises(ValueError, miner._validate_train_source, [{"id": "one", "sample_id": "different"}], Path("train.jsonl"), None)
    _raises(ValueError, miner._validate_train_source, [{"id": 3}], Path("train.jsonl"), None)


def test_conflicting_target_and_ground_truth_cannot_relabel_a_sample():
    _raises(ValueError, miner._record_ground_truth, {"target": {"evidence": []}, "ground_truth": {"evidence": [{"start": 1, "end": 2}]}})


def test_cached_predictions_require_complete_id_and_digest_provenance():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        predictions = root / "predictions.jsonl"
        manifest = root / "manifest.json"
        predictions.write_text(json.dumps({"id": "b", "generated_text": "B"}) + "\n" + json.dumps({"id": "a", "generated_text": "A"}) + "\n", encoding="utf-8")
        metadata = {"input_sha256": "input", "prepared_records_sha256": "prepared", "checkpoint_fingerprint": "checkpoint", "predictions_sha256": miner._sha256_file(predictions)}
        manifest.write_text(json.dumps(metadata), encoding="utf-8")
        kwargs = dict(ids=["a", "b"], input_sha256="input", prepared_records_sha256="prepared", checkpoint_fingerprint="checkpoint")
        assert miner._cached_predictions(predictions, manifest, **kwargs) == ["A", "B"]
        _raises(ValueError, miner._cached_predictions, predictions, manifest, **{**kwargs, "ids": ["a"]})
        _raises(ValueError, miner._cached_predictions, predictions, manifest, **{**kwargs, "checkpoint_fingerprint": "wrong"})
        predictions.write_text(json.dumps({"id": "a", "generated_text": "A"}) + "\n" + json.dumps({"id": "a", "generated_text": "A"}) + "\n", encoding="utf-8")
        metadata["predictions_sha256"] = miner._sha256_file(predictions)
        manifest.write_text(json.dumps(metadata), encoding="utf-8")
        _raises(ValueError, miner._cached_predictions, predictions, manifest, **kwargs)


def _prepared_record(directory: Path, localization: bool):
    from PIL import Image
    from ts_grounder.vlm_data import _record_from_sample
    from ts_grounder.vlm_prompting import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT

    Image.new("RGB", (64, 32), "white").save(directory / "sample.png")
    sample = {"sample_id": "sample", "series": [float(i) for i in range(256)],
              "point_labels": [0] * 10 + [1] * 3 + [0] * 243,
              "events": [{"start": 10, "end": 12}], "image_path": "sample.png",
              "context": {"series_length": 256}}
    if localization:
        sample.update(annotation_scope="localization_only", semantic_label_available=False)
    else:
        sample["events"][0].update(type="point", verbal_tags={"strength": "moderate", "direction": "upward"})
    return _record_from_sample(sample, source_root=directory, output_root=directory / "cache", split="train",
                               system_prompt=DEFAULT_SYSTEM_PROMPT, user_prompt=DEFAULT_USER_PROMPT,
                               include_indexed_series_text=True, indexed_series_precision=6)


def _arguments(root: Path):
    return miner.build_parser().parse_args(["--model-path", str(root / "checkpoint"),
        "--real-train-data", str(root / "train.jsonl"), "--output", str(root / "pool.jsonl")])


def test_mining_preserves_exact_full_and_localization_targets():
    for localization in (False, True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = _prepared_record(root, localization)
            original = copy.deepcopy(record)
            (root / "train.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            args = _arguments(root)
            with patch.object(miner, "checkpoint_provenance", return_value={"kind": "test_double", "sha256": "abc"}), patch.object(miner, "_generate_outputs", return_value=[record["assistant_text"]]):
                summary = miner.build_residual_pool(args)
            result = miner.load_jsonl(root / "pool.jsonl")[0]
            assert result["target"] == original["target"] == result["ground_truth"]
            assert result["assistant_text"] == original["assistant_text"]
            assert result["raw_sample"] == original["raw_sample"]
            assert result["error_type"] == "correct_abnormal"
            assert result["residual_mining"]["source_split"] == "train"
            assert summary["checkpoint_fingerprint"] == "abc"
            assert summary["residual_sha256"] == miner._sha256_file(root / "pool.jsonl")
            if localization:
                assert set(result["ground_truth"]) == {"evidence"}
                assert all(set(event) == {"start", "end"} for event in result["gt_evidence"])
            _raises(FileExistsError, miner.build_residual_pool, args)


def test_generation_count_mismatch_is_not_silently_truncated():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        record = _prepared_record(root, True)
        (root / "train.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
        with patch.object(miner, "checkpoint_provenance", return_value={"sha256": "abc"}), patch.object(miner, "_generate_outputs", return_value=[]):
            _raises(ValueError, miner.build_residual_pool, _arguments(root))
        assert not (root / "pool.jsonl").exists()


def test_changed_input_is_rejected_before_pool_write():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        record = _prepared_record(root, True)
        source = root / "train.jsonl"
        source.write_text(json.dumps(record) + "\n", encoding="utf-8")
        def mutate(**kwargs):
            source.write_text(json.dumps(record) + "\n\n", encoding="utf-8")
            return [record["assistant_text"]]
        with patch.object(miner, "checkpoint_provenance", return_value={"sha256": "abc"}), patch.object(miner, "_generate_outputs", side_effect=mutate):
            _raises(RuntimeError, miner.build_residual_pool, _arguments(root))
        assert not (root / "pool.jsonl").exists()


def test_checkpoint_fingerprint_rejects_unpinned_hub_reference():
    _raises(ValueError, miner.checkpoint_provenance, "does-not-exist/model")
    provenance = miner.checkpoint_provenance("some/model", "a" * 40)
    assert provenance["kind"] == "pinned_hub"
    assert len(provenance["sha256"]) == 64
