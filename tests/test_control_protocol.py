from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ts_grounder.control_protocol import (build_prompt_schedules, fingerprint_checkpoint,
                                         index_train_records, load_prompt_schedule)
from ts_grounder.schema import GROUNDING_CONTRACT_VERSION


def _reject(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except (ValueError, TypeError):
        return
    raise AssertionError("Invalid controlled-run input was accepted")


def _fixture(root: Path):
    image = root / "input.png"
    image.write_bytes(b"image-content-for-hashing")
    train = [{"id": f"sample-{i}", "split": "train", "system_prompt": "system",
              "user_prompt": f"sequence-{i}", "assistant_text": '{"evidence":[]}',
              "target": {"evidence": []}, "image_path": str(image),
              "metadata": {"annotation_scope": "localization_only"},
              "raw_sample": {"series": [i], "context": {"materialized_subset_split": "train"}}}
             for i in range(20)]
    residual = copy.deepcopy(train)
    classes = ["false_negative", "boundary_error", "false_positive", "correct_abnormal", "correct_normal"]
    for i, record in enumerate(residual):
        record["error_type"] = classes[i % len(classes)]
        record["residual_mining"] = {"checkpoint_fingerprint": "a" * 64,
                                     "source_split": "train", "input_sha256": "b" * 64,
                                     "grounding_contract_version": GROUNDING_CONTRACT_VERSION}
    return train, residual


def _build(train, residual, **overrides):
    options = {"seed": 2026, "epochs": 3, "prompts_per_epoch": 10,
               "checkpoint_fingerprint": "a" * 64}
    options.update(overrides)
    return build_prompt_schedules(train, residual, **options)


def test_schedule_determinism_equal_exposure_and_exact_d_g_order():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        train, residual = _fixture(root)
        matched, random = _build(train, residual)
        assert (matched, random) == _build(list(reversed(train)), list(reversed(residual)))
        assert matched["total_prompt_exposures"] == random["total_prompt_exposures"] == 30
        assert matched["equal_compute_claim"] is False
        assert all(len(ids) == len(set(ids)) == 10 for ids in matched["epochs"] + random["epochs"])
        assert all(set(ids) <= {record["id"] for record in train} for ids in matched["epochs"] + random["epochs"])
        path = root / "schedule.json"
        path.write_text(json.dumps(matched), encoding="utf-8")
        d_epochs = load_prompt_schedule(path, train)
        g_epochs = load_prompt_schedule(path, residual)
        assert [[item["id"] for item in epoch] for epoch in d_epochs] == [[item["id"] for item in epoch] for epoch in g_epochs]
        assert matched["epochs"] != _build(train, residual, seed=2027)[0]["epochs"]


def test_reject_heldout_duplicate_missing_and_mutated_training_records():
    with tempfile.TemporaryDirectory() as directory:
        train, residual = _fixture(Path(directory))
        heldout = copy.deepcopy(train)
        heldout[0]["split"] = "test"
        _reject(index_train_records, heldout)
        _reject(index_train_records, train + [train[0]])
        _reject(_build, train, residual[:-1])
        mutated = copy.deepcopy(residual)
        mutated[0]["target"] = {"evidence": [{"start": 1, "end": 1}]}
        _reject(_build, train, mutated)
        mutated = copy.deepcopy(residual)
        mutated[0]["user_prompt"] += " changed"
        _reject(_build, train, mutated)
        mutated = copy.deepcopy(train)
        mutated[0]["raw_sample"]["context"]["materialized_subset_split"] = "test"
        _reject(index_train_records, mutated)


def test_reject_full_population_or_noninteger_prompt_budget():
    with tempfile.TemporaryDirectory() as directory:
        train, residual = _fixture(Path(directory))
        for value in (0, -1, 20, 21, 1.5, True):
            _reject(_build, train, residual, prompts_per_epoch=value)
        for value in (0, -1, 1.5, True):
            _reject(_build, train, residual, epochs=value)


def test_reject_residual_predictions_from_different_checkpoint():
    with tempfile.TemporaryDirectory() as directory:
        train, residual = _fixture(Path(directory))
        residual[0]["residual_mining"]["checkpoint_fingerprint"] = "c" * 64
        _reject(_build, train, residual)


def test_reject_residual_pool_and_schedule_from_old_grounding_contract():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        train, residual = _fixture(root)
        for old_version in (None, "grounding-output-v1"):
            old_residual = copy.deepcopy(residual)
            old_residual[0]["residual_mining"]["grounding_contract_version"] = old_version
            _reject(_build, train, old_residual)
            schedule = _build(train, residual)[0]
            schedule["grounding_contract_version"] = old_version
            path = root / "old_schedule.json"
            path.write_text(json.dumps(schedule), encoding="utf-8")
            _reject(load_prompt_schedule, path, train)


def test_schedule_rejects_mutation_duplicate_unknown_id_and_budget():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        train, residual = _fixture(root)
        schedule = _build(train, residual)[0]
        path = root / "schedule.json"
        mutations = []
        duplicate = copy.deepcopy(schedule)
        duplicate["epochs"][0][1] = duplicate["epochs"][0][0]
        mutations.append(duplicate)
        unknown = copy.deepcopy(schedule)
        unknown["epochs"][0][0] = "heldout-id"
        mutations.append(unknown)
        wrong_total = copy.deepcopy(schedule)
        wrong_total["total_prompt_exposures"] += 1
        mutations.append(wrong_total)
        for invalid in mutations:
            path.write_text(json.dumps(invalid), encoding="utf-8")
            _reject(load_prompt_schedule, path, train)
        path.write_text(json.dumps(schedule), encoding="utf-8")
        train[0]["target"] = {"evidence": [{"start": 0, "end": 0}]}
        _reject(load_prompt_schedule, path, train)


def test_schedule_rejects_same_path_changed_image_content():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        train, residual = _fixture(root)
        schedule = _build(train, residual)[0]
        path = root / "schedule.json"
        path.write_text(json.dumps(schedule), encoding="utf-8")
        (root / "input.png").write_bytes(b"different-image-content")
        _reject(load_prompt_schedule, path, train)


def test_checkpoint_fingerprint_content_identity_changes_and_adapter_rejection():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name in ("one", "two"):
            checkpoint = root / name
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text('{"model_type":"test"}', encoding="utf-8")
            (checkpoint / "model.safetensors").write_bytes(b"test-only-weight-bytes")
            (checkpoint / "tokenizer.json").write_text("{}", encoding="utf-8")
        assert fingerprint_checkpoint(root / "one") == fingerprint_checkpoint(root / "two")
        (root / "two" / "model.safetensors").write_bytes(b"changed-weight-bytes")
        assert fingerprint_checkpoint(root / "one") != fingerprint_checkpoint(root / "two")
        baseline = fingerprint_checkpoint(root / "one")
        (root / "one" / "trainer_state.json").write_text("{}", encoding="utf-8")
        assert baseline == fingerprint_checkpoint(root / "one")
        (root / "one" / "adapter_config.json").write_text("{}", encoding="utf-8")
        _reject(fingerprint_checkpoint, root / "one")


def _checkpoint_fixture(root: Path) -> Path:
    checkpoint = root / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_type":"test"}', encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"test-only-weight-bytes")
    (checkpoint / "tokenizer.json").write_text("{}", encoding="utf-8")
    return checkpoint


def test_checkpoint_templates_add_modify_rename_remove_and_copy_identity():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = _checkpoint_fixture(root)
        empty = fingerprint_checkpoint(checkpoint)
        template = checkpoint / "chat_template.jinja"
        template.write_text("original {{ messages }}", encoding="utf-8")
        baseline = fingerprint_checkpoint(checkpoint)
        assert baseline != empty
        template.write_text("modified {{ messages }}", encoding="utf-8")
        assert fingerprint_checkpoint(checkpoint) != baseline
        template.write_text("original {{ messages }}", encoding="utf-8")
        assert fingerprint_checkpoint(checkpoint) == baseline

        named_dir = checkpoint / "chat_templates"
        named_dir.mkdir()
        named = named_dir / "tool_use.jinja"
        named.write_text("tool {{ messages }}", encoding="utf-8")
        with_named = fingerprint_checkpoint(checkpoint)
        assert with_named != baseline
        renamed = named.with_name("custom.jinja")
        named.rename(renamed)
        assert fingerprint_checkpoint(checkpoint) != with_named
        renamed.rename(named)
        assert fingerprint_checkpoint(checkpoint) == with_named
        copied = root / "relocated"
        shutil.copytree(checkpoint, copied)
        assert fingerprint_checkpoint(copied) == with_named
        named.unlink()
        assert fingerprint_checkpoint(checkpoint) == baseline
        template.unlink()
        assert fingerprint_checkpoint(checkpoint) == empty


def test_checkpoint_fingerprint_processor_files_and_bounded_traversal():
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = _checkpoint_fixture(Path(directory))
        for name in ("preprocessor_config.json", "processor_config.json",
                     "video_preprocessor_config.json", "generation_config.json", "tokenizer_config.json"):
            before = fingerprint_checkpoint(checkpoint)
            (checkpoint / name).write_text('{"setting":1}', encoding="utf-8")
            after = fingerprint_checkpoint(checkpoint)
            assert after != before
            (checkpoint / name).write_text('{"setting":2}', encoding="utf-8")
            assert fingerprint_checkpoint(checkpoint) != after
        baseline = fingerprint_checkpoint(checkpoint)
        # Training checkpoints can contain huge weights and must not be scanned.
        historical = checkpoint / "checkpoint-100"
        historical.mkdir()
        (historical / "model.safetensors").write_bytes(b"unrelated-history")
        for name in ("optimizer.bin", "training_args.bin", "trainer_state.json"):
            (checkpoint / name).write_text("training-only-state", encoding="utf-8")
        assert fingerprint_checkpoint(checkpoint) == baseline


def test_checkpoint_fingerprint_rejects_reparse_points_in_nested_assets(monkeypatch):
    import stat
    from types import SimpleNamespace

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = _checkpoint_fixture(Path(directory))
        templates = checkpoint / "chat_templates"
        templates.mkdir()
        template = templates / "tool_use.jinja"
        template.write_text("{{ messages }}", encoding="utf-8")
        nested = templates / "nested"
        nested.mkdir()
        original_lstat = Path.lstat

        for linked_path in (checkpoint, templates, nested, template):
            def fake_lstat(path):
                if path == linked_path:
                    result = original_lstat(path)
                    return SimpleNamespace(
                        st_mode=result.st_mode,
                        st_file_attributes=getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400),
                    )
                return original_lstat(path)

            monkeypatch.setattr(Path, "lstat", fake_lstat)
            _reject(fingerprint_checkpoint, checkpoint)


def test_checkpoint_fingerprint_rejects_symlink_template_and_directory(tmp_path):
    import pytest

    checkpoint = _checkpoint_fixture(tmp_path)
    external = tmp_path / "external.jinja"
    external.write_text("{{ messages }}", encoding="utf-8")
    link = checkpoint / "chat_template.jinja"
    try:
        link.symlink_to(external)
    except (OSError, NotImplementedError):
        pytest.skip("This OS/account does not permit creating symbolic links")
    _reject(fingerprint_checkpoint, checkpoint)
    link.unlink()
    templates = checkpoint / "chat_templates"
    templates.mkdir()
    (templates / "tool_use.jinja").symlink_to(external)
    _reject(fingerprint_checkpoint, checkpoint)
    (templates / "tool_use.jinja").unlink()
    templates.rmdir()
    templates.symlink_to(tmp_path, target_is_directory=True)
    _reject(fingerprint_checkpoint, checkpoint)
