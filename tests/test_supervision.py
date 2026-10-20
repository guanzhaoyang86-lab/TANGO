from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ts_grounder.supervision import (
    LOCALIZATION_SYSTEM_PROMPT, TARGET_ADAPTER_VERSION, annotation_scope,
    localization_target, project_localization_prediction, validate_generation_budget,
    validate_localization_payload, validate_prepared_record, forbidden_action_token_ids,
)
from ts_grounder.vlm_data import _record_from_sample, _record_from_qa_pair, build_vlm_sft_dataset
from ts_grounder.vlm_eval import evaluate_predictions
from ts_grounder.vlm_training import VisionChatCollator, generate_predictions, load_vlm_checkpoint
from ts_grounder.schema import GROUNDING_CONTRACT_VERSION


def make_sample(tmp_path, *, semantic=False, normal=False, many=False):
    image = tmp_path / "plot.png"
    Image.new("RGB", (32, 16), "white").save(image)
    intervals = [] if normal else ([{"start": idx * 4, "end": idx * 4} for idx in range(38)] if many else [{"start": 2, "end": 3}])
    labels = [0] * 256
    for item in intervals:
        labels[item["start"]:item["end"] + 1] = [1] * (item["end"] - item["start"] + 1)
    events = [{**item, "type": "trend", "verbal_tags": {"strength": "moderate", "direction": "upward"}} for item in intervals]
    return {
        "sample_id": "example", "series": [idx / 100 for idx in range(256)],
        "point_labels": labels, "events": events, "image_path": "plot.png",
        "context": {"series_length": 256, "interval_convention": "inclusive", "source_dataset": "synthetic" if semantic else "KPI"},
        "annotation_scope": "full_semantic" if semantic else "localization_only",
        "semantic_label_available": semantic,
    }


def make_record(tmp_path, **kwargs):
    return _record_from_sample(
        make_sample(tmp_path, **kwargs), source_root=tmp_path, output_root=None,
        split="train", system_prompt="Semantic prompt", user_prompt="Describe the anomaly",
    )


def test_loc_target_drops_semantic_placeholders_and_keeps_indices(tmp_path):
    record = make_record(tmp_path)
    assert record["target"] == {"evidence": [{"start": 2, "end": 3}]}
    assert record["system_prompt"] == LOCALIZATION_SYSTEM_PROMPT
    assert "255: 2.550000" in record["user_prompt"]
    assert record["metadata"]["indexed_series_precision"] == 6
    assert record["metadata"]["anomaly_type"] == "anomaly_unspecified"
    assert validate_prepared_record(record) == "localization_only"


def test_many_events_never_capped(tmp_path):
    assert len(make_record(tmp_path, many=True)["target"]["evidence"]) == 38


def test_normal_localization_has_empty_evidence(tmp_path):
    assert make_record(tmp_path, normal=True)["target"] == {"evidence": []}


def test_full_semantic_target_preserved(tmp_path):
    record = make_record(tmp_path, semantic=True)
    assert set(record["target"]) == {"evidence", "summary"}
    assert record["target"]["evidence"][0]["type"] == "trend"
    assert validate_prepared_record(record) == "full_grounding"


def test_loc_qa_rejected(tmp_path):
    with pytest.raises(ValueError, match="semantic QA"):
        _record_from_qa_pair(make_sample(tmp_path), {"question": "type?", "answer": "trend"},
                             source_root=tmp_path, output_root=None, split="train",
                             system_prompt="prompt", user_prompt_template="{question}")


def test_semantic_qa_kept_and_validated(tmp_path):
    record = _record_from_qa_pair(make_sample(tmp_path, semantic=True), {"question": "type?", "answer": "trend"},
                                 source_root=tmp_path, output_root=None, split="train",
                                 system_prompt="prompt", user_prompt_template="{question}")
    assert validate_prepared_record(record) == "full_grounding"
    assert record["assistant_text"] == "trend"


def test_annotation_conflicts_and_fake_booleans_rejected():
    with pytest.raises(ValueError):
        annotation_scope({"annotation_scope": "full_grounding", "semantic_label_available": False})
    with pytest.raises(ValueError):
        annotation_scope({"semantic_label_available": "false"})


@pytest.mark.parametrize("interval", [{"start": True, "end": 3}, {"start": 1.0, "end": 3}, {"start": 3, "end": 2}, {"start": 2, "end": 256}])
def test_invalid_localization_boundary_rejected(interval):
    with pytest.raises(ValueError):
        validate_localization_payload({"evidence": [interval]}, 256)


def test_point_labels_must_match_inclusive_events(tmp_path):
    sample = make_sample(tmp_path)
    sample["events"][0]["end"] = 4
    with pytest.raises(ValueError, match="disagrees"):
        localization_target(sample)


def test_old_cache_rejected_without_overwriting(tmp_path):
    output = tmp_path / "cache"
    output.mkdir()
    (output / "train.jsonl").write_text("old", encoding="utf-8")
    with pytest.raises(ValueError, match="Refusing"):
        build_vlm_sft_dataset(source_root=tmp_path, output_root=output)
    assert (output / "train.jsonl").read_text() == "old"


def test_unversioned_record_rejected(tmp_path):
    record = make_record(tmp_path)
    del record["metadata"]["target_adapter_version"]
    with pytest.raises(ValueError, match="Old/unversioned"):
        validate_prepared_record(record)


def test_loc_generation_budget_fails_before_model_loading(tmp_path):
    record = make_record(tmp_path)
    path = tmp_path / "test.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="2048"):
        generate_predictions(model_path="does-not-exist", dataset_jsonl=path,
                             output_path=tmp_path / "out.jsonl", max_new_tokens=256)
    validate_generation_budget([record], 2048)


def test_stage2_fullschema_projection_has_fair_loc_scores(tmp_path):
    record = make_record(tmp_path)
    prediction = {"evidence": [{"start": 2, "end": 3, "type": "trend", "strength": "strong"}], "summary": "text"}
    result = evaluate_predictions([record], [{"id": record["id"], "generated_text": json.dumps(prediction)}])
    assert result["metrics"]["point_f1"] == 1.0
    assert result["metrics"]["parse_success_rate"] == 1.0
    assert result["metrics"]["localization_schema_success_rate"] == 0.0
    assert result["metrics"]["type_accuracy"] is None
    assert result["metric_denominators"]["type_accuracy"] == 0


def test_missing_loc_predictions_penalized_without_fake_semantic_denominator(tmp_path):
    record = make_record(tmp_path)
    result = evaluate_predictions([record], [])
    assert result["metrics"]["point_f1"] == 0.0
    assert result["metrics"]["start_mae"] == 256.0
    assert result["metrics"]["type_accuracy"] is None


def test_invalid_loc_predictions_count_as_zero_not_dropped(tmp_path):
    record = make_record(tmp_path)
    another = copy.deepcopy(record)
    another["id"] = "second"
    result = evaluate_predictions([record, another], [
        {"id": record["id"], "generated_text": record["assistant_text"]},
        {"id": "second", "generated_text": "not JSON"},
    ])
    assert result["metrics"]["point_f1"] == 0.5
    assert result["metric_denominators"]["point_f1"] == 2


class CharProcessor:
    """Tiny deterministic tokenizer/processor; tensors are real torch, not doubles."""
    pad_token_id = 0
    eos_token_id = 0

    def __init__(self, side="right"):
        self.tokenizer = self
        self.padding_side = side

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = [item["text"] for msg in messages for item in msg["content"] if item["type"] == "text"]
        if add_generation_prompt:
            return "|".join(parts) + "|ASSISTANT|"
        return "|".join(parts[:-1]) + "|ASSISTANT|" + parts[-1] + "\0"

    def __call__(self, text=None, images=None, return_tensors=None, return_offsets_mapping=False, **kwargs):
        if isinstance(text, str):
            result = {"input_ids": [ord(ch) for ch in text]}
            if return_offsets_mapping:
                result["offset_mapping"] = [(idx, idx + 1) for idx in range(len(text))]
            return result
        import torch
        rows = [[ord(ch) for ch in value] for value in text]
        width = max(map(len, rows))
        ids, masks = [], []
        for row in rows:
            pads = width - len(row)
            ids.append(([0] * pads + row) if self.padding_side == "left" else (row + [0] * pads))
            masks.append(([0] * pads + [1] * len(row)) if self.padding_side == "left" else ([1] * len(row) + [0] * pads))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}


@pytest.mark.parametrize("side", ["left", "right"])
def test_collator_eos_padding_and_semantic_masks_real_torch(tmp_path, side):
    torch = pytest.importorskip("torch")
    records = [make_record(tmp_path), make_record(tmp_path, semantic=True)]
    records[0]["user_prompt"] += " A different prompt length."
    collator = VisionChatCollator(CharProcessor(side), evidence_loss_weight=1, summary_semantic_loss_weight=1)
    batch = collator(records)
    for idx, record in enumerate(records):
        # Keep one real EOS even though the EOS/PAD token ID is the same.
        assert (batch["labels"][idx] == 0).sum().item() == 1
        assert torch.all(batch["labels"][idx][batch["attention_mask"][idx] == 0] == -100)
        active_labels = batch["labels"][idx][batch["labels"][idx] != -100]
        decoded = "".join(chr(value) for value in active_labels.tolist())
        assert decoded == record["assistant_text"] + "\0"
        assert batch["evidence_mask"][idx].any()
    assert not batch["summary_mask"][0].any()
    assert batch["summary_mask"][1].any()


def test_collator_rejects_truncation(tmp_path):
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="truncation"):
        VisionChatCollator(CharProcessor(), processor_kwargs={"truncation": True})([make_record(tmp_path)])


def test_local_adapter_loads_base_then_peft_not_base_only(tmp_path, monkeypatch):
    calls = []
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    base = object()
    adapter = object()
    fake_transformers = SimpleNamespace(
        AutoModelForImageTextToText=SimpleNamespace(from_pretrained=lambda path, **kw: calls.append(("base", path, kw)) or base),
        AutoProcessor=SimpleNamespace(from_pretrained=lambda path, **kw: calls.append(("processor", path, kw)) or "processor"),
    )
    fake_peft = SimpleNamespace(
        PeftConfig=SimpleNamespace(from_pretrained=lambda path: SimpleNamespace(base_model_name_or_path="base-id")),
        PeftModel=SimpleNamespace(from_pretrained=lambda model, path, **kw: calls.append(("adapter", model, path, kw)) or adapter),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    processor, model = load_vlm_checkpoint(str(tmp_path))
    assert model is adapter
    assert calls[0][0:2] == ("base", "base-id")
    assert calls[1] == ("adapter", base, str(tmp_path), {"is_trainable": False})
    assert calls[2][0:2] == ("processor", "base-id")


def test_matched_sft_real_hf_trainer_runs_exact_order_and_updates_weights(tmp_path, monkeypatch):
    """CPU integration: real HF Trainer/autograd/optimizer, tiny model and fake processor.

    This checks wiring and gradients; it is not an end-to-end pretrained-VLM run.
    """
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers import PretrainedConfig, PreTrainedModel
    from transformers.modeling_outputs import CausalLMOutput
    from ts_grounder.control_protocol import fingerprint_checkpoint, json_sha256, record_fingerprint
    from ts_grounder.vlm_training import PreparedDataset, run_hf_vlm_sft

    class TinyModel(PreTrainedModel):
        config_class = PretrainedConfig

        def __init__(self, config):
            super().__init__(config)
            self.embed = torch.nn.Embedding(256, 8)
            self.dropout = torch.nn.Dropout(0.5)
            self.attention_dropout = 0.25
            self.head = torch.nn.Linear(8, 256)
            self.seen = []

        def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
            if self.training:
                self.seen.append("".join(chr(idx) for idx in input_ids[0].tolist()))
            logits = self.head(self.dropout(self.embed(input_ids)))
            loss = None if labels is None else torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, 256), labels[:, 1:].reshape(-1), ignore_index=-100,
            )
            return CausalLMOutput(loss=loss, logits=logits)

    stage2 = tmp_path / "stage2"
    config = PretrainedConfig(vocab_size=256, hidden_size=8, pad_token_id=0)
    model = TinyModel(config)
    model.save_pretrained(stage2)
    weights_before = model.head.weight.detach().clone()
    records = [make_record(tmp_path), make_record(tmp_path, normal=True)]
    for idx, record in enumerate(records):
        record["id"] = f"sample-{idx}"
        record["user_prompt"] = f"SAMPLE_{idx} " + record["user_prompt"]
    cache = tmp_path / "cache"
    cache.mkdir()
    for split in ("train", "val", "test"):
        (cache / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    fingerprints = {record["id"]: record_fingerprint(record) for record in records}
    schedule = {
        "schema_version": 1, "selection": "residual", "num_epochs": 2,
        "grounding_contract_version": GROUNDING_CONTRACT_VERSION,
        "prompts_per_epoch": 1, "total_prompt_exposures": 2,
        "epochs": [["sample-1"], ["sample-0"]],
        "sampling_without_replacement_within_epoch": True,
        "record_fingerprints": fingerprints, "train_population_sha256": json_sha256(fingerprints),
        "stage2_checkpoint_fingerprint": fingerprint_checkpoint(stage2),
    }
    schedule_path = tmp_path / "schedule.json"
    schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *a, **kw: CharProcessor())
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained", lambda *a, **kw: model)
    output = tmp_path / "run"
    output.mkdir()
    result = run_hf_vlm_sft({
        "seed": 7, "model": {"model_name_or_path": str(stage2), "trust_remote_code": False},
        "training": {"prompt_schedule_file": str(schedule_path), "max_steps": 2,
                     "evaluation_strategy": "no", "save_strategy": "no", "use_cpu": True,
                     "learning_rate": 0.01, "logging_steps": 1, "fixed_order": True,
                     "lr_scheduler_type": "constant", "warmup_steps": 0,
                     "warmup_ratio": 0.0, "max_grad_norm": 0.0, "weight_decay": 0.01},
    }, PreparedDataset(cache, {"target_adapter_version": TARGET_ADAPTER_VERSION}), output)
    assert len(model.seen) == 2
    assert "SAMPLE_1" in model.seen[0]
    assert "SAMPLE_0" in model.seen[1]
    assert not torch.equal(weights_before, model.head.weight.detach())
    assert result["prompt_exposures"] == 2
    assert model.dropout.p == 0.0
    assert model.attention_dropout == 0.0
    assert result["dropout_settings_disabled"] == 2
    assert result["matched_schedule_dropout_disabled"] is True
    assert (output / "model" / "model.safetensors").is_file()
    assert result["final_model_path"] == str(output / "model")
    assert result["resolved_optimizer"] == {
        "optim": "adamw_torch", "learning_rate": 0.01, "lr_scheduler_type": "constant",
        "warmup_steps": 0, "warmup_ratio": 0.0, "max_grad_norm": 0.0, "weight_decay": 0.01,
    }


def test_predict_explicit_dtype_and_device_override_environment(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import ts_grounder.vlm_training as module
    record = make_record(tmp_path)
    dataset = tmp_path / "test.jsonl"
    dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")
    captured = {}

    class Model:
        def to(self, target):
            captured["device"] = str(target)
            return self

    def load_model(path, *, model_kwargs):
        captured["load_kwargs"] = model_kwargs
        return object(), Model()

    monkeypatch.setattr(module, "_require_training_deps", lambda: None)
    monkeypatch.setattr(module, "load_vlm_checkpoint", load_model)
    monkeypatch.setattr(module, "_generate_predictions_for_records", lambda **kw: {"num_examples": len(kw["records"])})
    monkeypatch.setenv("TS_GROUNDER_PREDICT_TORCH_DTYPE", "float16")
    monkeypatch.setenv("TS_GROUNDER_PREDICT_DEVICE_MAP", "auto")
    result = module.generate_predictions(model_path="unused", dataset_jsonl=dataset,
        output_path=tmp_path / "out.jsonl", torch_dtype="bfloat16", device="cpu")
    assert captured["load_kwargs"] == {"torch_dtype": torch.bfloat16}
    assert captured["device"] == "cpu"
    assert result["inference"]["torch_dtype"] == "torch.bfloat16"
    assert result["inference"]["device"] == "cpu"


def test_predict_cli_forwards_explicit_dtype_device(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("tango_predict_cli_test", ROOT / "predict.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured = {}

    def generate(**kwargs):
        captured.update(kwargs)
        return {"num_examples": 0}

    monkeypatch.setattr(module, "generate_predictions", generate)
    monkeypatch.setattr(sys, "argv", ["predict.py", "--dataset-jsonl", "test.jsonl",
        "--model-path", "stage2", "--output", str(tmp_path / "predictions.jsonl"),
        "--torch-dtype", "bfloat16", "--device", "cpu"])
    module.main()
    assert captured["torch_dtype"] == "bfloat16"
    assert captured["device"] == "cpu"
    assert captured["max_new_tokens"] == 2048


def test_text_generation_suppresses_only_configured_visual_structure():
    model = SimpleNamespace(config=SimpleNamespace(
        image_token_id=12, video_token_id=13, vision_start_token_id=10,
        vision_end_token_id=11, text_config=SimpleNamespace(image_token_id=12),
    ))
    assert forbidden_action_token_ids(model) == [10, 11, 12, 13]
    assert forbidden_action_token_ids(SimpleNamespace()) == []
