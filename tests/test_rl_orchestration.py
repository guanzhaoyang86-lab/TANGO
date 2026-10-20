"""Launcher wiring test: no neural training or performance result is fabricated.

The subprocess boundary is replaced by artifact-producing test stubs. The real
preparation, hashing, schedule creation, backend config parser and preflight run.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ts_grounder.control_protocol import file_sha256, fingerprint_checkpoint, load_prompt_schedule
from ts_grounder.rl_train_grpo import _config_defaults_from_file, _training_plan, build_parser as rl_parser
from ts_grounder.vlm_eval import load_jsonl
from ts_grounder.schema import GROUNDING_CONTRACT_VERSION


def test_all_control_launcher_wiring_and_matched_backend_settings(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("rl_orchestration_under_test", ROOT / "scripts/run_ablation.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    source = tmp_path / "generated_unit_data"
    source.mkdir()
    Image.new("RGB", (8, 8), "white").save(source / "image.png")
    for split, count in (("train", 4), ("val", 1), ("test", 1)):
        rows = [{"sample_id": f"{split}_{i}", "series": [float(k) for k in range(256)],
                 "point_labels": [int(2 <= k <= 4) for k in range(256)],
                 "events": [{"start": 2, "end": 4}], "image_path": "image.png",
                 "annotation_scope": "localization_only", "semantic_label_available": False,
                 "context": {"series_length": 256, "source_dataset": "unit_test", "interval_convention": "inclusive"}}
                for i in range(count)]
        (source / f"{split}.json").write_text(json.dumps(rows), encoding="utf-8")
    stage2 = tmp_path / "stage2_stub"
    stage2.mkdir()
    (stage2 / "config.json").write_text("{}", encoding="utf-8")
    (stage2 / "model.safetensors").write_bytes(b"not-model-weights: unit-test identity only")
    run_root = tmp_path / "experiment"
    observed = {}

    def fake_execute(command, root, phase, **kwargs):
        observed[phase] = command
        if phase == "train_only_residual_mining":
            source_file = Path(command[command.index("--real-train-data") + 1])
            output = Path(command[command.index("--output") + 1])
            rows = load_jsonl(source_file)
            classes = ["false_negative", "false_negative", "boundary_error", "correct_abnormal"]
            residual = [{**row, "error_type": kind, "ground_truth": row["target"], "residual_mining": {
                "source_split": "train", "input_sha256": file_sha256(source_file),
                "grounding_contract_version": GROUNDING_CONTRACT_VERSION,
                "checkpoint_fingerprint": fingerprint_checkpoint(stage2)}} for row, kind in zip(rows, classes)]
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("\n".join(json.dumps(row) for row in residual), encoding="utf-8")
            output.with_suffix(".summary.json").write_text(json.dumps({
                "input_sha256": file_sha256(source_file), "residual_sha256": file_sha256(output),
                "grounding_contract_version": GROUNDING_CONTRACT_VERSION,
                "checkpoint_fingerprint": fingerprint_checkpoint(stage2)}), encoding="utf-8")
        elif phase.endswith("_training"):
            config_path = Path(command[command.index("--config") + 1])
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            variant = phase[0]
            if variant == "D":
                assert command[command.index("--prepared-dataset-root") + 1] == str(run_root / "shared/dataset_cache")
                output = Path(cfg["output_dir"])
            else:
                parser = rl_parser()
                parser.set_defaults(**_config_defaults_from_file(config_path, parser))
                parsed = parser.parse_args([])
                _, records, epochs = _training_plan(parsed)
                assert len(epochs) == 2 and sum(map(len, epochs)) == 4
                assert parsed.weight_decay == 0.0 and parsed.top_p == 1
                output = Path(parsed.output_dir)
            target = output / "model"
            target.mkdir(parents=True)
            for name in ("config.json", "model.safetensors"):
                (target / name).write_bytes((stage2 / name).read_bytes())
        elif phase.endswith("_evaluation"):
            variant = phase[0]
            actual_model = Path(command[command.index("--model-path") + 1])
            assert actual_model == (stage2 if variant == "A" else run_root / variant / "model")
            output = Path(command[command.index("--output") + 1])
            output.parent.mkdir(parents=True)
            output.write_text("", encoding="utf-8")
        else:
            raise AssertionError(phase)

    monkeypatch.setattr(launcher, "execute", fake_execute)
    args = launcher.build_parser().parse_args(["--action", "all", "--data-root", str(source),
        "--stage2-checkpoint", str(stage2), "--experiment-dir", str(run_root), "--source-terms-reviewed"])
    launcher.run(args)
    assert len(observed) == 8  # mining + D/F/G training + A/D/F/G evaluation
    f = json.loads((run_root / "configs/F.json").read_text())["rl"]
    g = json.loads((run_root / "configs/G.json").read_text())["rl"]
    ignored = {"output_dir", "prompt_schedule_file"}
    assert {k: v for k, v in f.items() if k not in ignored} == {k: v for k, v in g.items() if k not in ignored}
    d = json.loads((run_root / "configs/D.json").read_text())
    assert d["training"]["prompt_schedule_file"] == g["prompt_schedule_file"]
    assert d["training"]["learning_rate"] == g["learning_rate"]
    assert d["training"].get("weight_decay", 0) == g.get("weight_decay", 0)
    assert d["training"]["max_steps"] == 4
    assert d["training"]["lr_scheduler_type"] == "constant"
    assert d["training"]["max_grad_norm"] == 0
    assert not (run_root / "A_completed.json").exists()  # A must never be trained.
    assert json.loads((run_root / "experiment.json").read_text())["grounding_contract_version"] == GROUNDING_CONTRACT_VERSION
    assert json.loads((run_root / "G_completed.json").read_text())["grounding_contract_version"] == GROUNDING_CONTRACT_VERSION
    # An old mining summary must not be silently reused under the corrected parser.
    summary_path = run_root / "mining/residual_pool.summary.json"
    old_summary = json.loads(summary_path.read_text())
    old_summary.pop("grounding_contract_version")
    summary_path.write_text(json.dumps(old_summary), encoding="utf-8")
    args.action = "schedule"
    import pytest
    with pytest.raises(ValueError, match="old output contract"):
        launcher.run(args)
