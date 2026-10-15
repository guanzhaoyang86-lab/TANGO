"""Optional offline end-to-end launch smoke; NOT a benchmark or trained model.

Run explicitly, outside the default unit-test collection:
  python tests/run_tiny_cli_integration.py --work-dir /new/empty/path

Uses a tiny locally initialized real Qwen2.5-VL and newly generated fixtures.
The EOS-biased checkpoint intentionally produces empty outputs to keep all
CLI phases bounded. Detection accuracy/convergence are not being tested.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from test_tiny_vlm_integration import _tiny_qwen_components


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    work = args.work_dir.resolve()
    if work.exists() and any(work.iterdir()):
        raise ValueError("Use a new empty integration work directory")
    work.mkdir(parents=True, exist_ok=True)
    import torch
    import yaml
    from PIL import Image, ImageDraw
    from ts_grounder.utils import load_yaml

    torch.manual_seed(2026)
    torch.set_num_threads(1)
    processor, model = _tiny_qwen_components()
    with torch.no_grad():
        input_weights = model.get_input_embeddings().weight
        output_weights = model.get_output_embeddings().weight
        assert input_weights.data_ptr() != output_weights.data_ptr()
        eos_direction = input_weights[processor.tokenizer.convert_tokens_to_ids("assistant")].clone()
        output_weights.zero_()
        output_weights[processor.tokenizer.eos_token_id].copy_(eos_direction * 100)
    checkpoint = work / "stage2_fixture"
    model.save_pretrained(checkpoint)
    processor.save_pretrained(checkpoint)

    source = work / "fresh_generated_localization_fixture"
    source.mkdir()
    for split, count in (("train", 8), ("val", 2), ("test", 2)):
        samples = []
        for i in range(count):
            anomalous = i % 2 == 1
            series = [0.05 * ((j * (i + 1)) % 17) + 0.01 * i for j in range(256)]
            if anomalous:
                series[10:13] = [4.0 + i] * 3
            name = f"{split}-{i}"
            image = Image.new("RGB", (56, 56), "white")
            draw = ImageDraw.Draw(image)
            maximum = max(series) + 0.1
            draw.line([(2 + 51 * j / 255, 53 - 50 * v / maximum) for j, v in enumerate(series)], fill="blue")
            image.save(source / f"{name}.png")
            samples.append({"sample_id": name, "annotation_scope": "localization_only", "semantic_label_available": False,
                "series": series, "point_labels": [int(anomalous and 10 <= j <= 12) for j in range(256)],
                "events": [{"start": 10, "end": 12}] if anomalous else [], "image_path": f"{name}.png",
                "context": {"series_length": 256, "source_dataset": "NEWLY_GENERATED_INTEGRATION_FIXTURE",
                            "materialized_subset_split": split}})
        (source / f"{split}.json").write_text(json.dumps(samples), encoding="utf-8")
    control = load_yaml(ROOT / "configs/controls.yaml")
    control.update(epochs=1, prompt_fraction=0.5, optimizer="adamw", num_generations=2,
                   torch_dtype="float32", gradient_checkpointing=False)
    config = work / "controls.yaml"
    config.write_text(yaml.safe_dump(control), encoding="utf-8")
    experiment = work / "experiment"
    command = [sys.executable, str(ROOT / "scripts/run_ablation.py"), "--action", "all", "--variant", "all",
               "--data-root", str(source), "--stage2-checkpoint", str(checkpoint), "--experiment-dir", str(experiment),
               "--config", str(config), "--source-terms-reviewed"]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    started = time.perf_counter()
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    for variant in ("A", "D", "F", "G"):
        prediction = experiment / "evaluation" / variant / "test_predictions.jsonl"
        assert len(prediction.read_text(encoding="utf-8").splitlines()) == 2
        assert prediction.with_suffix(".metrics.json").is_file()
        if variant != "A":
            assert (experiment / f"{variant}_completed.json").is_file()
            assert (experiment / variant / "model/config.json").is_file()
    assert not (experiment / "A/model").exists()
    receipt = {"status": "passed", "utc": datetime.now(timezone.utc).isoformat(),
               "wall_seconds": time.perf_counter() - started, "variants": ["A", "D", "F", "G"],
               "framework": "actual offline randomly initialized tiny Qwen2.5-VL, CPU",
               "fixture": "newly generated localization records; no released benchmark data",
               "research_accuracy_or_gpu_claim": False,
               "generation_fixture": "EOS-biased to bound the launch smoke; no accuracy assertion"}
    (work / "integration_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
