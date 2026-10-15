from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from ts_grounder.utils import dump_json, load_yaml
from ts_grounder.vlm_training import generate_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VLM inference for TS Grounder")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional resolved training config recorded for provenance.",
    )
    parser.add_argument("--dataset-jsonl", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output", type=str, default="predictions.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--torch-dtype", choices=("bfloat16", "float16", "float32", "auto"), default=None,
                        help="Explicit model dtype; overrides TS_GROUNDER_PREDICT_TORCH_DTYPE.")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto",
                        help="Inference device; auto selects CUDA when available, otherwise CPU.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    if args.config:
        config_path = Path(args.config)
        _ = load_yaml(repo_root / config_path if not config_path.is_absolute() else config_path)
    metrics = generate_predictions(
        model_path=args.model_path,
        dataset_jsonl=args.dataset_jsonl,
        output_path=args.output,
        max_new_tokens=args.max_new_tokens,
        torch_dtype=args.torch_dtype,
        device=args.device,
    )
    dump_json(metrics, Path(args.output).with_suffix(".metrics.json"))


if __name__ == "__main__":
    main()
