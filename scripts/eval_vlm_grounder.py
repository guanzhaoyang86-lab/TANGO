from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ts_grounder.utils import dump_json
from ts_grounder.vlm_eval import evaluate_prediction_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VLM grounding predictions.")
    parser.add_argument("--dataset-jsonl", type=str, required=True)
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--output", type=str, default="metrics.json")
    args = parser.parse_args()

    metrics = evaluate_prediction_files(
        dataset_jsonl=args.dataset_jsonl,
        predictions_jsonl=args.predictions,
        output_path=args.output,
    )
    dump_json(metrics, args.output)
    print(metrics["metrics"])


if __name__ == "__main__":
    main()
