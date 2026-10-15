from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ts_grounder.rl_reward import compute_grounder_reward_details  # noqa: E402


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(payload)
    return records


def _write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _maybe_parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "{[":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _field(record: dict[str, Any], names: tuple[str, ...], *, required: bool = True) -> Any:
    for name in names:
        if name in record:
            return record[name]
    if required:
        joined = ", ".join(names)
        raise KeyError(f"record is missing one of: {joined}")
    return None


def _infer_seq_len(record: dict[str, Any], ground_truth: Any) -> int:
    value = _field(record, ("seq_len", "series_length"), required=False)
    if value is None:
        metadata = record.get("metadata", {})
        if isinstance(metadata, dict):
            value = metadata.get("series_length")
    if value is not None:
        return int(value)
    if isinstance(ground_truth, list):
        return len(ground_truth)
    if isinstance(ground_truth, dict):
        for key in ("point_labels", "labels"):
            labels = ground_truth.get(key)
            if isinstance(labels, list):
                return len(labels)
        intervals = ground_truth.get("evidence") or ground_truth.get("events") or []
        if isinstance(intervals, list) and intervals:
            max_end = max(int(item.get("end", 0)) for item in intervals if isinstance(item, dict))
            return max_end + 1
    raise KeyError("record is missing seq_len and it could not be inferred")


def _summarize(scored_records: list[dict[str, Any]]) -> dict[str, float]:
    if not scored_records:
        return {
            "num_samples": 0.0,
            "mean_reward": 0.0,
            "json_validity_rate": 0.0,
            "event_f1": 0.0,
            "boundary_iou": 0.0,
        }

    def mean_detail(key: str) -> float:
        values = [float(item["reward_details"].get(key, 0.0)) for item in scored_records]
        return float(sum(values) / len(values))

    return {
        "num_samples": float(len(scored_records)),
        "mean_reward": mean_detail("reward"),
        "json_validity_rate": mean_detail("json_valid"),
        "event_f1": mean_detail("event_f1"),
        "boundary_iou": mean_detail("boundary_iou"),
    }


def score_file(predictions_path: str | Path, output_path: str | Path) -> dict[str, float]:
    records = _load_jsonl(predictions_path)
    scored: list[dict[str, Any]] = []
    for record in records:
        model_output = _field(record, ("model_output", "generated_text", "output", "prediction"))
        ground_truth = _field(record, ("ground_truth", "target", "labels", "point_labels"))
        ground_truth = _maybe_parse_json(ground_truth)
        seq_len = _infer_seq_len(record, ground_truth)
        details = compute_grounder_reward_details(
            pred_output=model_output,
            gt_json_or_labels=ground_truth,
            seq_len=seq_len,
        )
        scored_record = dict(record)
        scored_record["reward"] = details["reward"]
        scored_record["reward_details"] = details
        scored.append(scored_record)

    _write_jsonl(scored, output_path)
    summary = _summarize(scored)
    summary_path = Path(output_path).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Score grounder model outputs with RL reward components.")
    parser.add_argument("--predictions", type=str, required=True, help="Input predictions JSONL.")
    parser.add_argument("--output", type=str, default="scored_predictions.jsonl", help="Output scored JSONL.")
    args = parser.parse_args()

    summary = score_file(args.predictions, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
