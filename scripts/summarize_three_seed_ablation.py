#!/usr/bin/env python3
"""Summarize repeated Qwen-VL ablation runs as mean +/- std tables."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Iterable


DEFAULT_MODELS = ["qwen25vl_7b", "qwen3_vl_2b", "qwen3_vl_4b", "qwen3_vl_8b"]
DEFAULT_SETTINGS = ["sft_only", "no_residual", "full_residual"]
DEFAULT_SUBSETS = ["Daphnet", "MSL", "NEK", "Power", "SED", "TAO", "TODS", "YAHOO"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs", help="Output root containing seed batch dirs.")
    parser.add_argument(
        "--prefix",
        default="qwen_vl_all_models_sft_residual_ablation_8subsets_qwen_vl_three_setting_3seed",
        help="Batch directory prefix before _seed*. The script globs '<prefix>_seed*'.",
    )
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--settings", nargs="*", default=DEFAULT_SETTINGS)
    parser.add_argument("--subsets", nargs="*", default=DEFAULT_SUBSETS)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-tsv", default=None)
    return parser.parse_args()


def metric_value(metrics: dict, *keys: str) -> float | None:
    for key in keys:
        if key in metrics and metrics[key] is not None:
            return float(metrics[key])
    return None


def read_metrics(path: Path) -> tuple[float, float, float] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("metrics", data)
    precision = metric_value(metrics, "point_precision", "precision", "event_precision")
    recall = metric_value(metrics, "point_recall", "recall", "event_recall")
    f1 = metric_value(metrics, "point_f1", "f1", "event_f1")
    if precision is None or recall is None or f1 is None:
        return None
    return precision, recall, f1


def mean_std(values: Iterable[float]) -> tuple[float | None, float | None, int]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None, 0
    if len(vals) == 1:
        return vals[0], 0.0, 1
    return statistics.mean(vals), statistics.stdev(vals), len(vals)


def fmt(mean: float | None, std: float | None, n: int) -> str:
    if mean is None or std is None:
        return "MISSING"
    return f"{mean:.4f}+/-{std:.4f} (n={n})"


def seed_name(batch_dir: Path) -> str:
    match = re.search(r"_seed([^_/\\]+)_", batch_dir.name)
    return match.group(1) if match else batch_dir.name


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    batch_dirs = sorted(p for p in root.glob(f"{args.prefix}_seed*") if p.is_dir())
    if not batch_dirs:
        raise SystemExit(f"No seed batch dirs matched: {root / (args.prefix + '_seed*')}")

    records = []
    missing = []
    per_seed = {}

    for batch_dir in batch_dirs:
        seed = seed_name(batch_dir)
        for model in args.models:
            for setting in args.settings:
                subset_values = []
                for subset in args.subsets:
                    metrics_path = batch_dir / model / setting / subset / "rl" / "eval" / "test_metrics.json"
                    metrics = read_metrics(metrics_path)
                    if metrics is None:
                        missing.append(str(metrics_path))
                        continue
                    precision, recall, f1 = metrics
                    per_seed[(seed, model, setting, subset)] = metrics
                    subset_values.append(metrics)
                if subset_values:
                    avg_precision = statistics.mean(v[0] for v in subset_values)
                    avg_recall = statistics.mean(v[1] for v in subset_values)
                    avg_f1 = statistics.mean(v[2] for v in subset_values)
                    per_seed[(seed, model, setting, "AVG")] = (avg_precision, avg_recall, avg_f1)

    all_subsets = list(args.subsets) + ["AVG"]
    for model in args.models:
        for setting in args.settings:
            for subset in all_subsets:
                values = [per_seed.get((seed_name(b), model, setting, subset)) for b in batch_dirs]
                p_mean, p_std, p_n = mean_std(v[0] for v in values if v is not None)
                r_mean, r_std, r_n = mean_std(v[1] for v in values if v is not None)
                f_mean, f_std, f_n = mean_std(v[2] for v in values if v is not None)
                records.append(
                    {
                        "model": model,
                        "setting": setting,
                        "subset": subset,
                        "precision_mean": p_mean,
                        "precision_std": p_std,
                        "precision_n": p_n,
                        "recall_mean": r_mean,
                        "recall_std": r_std,
                        "recall_n": r_n,
                        "f1_mean": f_mean,
                        "f1_std": f_std,
                        "f1_n": f_n,
                    }
                )

    print(f"seed_batches={len(batch_dirs)}")
    for batch_dir in batch_dirs:
        print(f"  {batch_dir}")
    print()
    print("model\tsetting\tsubset\tprecision\trecall\tf1")
    for row in records:
        print(
            "\t".join(
                [
                    row["model"],
                    row["setting"],
                    row["subset"],
                    fmt(row["precision_mean"], row["precision_std"], row["precision_n"]),
                    fmt(row["recall_mean"], row["recall_std"], row["recall_n"]),
                    fmt(row["f1_mean"], row["f1_std"], row["f1_n"]),
                ]
            )
        )

    if missing:
        print()
        print(f"missing_metrics={len(missing)}")
        for path in missing[:50]:
            print(path)
        if len(missing) > 50:
            print(f"... {len(missing) - 50} more")

    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps({"records": records, "missing_metrics": missing}, indent=2),
            encoding="utf-8",
        )
    if args.output_tsv:
        lines = ["model\tsetting\tsubset\tprecision_mean\tprecision_std\trecall_mean\trecall_std\tf1_mean\tf1_std"]
        for row in records:
            lines.append(
                "\t".join(
                    [
                        row["model"],
                        row["setting"],
                        row["subset"],
                        "" if row["precision_mean"] is None else f"{row['precision_mean']:.6f}",
                        "" if row["precision_std"] is None else f"{row['precision_std']:.6f}",
                        "" if row["recall_mean"] is None else f"{row['recall_mean']:.6f}",
                        "" if row["recall_std"] is None else f"{row['recall_std']:.6f}",
                        "" if row["f1_mean"] is None else f"{row['f1_mean']:.6f}",
                        "" if row["f1_std"] is None else f"{row['f1_std']:.6f}",
                    ]
                )
            )
        Path(args.output_tsv).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
