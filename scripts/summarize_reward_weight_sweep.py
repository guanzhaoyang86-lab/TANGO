from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


DEFAULT_PREFIX = (
    "qwen_vl_all_models_sft_residual_ablation_8subsets_"
    "qwen3_vl_8b_reward_weight_sweep_8subsets_3seed"
)
DEFAULT_MODEL = "qwen3_vl_8b"
DEFAULT_SETTING = "residual_v2"
DEFAULT_SUBSETS = ("Daphnet", "MSL", "NEK", "Power", "SED", "TAO", "TODS", "YAHOO")


def _metric(metrics: dict, *keys: str) -> float | None:
    for key in keys:
        if key in metrics:
            return float(metrics[key])
    return None


def _fmt(values: list[float]) -> str:
    if not values:
        return "MISSING"
    sigma = stdev(values) if len(values) > 1 else 0.0
    return f"{mean(values):.4f}+/-{sigma:.4f} (n={len(values)})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize reward-weight sweep metrics.")
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--setting", default=DEFAULT_SETTING)
    parser.add_argument("--subsets", nargs="*", default=list(DEFAULT_SUBSETS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_pattern = re.compile(
        rf"^{re.escape(args.prefix)}_(?P<config>.+)_seed(?P<seed>\d+)_\d{{8}}_\d{{6}}$"
    )
    batches: list[tuple[str, str, Path]] = []
    for path in sorted(args.root.glob(args.prefix + "_*")):
        if not path.is_dir():
            continue
        match = batch_pattern.match(path.name)
        if match:
            batches.append((match.group("config"), match.group("seed"), path))

    print(f"batches={len(batches)}")
    for config, seed, path in batches:
        print(f"  config={config}\tseed={seed}\tpath={path}")

    rows: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)
    missing: list[Path] = []

    for config, _seed, batch in batches:
        for subset in args.subsets:
            metrics_path = (
                batch
                / args.model
                / args.setting
                / subset
                / "rl"
                / "eval"
                / "test_metrics.json"
            )
            if not metrics_path.exists():
                missing.append(metrics_path)
                continue
            data = json.loads(metrics_path.read_text())
            metrics = data.get("metrics", data)
            precision = _metric(metrics, "point_precision", "precision", "event_precision")
            recall = _metric(metrics, "point_recall", "recall", "event_recall")
            f1 = _metric(metrics, "point_f1", "f1", "event_f1")
            if precision is None or recall is None or f1 is None:
                missing.append(metrics_path)
                continue
            rows[(config, subset)].append((precision, recall, f1))

    configs = sorted({config for config, _seed, _path in batches})
    print("\nconfig\tsubset\tprecision\trecall\tf1")
    for config in configs:
        pooled: list[tuple[float, float, float]] = []
        for subset in args.subsets:
            values = rows[(config, subset)]
            pooled.extend(values)
            print(
                config,
                subset,
                _fmt([value[0] for value in values]),
                _fmt([value[1] for value in values]),
                _fmt([value[2] for value in values]),
                sep="\t",
            )
        print(
            config,
            "AVG",
            _fmt([value[0] for value in pooled]),
            _fmt([value[1] for value in pooled]),
            _fmt([value[2] for value in pooled]),
            sep="\t",
        )

    print(f"\nmissing_metrics={len(missing)}")
    for path in missing[:50]:
        print(path)
    if len(missing) > 50:
        print(f"... {len(missing) - 50} more")


if __name__ == "__main__":
    main()
