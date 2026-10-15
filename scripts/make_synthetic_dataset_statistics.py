#!/usr/bin/env python3
"""Audit the synthetic dataset and write the appendix statistics table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs" / "dataset_appendix"
SPLITS = ("train", "test")
ALL_SOURCE_SPLITS = ("train", "val", "test")
PAPER_ORDER = ("point", "freq", "trend", "range", "normal")
PAPER_NAMES = {
    "point": "Point",
    "freq": "Frequency",
    "trend": "Trend",
    "range": "Range",
    "normal": "Normal",
}


def _load_training_type_rule():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from ts_grounder.vlm_data import _sample_key

    return _sample_key


def _resolve_data_root(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    env_root = os.environ.get("SYNTHETIC_SOURCE_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend(
        [
            REPO_ROOT / "dataset" / "anomaly_db_v1",
        ]
    )
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if all((resolved / f"{split}.json").is_file() for split in ALL_SOURCE_SPLITS):
            return resolved
    rendered = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Could not locate synthetic train/val/test JSON files. Checked:\n"
        f"{rendered}"
    )


def _resolve_generator_script() -> Path | None:
    candidates = [
        REPO_ROOT.parent
        / "PatternFinder-main"
        / "dev"
        / "scripts"
        / "generate_balanced_dataset.py",
        REPO_ROOT / "dev" / "scripts" / "generate_balanced_dataset.py",
    ]
    return next((path.resolve() for path in candidates if path.is_file()), None)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_recorded_checksums(data_root: Path) -> None:
    checksum_path = data_root / "checksums.sha256"
    if not checksum_path.is_file():
        raise FileNotFoundError(f"Missing checksum metadata: {checksum_path}")
    expected: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        expected[relative.strip()] = digest
    for name in ("manifest.json", "train.json", "val.json", "test.json"):
        if name not in expected:
            raise AssertionError(f"{checksum_path} has no checksum for {name}")
        actual = _sha256(data_root / name)
        if actual != expected[name]:
            raise AssertionError(f"Checksum mismatch for {data_root / name}")


def _load_manifest(data_root: Path) -> dict[str, Any]:
    path = data_root / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def _count_raw_split(
    path: Path,
    *,
    split: str,
    sample_key,
) -> tuple[Counter[str], dict[str, int], set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"Expected a JSON list in {path}")

    counts: Counter[str] = Counter()
    multi_event_samples = 0
    mixed_type_multi_samples = 0
    parameter_mismatches = 0
    unknown_types: set[str] = set()
    seen_ids: set[str] = set()

    for sample in payload:
        if not isinstance(sample, dict):
            raise TypeError(f"{path} contains a non-object record")
        sample_id = str(sample.get("sample_id", ""))
        if not sample_id:
            raise AssertionError(f"{path} contains a record without sample_id")
        if sample_id in seen_ids:
            raise AssertionError(f"Duplicate sample_id in {path}: {sample_id}")
        seen_ids.add(sample_id)
        if not sample_id.startswith(f"{split}_"):
            raise AssertionError(
                f"{path}: sample_id {sample_id!r} disagrees with file split {split!r}"
            )

        canonical_type = str(sample_key(sample))
        if canonical_type not in PAPER_NAMES:
            unknown_types.add(canonical_type)
            continue
        counts[canonical_type] += 1

        events = sample.get("events", [])
        if not isinstance(events, list):
            raise TypeError(f"{sample_id}: events is not a list")
        if len(events) > 1:
            multi_event_samples += 1
            event_types = {str(event.get("type", "")) for event in events}
            if len(event_types) > 1:
                mixed_type_multi_samples += 1

        parameter_type = str(
            sample.get("parameters", {}).get("anomaly_type", "")
        ).strip()
        if parameter_type:
            normalized_parameter = {
                "frequency": "freq",
                "level": "range",
            }.get(parameter_type.lower(), parameter_type.lower())
            if normalized_parameter != canonical_type:
                parameter_mismatches += 1

    if unknown_types:
        raise AssertionError(f"{path} contains unknown canonical types: {unknown_types}")
    if sum(counts.values()) != len(payload):
        raise AssertionError(f"Type counts do not sum to record count in {path}")
    return counts, {
        "num_records": len(payload),
        "multi_event_samples": multi_event_samples,
        "mixed_type_multi_samples": mixed_type_multi_samples,
        "parameter_type_mismatches": parameter_mismatches,
    }, seen_ids


def _find_latest_prepared_root(data_root: Path) -> Path | None:
    output_root = REPO_ROOT / "outputs"
    if not output_root.exists():
        return None
    candidates: list[tuple[float, Path]] = []
    for manifest_path in output_root.glob("**/dataset_cache/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            source_root = Path(str(manifest.get("source_root", ""))).expanduser().resolve()
        except Exception:
            continue
        prepared_root = manifest_path.parent
        if (
            source_root == data_root
            and all((prepared_root / f"{split}.jsonl").is_file() for split in SPLITS)
        ):
            candidates.append((manifest_path.stat().st_mtime, prepared_root))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _validate_prepared_jsonl(
    prepared_root: Path,
    raw_counts: dict[str, Counter[str]],
    raw_ids: dict[str, set[str]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in SPLITS:
        path = prepared_root / f"{split}.jsonl"
        counts: Counter[str] = Counter()
        seen_ids: set[str] = set()
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                if str(record.get("split", "")) != split:
                    raise AssertionError(
                        f"{path}:{line_number}: record split does not equal {split}"
                    )
                record_id = str(record.get("id", ""))
                if record_id in seen_ids:
                    raise AssertionError(f"Duplicate prepared id in {path}: {record_id}")
                seen_ids.add(record_id)
                anomaly_type = str(
                    record.get("metadata", {}).get("anomaly_type", "")
                )
                if anomaly_type not in PAPER_NAMES:
                    raise AssertionError(
                        f"{path}:{line_number}: unknown metadata.anomaly_type "
                        f"{anomaly_type!r}"
                    )
                counts[anomaly_type] += 1
        if counts != raw_counts[split]:
            raise AssertionError(
                f"Prepared {split} type counts differ from raw source: "
                f"{dict(counts)} != {dict(raw_counts[split])}"
            )
        if seen_ids != raw_ids[split]:
            raise AssertionError(
                f"Prepared {split} IDs differ from the raw source IDs"
            )
        result[split] = {
            "num_records": len(seen_ids),
            "type_counts": dict(counts),
            "split_field": split,
            "type_field": "metadata.anomaly_type",
        }
    return result


def _validate_manifest_counts(
    manifest: dict[str, Any],
    raw_counts: dict[str, Counter[str]],
    raw_info: dict[str, dict[str, int]],
) -> None:
    manifest_splits = manifest.get("splits", {})
    for split in ALL_SOURCE_SPLITS:
        split_manifest = manifest_splits.get(split, {})
        if int(split_manifest.get("num_samples", -1)) != raw_info[split]["num_records"]:
            raise AssertionError(f"Manifest num_samples differs for {split}")
        manifest_counts = Counter(
            {
                str(key): int(value)
                for key, value in split_manifest.get(
                    "anomaly_type_counts", {}
                ).items()
            }
        )
        if manifest_counts != raw_counts[split]:
            raise AssertionError(f"Manifest anomaly_type_counts differs for {split}")


def _percentage(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def _build_rows(
    raw_counts: dict[str, Counter[str]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    train_total = sum(raw_counts["train"].values())
    test_total = sum(raw_counts["test"].values())
    overall_total = train_total + test_total
    included_types = [
        anomaly_type
        for anomaly_type in PAPER_ORDER
        if raw_counts["train"][anomaly_type] + raw_counts["test"][anomaly_type] > 0
    ]
    rows: list[dict[str, Any]] = []
    for anomaly_type in included_types:
        train_count = raw_counts["train"][anomaly_type]
        test_count = raw_counts["test"][anomaly_type]
        total_count = train_count + test_count
        rows.append(
            {
                "Anomaly Type": PAPER_NAMES[anomaly_type],
                "Train Samples": train_count,
                "Train Proportion (%)": _percentage(train_count, train_total),
                "Test Samples": test_count,
                "Test Proportion (%)": _percentage(test_count, test_total),
                "Total Samples": total_count,
                "Overall Proportion (%)": _percentage(total_count, overall_total),
            }
        )

    if sum(row["Train Samples"] for row in rows) != train_total:
        raise AssertionError("Train type counts do not sum to the train total")
    if sum(row["Test Samples"] for row in rows) != test_total:
        raise AssertionError("Test type counts do not sum to the test total")
    if sum(row["Total Samples"] for row in rows) != overall_total:
        raise AssertionError("Overall type counts do not sum to the overall total")
    for key in (
        "Train Proportion (%)",
        "Test Proportion (%)",
        "Overall Proportion (%)",
    ):
        if abs(sum(float(row[key]) for row in rows) - 100.0) > 0.01:
            raise AssertionError(f"{key} values do not sum to approximately 100%")
    return rows, {
        "train": train_total,
        "test": test_total,
        "overall": overall_total,
    }


def _write_csv(
    rows: list[dict[str, Any]],
    totals: dict[str, int],
    output_path: Path,
) -> None:
    fieldnames = [
        "Anomaly Type",
        "Train Samples",
        "Train Proportion (%)",
        "Test Samples",
        "Test Proportion (%)",
        "Total Samples",
        "Overall Proportion (%)",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "Train Proportion (%)": f"{row['Train Proportion (%)']:.2f}",
                    "Test Proportion (%)": f"{row['Test Proportion (%)']:.2f}",
                    "Overall Proportion (%)": f"{row['Overall Proportion (%)']:.2f}",
                }
            )
        writer.writerow(
            {
                "Anomaly Type": "Total",
                "Train Samples": totals["train"],
                "Train Proportion (%)": "100.00",
                "Test Samples": totals["test"],
                "Test Proportion (%)": "100.00",
                "Total Samples": totals["overall"],
                "Overall Proportion (%)": "100.00",
            }
        )


def _write_latex(
    rows: list[dict[str, Any]],
    totals: dict[str, int],
    output_path: Path,
) -> None:
    body = "\n".join(
        (
            f"{row['Anomaly Type']} & {row['Train Samples']:,} "
            f"& {row['Train Proportion (%)']:.2f} "
            f"& {row['Test Samples']:,} & {row['Test Proportion (%)']:.2f} "
            f"& {row['Total Samples']:,} & {row['Overall Proportion (%)']:.2f} \\\\"
        )
        for row in rows
    )
    content = rf"""\begin{{table}}[t]
\centering
\caption{{Statistics of the synthetic anomaly grounding dataset.}}
\label{{tab:synthetic_dataset_statistics}}
\setlength{{\tabcolsep}}{{2.2pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}lrrrrrr@{{}}}}
\toprule
Type & Train & Train (\%) & Test & Test (\%) & Total & Overall (\%) \\
\midrule
{body}
\midrule
Total & {totals['train']:,} & 100.00
& {totals['test']:,} & 100.00
& {totals['overall']:,} & 100.00 \\
\bottomrule
\end{{tabular}}%
}}
\end{{table}}
"""
    output_path.write_text(content, encoding="utf-8")


def _write_text(output_path: Path, *, has_validation: bool) -> None:
    validation_sentence = (
        "A separately stored validation split is not included in these train/test totals."
        if has_validation
        else ""
    )
    content = (
        "The synthetic anomaly grounding dataset contains point, frequency, "
        "trend, and range anomalies. Table~\\ref{tab:synthetic_dataset_statistics} "
        "reports the sample counts and class proportions computed directly from "
        "the training and test records. "
        f"{validation_sentence}\n"
    )
    output_path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate audited synthetic dataset statistics for the appendix."
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--prepared-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_root = _resolve_data_root(args.data_root)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_key = _load_training_type_rule()

    _validate_recorded_checksums(data_root)
    manifest = _load_manifest(data_root)
    raw_counts: dict[str, Counter[str]] = {}
    raw_info: dict[str, dict[str, int]] = {}
    raw_ids: dict[str, set[str]] = {}
    for split in ALL_SOURCE_SPLITS:
        counts, info, ids = _count_raw_split(
            data_root / f"{split}.json",
            split=split,
            sample_key=sample_key,
        )
        raw_counts[split] = counts
        raw_info[split] = info
        raw_ids[split] = ids
    _validate_manifest_counts(manifest, raw_counts, raw_info)

    prepared_root = (
        args.prepared_root.expanduser().resolve()
        if args.prepared_root is not None
        else _find_latest_prepared_root(data_root)
    )
    prepared_check: dict[str, Any] | None = None
    if prepared_root is not None:
        prepared_check = _validate_prepared_jsonl(
            prepared_root,
            {split: raw_counts[split] for split in SPLITS},
            {split: raw_ids[split] for split in SPLITS},
        )

    rows, totals = _build_rows(raw_counts)
    anomaly_rows = [row for row in rows if row["Anomaly Type"] != "Normal"]
    balanced_train = len({row["Train Samples"] for row in anomaly_rows}) == 1
    balanced_test = len({row["Test Samples"] for row in anomaly_rows}) == 1
    csv_path = output_dir / "synthetic_dataset_statistics.csv"
    tex_path = output_dir / "synthetic_dataset_statistics.tex"
    text_path = output_dir / "synthetic_dataset_statistics_text.tex"
    _write_csv(rows, totals, csv_path)
    _write_latex(rows, totals, tex_path)
    _write_text(text_path, has_validation=raw_info["val"]["num_records"] > 0)

    print(f"synthetic_data_root={data_root}")
    print(f"manifest={data_root / 'manifest.json'}")
    print(
        "generation_script="
        f"{_resolve_generator_script() or 'not found in the current codebase'}"
    )
    print(f"split_source=independent train.json/test.json files")
    print(f"training_type_rule=events[0].type via ts_grounder.vlm_data._sample_key")
    print(f"generation_type_field=parameters.anomaly_type")
    print(f"prepared_jsonl_root={prepared_root or 'not found in the current codebase'}")
    print(f"prepared_jsonl_check={'PASS' if prepared_check is not None else 'NOT RUN'}")
    print(
        "multi_event_samples="
        f"{sum(raw_info[split]['multi_event_samples'] for split in ALL_SOURCE_SPLITS)}"
    )
    print(
        "parameter_type_mismatches="
        f"{sum(raw_info[split]['parameter_type_mismatches'] for split in ALL_SOURCE_SPLITS)}"
    )
    print(
        f"totals=train:{totals['train']} test:{totals['test']} "
        f"overall_train_plus_test:{totals['overall']} "
        f"validation_excluded:{raw_info['val']['num_records']}"
    )
    print(
        "four_anomaly_types_balanced="
        f"{'YES' if balanced_train and balanced_test else 'NO'}"
    )
    print(
        "paper_8000_train_2000_test_description_matches="
        f"{'YES' if totals['train'] == 8000 and totals['test'] == 2000 else 'NO'}"
    )
    for row in rows:
        print(
            f"{row['Anomaly Type']}: train={row['Train Samples']} "
            f"({row['Train Proportion (%)']:.2f}%), "
            f"test={row['Test Samples']} ({row['Test Proportion (%)']:.2f}%), "
            f"total={row['Total Samples']} "
            f"({row['Overall Proportion (%)']:.2f}%)"
        )
    print("PASS: raw JSON counts match manifest metadata")
    print("PASS: recorded manifest/train/val/test checksums match")
    print("PASS: train/test type counts and proportions sum to their totals")
    for path in (csv_path, tex_path, text_path):
        print(f"wrote={path}")


if __name__ == "__main__":
    main()
