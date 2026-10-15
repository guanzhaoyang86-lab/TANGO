from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .schema import EvidenceItem, GroundingResult
from .series_normalization import normalize_series_window
from .supervision import (
    TARGET_ADAPTER_VERSION, LOCALIZATION_SYSTEM_PROMPT, LOCALIZATION_USER_PROMPT,
    annotation_scope, localization_target, validate_prepared_record,
)
from .taxonomy import canonicalize_type_name
from .utils import dump_json, ensure_dir
from .vlm_prompting import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT


DEFAULT_NORMALIZATION_PROMPT_NOTE = (
    "Note: the plot and indexed values are standardized within this window; "
    "values are not raw sensor readings."
)


def resolve_source_root(candidate: str | Path | None, repo_root: Path) -> Path:
    if candidate not in (None, ""):
        path = Path(candidate).expanduser()
        return path if path.is_absolute() else (repo_root / path).resolve()

    local_dataset = repo_root / "dataset" / "anomaly_db_v1"
    if local_dataset.exists():
        return local_dataset.resolve()
    raise FileNotFoundError(
        "未找到数据集。请显式设置 data.source_root，或在当前仓库 dataset/anomaly_db_v1 下放置数据。"
    )


def _load_split_json(
    source_root: Path,
    split: str,
    *,
    split_file_suffix: str = "",
) -> list[dict[str, Any]]:
    path = source_root / f"{split}{split_file_suffix}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset split: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_source_dataset_filter(
    value: str | list[str] | tuple[str, ...] | None,
) -> set[str] | None:
    if value in (None, "", []):
        return None
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",")]
    else:
        items = [str(part).strip() for part in value]
    normalized = {item for item in items if item}
    return normalized or None


def _filter_raw_samples_by_source_dataset(
    samples: list[dict[str, Any]],
    *,
    source_dataset_filter: set[str] | None,
) -> list[dict[str, Any]]:
    if not source_dataset_filter:
        return samples
    return [
        sample
        for sample in samples
        if str(sample.get("context", {}).get("source_dataset", "")).strip() in source_dataset_filter
    ]


def _sample_key(sample: dict[str, Any]) -> str:
    events = sample.get("events", [])
    if not events:
        return "normal"
    if annotation_scope(sample) == "localization_only":
        return "anomaly_unspecified"
    return canonicalize_type_name(events[0]["type"])


def _subsample_records(
    records: list[dict[str, Any]],
    max_samples: int | None,
    *,
    balanced_by_type: bool,
    seed: int,
) -> list[dict[str, Any]]:
    if max_samples is None or len(records) <= max_samples:
        return records
    rng = random.Random(seed)
    if not balanced_by_type:
        shuffled = list(records)
        rng.shuffle(shuffled)
        return shuffled[:max_samples]

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[_sample_key(record["raw_sample"])].append(record)
    for values in buckets.values():
        rng.shuffle(values)

    ordered_keys = sorted(buckets.keys())
    picked: list[dict[str, Any]] = []
    while len(picked) < max_samples and ordered_keys:
        next_keys = []
        for key in ordered_keys:
            bucket = buckets[key]
            if bucket and len(picked) < max_samples:
                picked.append(bucket.pop())
            if bucket:
                next_keys.append(key)
        ordered_keys = next_keys
    return picked


def _event_to_evidence_item(event: dict[str, Any]) -> EvidenceItem:
    verbal_tags = event.get("verbal_tags", {})
    return EvidenceItem(
        start=int(event["start"]),
        end=int(event["end"]),
        type=canonicalize_type_name(str(event["type"])),
        strength=str(verbal_tags.get("strength", "unknown")),
        direction=str(verbal_tags.get("direction", "unknown")),
    )


def _format_interval(start: int, end: int) -> str:
    if start == end:
        return str(start)
    return f"{start}-{end}"


def _build_multi_event_summary(evidence: list[EvidenceItem]) -> str:
    if not evidence:
        return GroundingResult.no_anomaly().summary
    if len(evidence) == 1:
        item = evidence[0]
        return f"An anomaly occurs from index {item.start} to {item.end}, and this segment {item.direction}."

    preview = [_format_interval(item.start, item.end) for item in evidence[:4]]
    if len(preview) == 1:
        preview_text = preview[0]
    elif len(preview) == 2:
        preview_text = f"{preview[0]} and {preview[1]}"
    else:
        preview_text = ", ".join(preview[:-1]) + f", and {preview[-1]}"
    return (
        f"Multiple anomalies occur across {len(evidence)} intervals in this window, "
        f"including {preview_text}, and these segments become irregular."
    )


def _sample_to_result(sample: dict[str, Any]) -> GroundingResult:
    events = sample.get("events", [])
    if not events:
        return GroundingResult.no_anomaly()

    evidence = [_event_to_evidence_item(event) for event in events]
    description = sample.get("description")
    if not description and len(events) == 1:
        description = events[0].get("description")
    summary = str(description or _build_multi_event_summary(evidence))
    return GroundingResult(evidence=evidence, summary=summary)


def _resolve_image_path(
    source_root: Path,
    sample: dict[str, Any],
    split: str,
    image_subdir: str | None = None,
) -> Path:
    if image_subdir not in (None, ""):
        sample_id = sample["sample_id"]
        return (source_root / image_subdir / split / f"{sample_id}.png").resolve()
    stored = sample.get("image_path")
    if stored:
        return (source_root / stored).resolve()
    sample_id = sample["sample_id"]
    return (source_root / "images_plain_768x384" / split / f"{sample_id}.png").resolve()


def _parse_optional_float(value: Any, *, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
        return None
    return float(value)


def _coerce_series_normalization_config(value: Any) -> dict[str, Any]:
    if value in (None, "", False):
        return {"enabled": False}
    if value is True:
        cfg: dict[str, Any] = {"enabled": True}
    elif isinstance(value, str):
        normalized = value.strip().lower()
        cfg = {"enabled": normalized not in {"", "none", "raw", "identity", "false", "0"}, "method": normalized}
    else:
        cfg = dict(value)
        cfg["enabled"] = bool(cfg.get("enabled", True))

    if not cfg.get("enabled", False):
        return {"enabled": False}

    cfg["method"] = str(cfg.get("method", "robust_zscore"))
    cfg["eps"] = float(cfg.get("eps", 1e-6))
    cfg["clip"] = _parse_optional_float(cfg.get("clip", 8.0), default=8.0)
    cfg["normalize_text"] = bool(cfg.get("normalize_text", True))
    cfg["normalize_image"] = bool(cfg.get("normalize_image", True))
    cfg["fit_on_original_window"] = bool(cfg.get("fit_on_original_window", True))
    cfg["image_subdir"] = str(cfg.get("image_subdir", "images_normalized"))
    cfg["overwrite_images"] = bool(cfg.get("overwrite_images", False))
    if "prompt_note" in cfg:
        cfg["prompt_note"] = str(cfg.get("prompt_note", ""))
    elif cfg["normalize_text"] and cfg["normalize_image"]:
        cfg["prompt_note"] = DEFAULT_NORMALIZATION_PROMPT_NOTE
    elif cfg["normalize_text"]:
        cfg["prompt_note"] = (
            "Note: the indexed values are standardized within this window; "
            "values are not raw sensor readings."
        )
    else:
        cfg["prompt_note"] = ""
    return cfg


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _series_fit_length(sample: dict[str, Any], *, series_length: int, fit_on_original_window: bool) -> int:
    if not fit_on_original_window:
        return series_length
    context = sample.get("context", {})
    if not isinstance(context, dict):
        return series_length

    if context.get("window_original_length") not in (None, ""):
        return max(0, min(_safe_int(context.get("window_original_length"), series_length), series_length))
    if context.get("padding_right") not in (None, ""):
        padding_right = max(0, _safe_int(context.get("padding_right"), 0))
        return max(0, min(series_length - padding_right, series_length))
    return series_length


def _adapt_sample_series(
    sample: dict[str, Any],
    *,
    series_normalization: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if not series_normalization.get("enabled", False):
        return sample, None

    series = sample.get("series", [])
    if not series:
        return sample, None

    fit_length = _series_fit_length(
        sample,
        series_length=len(series),
        fit_on_original_window=bool(series_normalization.get("fit_on_original_window", True)),
    )
    normalized_series, normalization_info = normalize_series_window(
        series,
        fit_length=fit_length,
        method=str(series_normalization.get("method", "robust_zscore")),
        eps=float(series_normalization.get("eps", 1e-6)),
        clip=series_normalization.get("clip", 8.0),
    )

    adapted = dict(sample)
    adapted["series"] = [float(value) for value in normalized_series]
    context = dict(sample.get("context", {}) or {})
    context["series_normalization"] = normalization_info
    adapted["context"] = context
    return adapted, normalization_info


def _draw_normalized_plain_plot(series: list[float], *, width: int, height: int):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    margin_left = 22
    margin_right = 14
    margin_top = 18
    margin_bottom = 22

    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right - 1
    plot_bottom = height - margin_bottom - 1
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    values = [float(point) for point in series]
    value_min = min(values)
    value_max = max(values)
    if value_max == value_min:
        value_min -= 1.0
        value_max += 1.0
    value_pad = 0.06 * (value_max - value_min)
    value_min -= value_pad
    value_max += value_pad

    def x_to_px(index: int) -> float:
        if len(values) <= 1:
            return float(plot_left)
        return plot_left + (index / (len(values) - 1)) * plot_width

    def y_to_px(value: float) -> float:
        ratio = (value - value_min) / (value_max - value_min)
        return plot_bottom - ratio * plot_height

    draw.line([(plot_left, plot_top), (plot_left, plot_bottom)], fill=(192, 198, 210), width=1)
    draw.line([(plot_left, plot_bottom), (plot_right, plot_bottom)], fill=(192, 198, 210), width=1)
    points = [(x_to_px(idx), y_to_px(value)) for idx, value in enumerate(values)]
    draw.line(points, fill=(35, 96, 181), width=2)
    return image


def _infer_image_size(image_path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as image:
        width, height = image.size
    return int(width), int(height)


def _render_normalized_image(
    *,
    series: list[float],
    source_image_path: Path,
    output_root: Path,
    split: str,
    sample_id: str,
    series_normalization: dict[str, Any],
) -> Path:
    if not series:
        return source_image_path

    image_subdir = str(series_normalization.get("image_subdir", "images_normalized"))
    output_path = output_root / image_subdir / split / f"{sample_id}.png"
    if output_path.exists() and not bool(series_normalization.get("overwrite_images", False)):
        return output_path.resolve()

    ensure_dir(output_path.parent)
    width = series_normalization.get("image_width")
    height = series_normalization.get("image_height")
    if width in (None, "") or height in (None, ""):
        width, height = _infer_image_size(source_image_path)
    image = _draw_normalized_plain_plot(series, width=int(width), height=int(height))
    image.save(output_path, format="PNG", optimize=True)
    return output_path.resolve()


def _record_image_relpath(image_path: Path, *, source_root: Path, output_root: Path | None) -> str:
    for root in (source_root, output_root):
        if root is None:
            continue
        try:
            return str(image_path.relative_to(root))
        except ValueError:
            continue
    return str(image_path)


def _prepare_model_input_sample(
    sample: dict[str, Any],
    *,
    source_root: Path,
    output_root: Path | None,
    split: str,
    image_subdir: str | None,
    series_normalization: dict[str, Any],
) -> tuple[dict[str, Any], Path, Path, dict[str, Any] | None]:
    source_image_path = _resolve_image_path(source_root, sample, split, image_subdir=image_subdir)
    if not source_image_path.exists():
        raise FileNotFoundError(f"Missing image for sample {sample['sample_id']}: {source_image_path}")

    adapted_sample, normalization_info = _adapt_sample_series(
        sample,
        series_normalization=series_normalization,
    )
    text_sample = (
        adapted_sample
        if series_normalization.get("enabled", False) and series_normalization.get("normalize_text", True)
        else sample
    )

    image_path = source_image_path
    if series_normalization.get("enabled", False) and series_normalization.get("normalize_image", True):
        if output_root is None:
            raise ValueError("series_normalization.normalize_image=true requires output_root.")
        image_path = _render_normalized_image(
            series=list(adapted_sample.get("series", [])),
            source_image_path=source_image_path,
            output_root=output_root,
            split=split,
            sample_id=str(sample["sample_id"]),
            series_normalization=series_normalization,
        )

    return text_sample, image_path, source_image_path, normalization_info


def _render_indexed_series_text(
    sample: dict[str, Any],
    *,
    precision: int = 4,
    compact: bool = False,
) -> str:
    series = sample.get("series", [])
    if not series:
        return "Sequence length: 0.\nIndexed values:\n"

    lines = [f"Sequence length: {len(series)}.", "Indexed values:"]
    if compact:
        lines.append(
            ", ".join(f"{idx}:{float(value):.{precision}f}" for idx, value in enumerate(series))
        )
    else:
        lines.extend(f"{idx}: {float(value):.{precision}f}" for idx, value in enumerate(series))
    return "\n".join(lines)


def _record_from_sample(
    sample: dict[str, Any],
    *,
    source_root: Path,
    output_root: Path | None,
    split: str,
    system_prompt: str,
    user_prompt: str,
    image_subdir: str | None = None,
    include_indexed_series_text: bool = False,
    indexed_series_precision: int = 4,
    indexed_series_compact: bool = False,
    series_normalization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scope = annotation_scope(sample)
    if scope == "localization_only":
        target = localization_target(sample)
        system_prompt, user_prompt = LOCALIZATION_SYSTEM_PROMPT, LOCALIZATION_USER_PROMPT
        include_indexed_series_text, indexed_series_precision = True, 6
    else:
        target = _sample_to_result(sample).to_dict()
    normalization_cfg = _coerce_series_normalization_config(series_normalization)
    text_sample, image_path, source_image_path, normalization_info = _prepare_model_input_sample(
        sample,
        source_root=source_root,
        output_root=output_root,
        split=split,
        image_subdir=image_subdir,
        series_normalization=normalization_cfg,
    )
    resolved_user_prompt = user_prompt
    if include_indexed_series_text:
        series_text = _render_indexed_series_text(
            text_sample,
            precision=indexed_series_precision,
            compact=indexed_series_compact,
        )
        prompt_parts = [user_prompt.rstrip()]
        if normalization_info and normalization_cfg.get("normalize_text", True):
            prompt_note = str(normalization_cfg.get("prompt_note", "")).strip()
            if prompt_note:
                prompt_parts.append(prompt_note)
        prompt_parts.append(series_text)
        resolved_user_prompt = "\n\n".join(part for part in prompt_parts if part)

    metadata = {
        "target_adapter_version": TARGET_ADAPTER_VERSION,
        "annotation_scope": scope,
        "semantic_label_available": scope == "full_grounding",
        "indexed_series_text_enabled": include_indexed_series_text,
        "indexed_series_precision": indexed_series_precision,
        "series_length": int(sample.get("context", {}).get("series_length", len(sample.get("series", [])))),
        "background_noise": sample.get("context", {}).get("background_noise"),
        "source_model": sample.get("context", {}).get("source_model"),
        "source_dataset": sample.get("context", {}).get("source_dataset"),
        "source_file": sample.get("context", {}).get("source_file"),
        "window_start_global": sample.get("context", {}).get("window_start_global"),
        "window_end_global": sample.get("context", {}).get("window_end_global"),
        "sampling_regime": sample.get("context", {}).get("sampling_regime"),
        "anomaly_type": _sample_key(sample),
        "num_events": len(sample.get("events", [])),
        "num_positive_points": int(sum(int(point) for point in sample.get("point_labels", []))),
    }
    if normalization_info:
        metadata["series_normalization"] = normalization_info
        metadata["original_image_path"] = str(source_image_path)

    return {
        "id": sample["sample_id"],
        "split": split,
        "image_path": str(image_path),
        "image_relpath": _record_image_relpath(image_path, source_root=source_root, output_root=output_root),
        "system_prompt": system_prompt,
        "user_prompt": resolved_user_prompt,
        "assistant_text": json.dumps(target, ensure_ascii=False, separators=(",", ": ")),
        "target": target,
        "metadata": metadata,
        "raw_sample": sample,
    }


def _record_from_qa_pair(
    sample: dict[str, Any],
    qa_pair: dict[str, Any],
    *,
    source_root: Path,
    output_root: Path | None,
    split: str,
    system_prompt: str,
    user_prompt_template: str,
    image_subdir: str | None = None,
    include_indexed_series_text: bool = False,
    indexed_series_precision: int = 4,
    indexed_series_compact: bool = False,
    series_normalization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if annotation_scope(sample) == "localization_only":
        raise ValueError("Point-label-only datasets cannot supply semantic QA targets")
    sample_id = sample["sample_id"]
    qa_id = str(qa_pair.get("id") or f"{sample_id}_qa")
    question = str(qa_pair["question"])
    answer = str(qa_pair["answer"])
    normalization_cfg = _coerce_series_normalization_config(series_normalization)
    text_sample, image_path, source_image_path, normalization_info = _prepare_model_input_sample(
        sample,
        source_root=source_root,
        output_root=output_root,
        split=split,
        image_subdir=image_subdir,
        series_normalization=normalization_cfg,
    )

    resolved_user_prompt = user_prompt_template.format(question=question).rstrip()
    if include_indexed_series_text:
        series_text = _render_indexed_series_text(
            text_sample,
            precision=indexed_series_precision,
            compact=indexed_series_compact,
        )
        prompt_parts = [resolved_user_prompt]
        if normalization_info and normalization_cfg.get("normalize_text", True):
            prompt_note = str(normalization_cfg.get("prompt_note", "")).strip()
            if prompt_note:
                prompt_parts.append(prompt_note)
        prompt_parts.append(series_text)
        resolved_user_prompt = "\n\n".join(part for part in prompt_parts if part)

    metadata = {
        "target_adapter_version": TARGET_ADAPTER_VERSION,
        "annotation_scope": "full_grounding",
        "semantic_label_available": True,
        "task_type": "qa",
        "parent_sample_id": sample_id,
        "qa_id": qa_id,
        "qa_question": question,
        "series_length": int(sample.get("context", {}).get("series_length", len(sample.get("series", [])))),
        "background_noise": sample.get("context", {}).get("background_noise"),
        "source_model": sample.get("context", {}).get("source_model"),
        "source_dataset": sample.get("context", {}).get("source_dataset"),
        "source_file": sample.get("context", {}).get("source_file"),
        "sampling_regime": sample.get("context", {}).get("sampling_regime"),
        "anomaly_type": _sample_key(sample),
        "num_events": len(sample.get("events", [])),
        "num_positive_points": int(sum(int(point) for point in sample.get("point_labels", []))),
    }
    if normalization_info:
        metadata["series_normalization"] = normalization_info
        metadata["original_image_path"] = str(source_image_path)

    return {
        "id": qa_id,
        "split": split,
        "image_path": str(image_path),
        "image_relpath": _record_image_relpath(image_path, source_root=source_root, output_root=output_root),
        "system_prompt": system_prompt,
        "user_prompt": resolved_user_prompt,
        "assistant_text": answer,
        "target": {"answer": answer},
        "metadata": metadata,
        "raw_sample": sample,
    }


def _write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            serializable = dict(record)
            serializable.pop("raw_sample", None)
            f.write(json.dumps(serializable, ensure_ascii=False) + "\n")


def build_vlm_sft_dataset(
    *,
    source_root: Path,
    output_root: Path,
    split_file_suffix: str = "",
    image_subdir: str | None = None,
    include_grounding_records: bool = True,
    include_qa_pairs: bool = False,
    qa_system_prompt: str | None = None,
    qa_user_prompt_template: str = "Answer this question about the time-series plot exactly and concisely.\n\nQuestion: {question}",
    include_indexed_series_text: bool = False,
    indexed_series_precision: int = 4,
    indexed_series_compact: bool = False,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
    balanced_by_type: bool = True,
    source_dataset_filter: str | list[str] | tuple[str, ...] | None = None,
    series_normalization: dict[str, Any] | str | bool | None = None,
    seed: int = 2026,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_prompt: str = DEFAULT_USER_PROMPT,
) -> dict[str, Any]:
    ensure_dir(output_root)
    if any((output_root / f"{split}.jsonl").exists() for split in ("train", "val", "test")):
        previous_manifest = output_root / "manifest.json"
        if not previous_manifest.exists() or json.loads(previous_manifest.read_text(encoding="utf-8")).get("target_adapter_version") != TARGET_ADAPTER_VERSION:
            raise ValueError("Refusing to overwrite an old/unversioned dataset cache; use a new output directory")
    normalization_cfg = _coerce_series_normalization_config(series_normalization)
    split_limits = {
        "train": max_train_samples,
        "val": max_val_samples,
        "test": max_test_samples,
    }
    split_seed_offset = {"train": 11, "val": 29, "test": 47}

    manifest: dict[str, Any] = {
        "target_adapter_version": TARGET_ADAPTER_VERSION,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "split_file_suffix": split_file_suffix,
        "image_subdir": image_subdir,
        "include_grounding_records": include_grounding_records,
        "include_qa_pairs": include_qa_pairs,
        "source_dataset_filter": sorted(_normalize_source_dataset_filter(source_dataset_filter) or []),
        "indexed_series_text": {
            "enabled": include_indexed_series_text,
            "precision": indexed_series_precision,
            "compact": indexed_series_compact,
        },
        "series_normalization": normalization_cfg,
        "schema": {
            "localization_only": {"keys": ["evidence"], "evidence_keys": ["start", "end"]},
            "type_values": ["point", "freq", "trend", "range"],
            "keys": ["evidence", "summary"],
            "evidence_keys": ["start", "end", "type", "strength", "direction"],
        },
        "prompts": {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        },
        "splits": {},
    }
    selected_source_datasets = _normalize_source_dataset_filter(source_dataset_filter)

    for split, limit in split_limits.items():
        raw_samples = _load_split_json(source_root, split, split_file_suffix=split_file_suffix)
        raw_samples = _filter_raw_samples_by_source_dataset(
            raw_samples,
            source_dataset_filter=selected_source_datasets,
        )
        split_records = []
        for sample in raw_samples:
            if include_grounding_records:
                split_records.append(
                    _record_from_sample(
                        sample,
                        source_root=source_root,
                        output_root=output_root,
                        split=split,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        image_subdir=image_subdir,
                        include_indexed_series_text=include_indexed_series_text,
                        indexed_series_precision=indexed_series_precision,
                        indexed_series_compact=indexed_series_compact,
                        series_normalization=normalization_cfg,
                    )
                )
            if include_qa_pairs:
                qa_pairs = sample.get("qa_pairs", [])
                if not isinstance(qa_pairs, list):
                    raise ValueError(f"{sample.get('sample_id', '<missing sample_id>')}: qa_pairs must be a list")
                for qa_pair in qa_pairs:
                    if not isinstance(qa_pair, dict):
                        raise ValueError(f"{sample.get('sample_id', '<missing sample_id>')}: qa_pair must be an object")
                    split_records.append(
                        _record_from_qa_pair(
                            sample,
                            qa_pair,
                            source_root=source_root,
                            output_root=output_root,
                            split=split,
                            system_prompt=qa_system_prompt or system_prompt,
                            user_prompt_template=qa_user_prompt_template,
                            image_subdir=image_subdir,
                            include_indexed_series_text=include_indexed_series_text,
                            indexed_series_precision=indexed_series_precision,
                            indexed_series_compact=indexed_series_compact,
                            series_normalization=normalization_cfg,
                        )
                    )
        split_records = _subsample_records(
            split_records,
            limit,
            balanced_by_type=balanced_by_type,
            seed=seed + split_seed_offset[split],
        )
        for record in split_records:
            validate_prepared_record(record)
        ids = [record["id"] for record in split_records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate record IDs in {split}")
        _write_jsonl(split_records, output_root / f"{split}.jsonl")

        type_counter = Counter(record["metadata"]["anomaly_type"] for record in split_records)
        task_counter = Counter(str(record["metadata"].get("task_type", "grounding")) for record in split_records)
        dataset_counter = Counter(str(record["metadata"].get("source_dataset", "unknown")) for record in split_records)
        manifest["splits"][split] = {
            "num_records": len(split_records),
            "task_counts": dict(task_counter),
            "type_counts": dict(type_counter),
            "source_dataset_counts": dict(dataset_counter),
            "annotation_scope_counts": dict(Counter(record["metadata"]["annotation_scope"] for record in split_records)),
        }

    dump_json(manifest, output_root / "manifest.json")
    return manifest
