from __future__ import annotations

import math
from statistics import median
from typing import Any, Sequence


def _as_finite_floats(values: Sequence[float]) -> list[float]:
    floats = [float(value) for value in values]
    invalid = [value for value in floats if not math.isfinite(value)]
    if invalid:
        raise ValueError(f"Series contains non-finite values: {invalid[:4]}")
    return floats


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _std(values: Sequence[float], center: float) -> float:
    if not values:
        return 0.0
    return float(math.sqrt(sum((value - center) ** 2 for value in values) / len(values)))


def _clip_value(value: float, clip: float | None) -> float:
    if clip is None or clip <= 0:
        return value
    return max(-float(clip), min(float(clip), value))


def normalize_series_window(
    series: Sequence[float],
    *,
    fit_length: int | None = None,
    method: str = "robust_zscore",
    eps: float = 1e-6,
    clip: float | None = 8.0,
) -> tuple[list[float], dict[str, Any]]:
    """Normalize one time-series window and return normalized values plus metadata.

    The normalization statistics are fit on ``series[:fit_length]`` when
    ``fit_length`` is provided. This lets padded windows avoid fitting on the
    repeated padding tail while still transforming the full padded sequence.
    """

    values = _as_finite_floats(series)
    if not values:
        return [], {
            "enabled": True,
            "requested_method": method,
            "method": "empty",
            "fit_length": 0,
            "center": 0.0,
            "scale": 1.0,
            "eps": float(eps),
            "clip": clip,
        }

    if fit_length is None:
        fit_values = list(values)
    else:
        bounded_length = max(0, min(int(fit_length), len(values)))
        fit_values = values[:bounded_length] or list(values)

    requested_method = str(method or "robust_zscore").strip().lower()
    if requested_method in {"none", "raw", "identity"}:
        return list(values), {
            "enabled": False,
            "requested_method": requested_method,
            "method": "identity",
            "fit_length": len(fit_values),
            "center": 0.0,
            "scale": 1.0,
            "eps": float(eps),
            "clip": clip,
        }

    raw_mean = _mean(fit_values)
    raw_std = _std(fit_values, raw_mean)
    raw_median = float(median(fit_values))
    raw_mad = float(median(abs(value - raw_median) for value in fit_values))

    if requested_method == "zscore":
        center = raw_mean
        scale = raw_std
        method_used = "zscore"
    elif requested_method in {"robust", "robust_zscore", "mad"}:
        center = raw_median
        scale = 1.4826 * raw_mad
        method_used = "robust_zscore"
        if scale < eps:
            center = raw_mean
            scale = raw_std
            method_used = "zscore_fallback"
    else:
        raise ValueError(f"Unsupported series normalization method: {method}")

    if scale < eps:
        scale = 1.0
        method_used = "constant"

    normalized = [_clip_value((value - center) / scale, clip) for value in values]
    metadata = {
        "enabled": True,
        "requested_method": requested_method,
        "method": method_used,
        "fit_length": len(fit_values),
        "center": float(center),
        "scale": float(scale),
        "eps": float(eps),
        "clip": clip,
        "raw_min": float(min(fit_values)),
        "raw_max": float(max(fit_values)),
        "raw_mean": float(raw_mean),
        "raw_std": float(raw_std),
        "raw_median": float(raw_median),
        "raw_mad": float(raw_mad),
        "normalized_min": float(min(normalized)),
        "normalized_max": float(max(normalized)),
    }
    return normalized, metadata
