from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .taxonomy import TYPE_NAMES, canonicalize_type_name


DEFAULT_NO_ANOMALY_SUMMARY = "No anomaly is detected."
GROUNDING_CONTRACT_VERSION = "grounding-output-v2"


def parse_json_object(output: str | dict[str, Any]) -> dict[str, Any]:
    """Decode one complete JSON object, never extracting JSON from prose.

    Dictionary inputs are supported for programmatic callers. Text predictions
    must contain exactly one object; ambiguous duplicate keys and non-JSON
    numeric constants are rejected instead of silently repaired.
    """
    if isinstance(output, dict):
        # Apply the same JSON numeric/serialization rules to programmatic
        # payloads, including optional fields that the schema does not score.
        output = json.dumps(output, allow_nan=False)
    if not isinstance(output, str):
        raise ValueError("Grounding output must be JSON text or a JSON object.")

    def reject_constant(value: str) -> Any:
        raise ValueError(f"Non-JSON numeric constant: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError(f"Duplicate JSON key: {key}")
            payload[key] = value
        return payload

    payload = json.loads(output, parse_constant=reject_constant, object_pairs_hook=unique_object)
    if not isinstance(payload, dict):
        raise ValueError("Grounding output must be a JSON object.")
    return payload


@dataclass(frozen=True)
class EvidenceItem:
    start: int
    end: int
    type: str
    strength: str
    direction: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": int(self.start),
            "end": int(self.end),
            "type": canonicalize_type_name(self.type),
            "strength": str(self.strength),
            "direction": str(self.direction),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvidenceItem":
        if not isinstance(payload, dict):
            raise ValueError("Each evidence item must be a JSON object.")
        required = {"start", "end", "type", "strength", "direction"}
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError(f"Evidence item is missing required fields: {', '.join(missing)}")

        start = payload["start"]
        end = payload["end"]
        if isinstance(start, bool) or not isinstance(start, int):
            raise ValueError("Evidence start must be an integer.")
        if isinstance(end, bool) or not isinstance(end, int):
            raise ValueError("Evidence end must be an integer.")
        if start < 0 or end < start:
            raise ValueError("Evidence indices must satisfy 0 <= start <= end.")

        if not isinstance(payload["type"], str):
            raise ValueError("Evidence type must be a supported anomaly type string.")
        type_name = canonicalize_type_name(payload["type"])
        if type_name not in TYPE_NAMES:
            raise ValueError(f"Unsupported anomaly type: {payload['type']}")
        strength = payload["strength"]
        direction = payload["direction"]
        if not isinstance(strength, str) or not strength.strip():
            raise ValueError("Evidence strength must be a non-empty string.")
        if not isinstance(direction, str) or not direction.strip():
            raise ValueError("Evidence direction must be a non-empty string.")

        return cls(
            start=start,
            end=end,
            type=type_name,
            strength=strength.strip(),
            direction=direction.strip(),
        )


@dataclass(frozen=True)
class GroundingResult:
    evidence: list[EvidenceItem]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence": [item.to_dict() for item in self.evidence],
            "summary": str(self.summary),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GroundingResult":
        if not isinstance(payload, dict):
            raise ValueError("Grounding output must be a JSON object.")
        missing = [key for key in ("evidence", "summary") if key not in payload]
        if missing:
            raise ValueError(f"Grounding output is missing required fields: {', '.join(missing)}")
        if not isinstance(payload["evidence"], list):
            raise ValueError("Grounding evidence must be a list.")
        if not isinstance(payload["summary"], str) or not payload["summary"].strip():
            raise ValueError("Grounding summary must be a non-empty string.")
        evidence = [EvidenceItem.from_dict(item) for item in payload["evidence"]]
        summary = payload["summary"].strip()
        return cls(evidence=evidence, summary=summary)

    @classmethod
    def no_anomaly(cls) -> "GroundingResult":
        return cls(evidence=[], summary=DEFAULT_NO_ANOMALY_SUMMARY)


def parse_grounding_output(
    output: str | dict[str, Any],
    *,
    series_length: int | None = None,
) -> GroundingResult:
    """Shared full-grounding prediction contract for reward, mining and eval."""
    if series_length is not None and (type(series_length) is not int or series_length <= 0):
        raise ValueError("series_length must be a positive integer.")
    result = GroundingResult.from_dict(parse_json_object(output))
    if series_length is not None:
        for item in result.evidence:
            if item.end >= series_length:
                raise ValueError(
                    f"Evidence interval [{item.start}, {item.end}] exceeds series length {series_length}."
                )
    return result
