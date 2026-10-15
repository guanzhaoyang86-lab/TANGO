from __future__ import annotations

TYPE_NAMES = ("point", "freq", "trend", "range")
TYPE_TO_ID = {name: idx for idx, name in enumerate(TYPE_NAMES)}
TYPE_NAME_ALIASES = {
    "frequency": "freq",
    "level": "range",
}
NUM_TYPES = len(TYPE_NAMES)


def canonicalize_type_name(name: str) -> str:
    normalized = name.strip().lower()
    return TYPE_NAME_ALIASES.get(normalized, normalized)


def type_name_from_id(type_idx: int) -> str:
    if 0 <= type_idx < len(TYPE_NAMES):
        return TYPE_NAMES[type_idx]
    return str(type_idx)
