from __future__ import annotations

import json

from .schema import GroundingResult


DEFAULT_SYSTEM_PROMPT = (
    "You are a vision-language anomaly grounding model for time-series plots. "
    "Read the input PNG plot and return exactly one JSON object. "
    "The JSON object must contain two keys: `evidence` and `summary`. "
    "`evidence` must be a list. Each item in `evidence` must contain `start`, `end`, `type`, `strength`, and `direction`. "
    "`type` must be one of point, freq, trend, range. "
    "If no anomaly is detected, return `{\"evidence\": [], \"summary\": \"No anomaly is detected.\"}`. "
    "Do not output markdown, explanations, or extra text."
)

DEFAULT_USER_PROMPT = (
    "Inspect this plain time-series plot and ground the anomaly. "
    "Return one JSON object with structured evidence and one English summary sentence."
)


def render_target_text(result: GroundingResult) -> str:
    return json.dumps(result.to_dict(), ensure_ascii=False, separators=(",", ": "))


def build_chat_messages(system_prompt: str, user_prompt: str, assistant_text: str | None = None) -> list[dict]:
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    if assistant_text is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            }
        )
    return messages
