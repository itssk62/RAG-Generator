from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import litellm


class LiteLLMClient:
    """Any provider LiteLLM supports; the model is a config string (e.g. `gpt-4o-mini`)."""

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 700, timeout_s: float = 60, **extra: Any):
        self.model = model
        self.params = {"temperature": temperature, "max_tokens": max_tokens, "timeout": timeout_s, "num_retries": 2, **extra}

    def missing_keys(self) -> list[str]:
        """Provider env vars LiteLLM expects for this model but cannot find."""
        try:
            return list(litellm.validate_environment(self.model).get("missing_keys", []))
        except Exception:
            return []

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = litellm.completion(model=self.model, messages=messages, **self.params)
        return response.choices[0].message.content or ""

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        for part in litellm.completion(model=self.model, messages=messages, stream=True, **self.params):
            delta = part.choices[0].delta.content if part.choices else None
            if delta:
                yield delta
