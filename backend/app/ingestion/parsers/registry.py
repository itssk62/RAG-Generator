from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from app.core.interfaces import Parser


class UnsupportedFormat(Exception):
    pass


def _load(spec: str | dict[str, Any]) -> Parser:
    if isinstance(spec, str):
        target, options = spec, {}
    else:
        target, options = spec["class"], spec.get("options", {})
    module_name, _, class_name = target.partition(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(**options)


class ParserRegistry:
    """Maps a file extension to a parser, as declared in `settings.parsers`."""

    def __init__(self, mapping: dict[str, str | dict[str, Any]]):
        self._specs = {ext.lower(): spec for ext, spec in mapping.items()}
        self._cache: dict[str, Parser] = {}

    @property
    def extensions(self) -> list[str]:
        return sorted(self._specs)

    def supports(self, filename: str) -> bool:
        return Path(filename).suffix.lower() in self._specs

    def for_file(self, filename: str) -> Parser:
        ext = Path(filename).suffix.lower()
        if ext not in self._specs:
            raise UnsupportedFormat(f"unsupported file type '{ext or filename}'; supported: {', '.join(self.extensions)}")
        if ext not in self._cache:
            self._cache[ext] = _load(self._specs[ext])
        return self._cache[ext]
