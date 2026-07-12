"""Hot LRU cache for coalition generation results."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class HotResultStore:
    """In-memory LRU store for recently generated coalition outputs."""

    maxsize: int = 128
    """Maximum number of coalition results retained in memory."""

    _values: OrderedDict[str, str] = field(default_factory=OrderedDict)
    """LRU mapping from coalition key to generated text."""

    def get(self, coalition_key: str) -> str | None:
        """Return a cached generation or ``None`` when absent."""
        if coalition_key not in self._values:
            return None
        self._values.move_to_end(coalition_key)
        return self._values[coalition_key]

    def put(self, coalition_key: str, generation_text: str) -> None:
        """Insert or refresh a cached generation result."""
        if coalition_key in self._values:
            self._values.move_to_end(coalition_key)
        self._values[coalition_key] = generation_text
        while len(self._values) > self.maxsize:
            self._values.popitem(last=False)

    def __len__(self) -> int:
        """Return the number of cached coalition results."""
        return len(self._values)
