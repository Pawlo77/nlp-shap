"""Prefix-cache storage for shared prompt token prefixes."""

import hashlib
from collections.abc import Callable, Sequence
from typing import Any

from ..domain.conversation import ConversationSnapshot


def build_snapshot_prefix_hash(snapshot: ConversationSnapshot) -> str:
    """Hash the shared text prefix excluding the final whitespace token."""
    texts = [message.text for turn in snapshot.turns for message in turn.messages]
    combined = " ".join(texts)
    words = combined.split()
    prefix = " ".join(words[:-1]) if len(words) > 1 else combined
    digest = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
    return digest[:16]


class PrefixCacheManager:
    """Store and reuse transformer ``past_key_values`` for shared prefixes."""

    def __init__(self) -> None:
        self._entries: dict[tuple[int, ...], Any] = {}
        self.hits = 0
        self.misses = 0

    def get_prefix(self, token_ids: Sequence[int]) -> tuple[Any | None, int]:
        """Return cached KV state and the longest matching prefix length."""
        ids = tuple(int(token_id) for token_id in token_ids)
        for length in range(len(ids), 0, -1):
            cached = self._entries.get(ids[:length])
            if cached is not None:
                return cached, length
        return None, 0

    def lookup(self, token_ids: Sequence[int]) -> tuple[Any | None, int]:
        """Return cached KV state and update hit or miss counters."""
        matched, length = self.get_prefix(token_ids)
        if length > 0:
            self.hits += 1
        else:
            self.misses += 1
        return matched, length

    def get_at_length(self, token_ids: Sequence[int], length: int) -> Any | None:
        """Return cached KV state for an exact prefix length."""
        if length <= 0:
            return None
        return self._entries.get(
            tuple(int(token_id) for token_id in token_ids[:length])
        )

    def store(self, token_ids: Sequence[int], past_key_values: Any) -> None:
        """Persist KV state for an exact token-id prefix."""
        self._entries[tuple(int(token_id) for token_id in token_ids)] = past_key_values

    def reset(self) -> None:
        """Clear cached prefixes and counters."""
        self._entries.clear()
        self.hits = 0
        self.misses = 0


def group_jobs_for_prefix_cache[T](
    jobs: list[T],
    *,
    key: Callable[[T], str],
) -> list[T]:
    """Sort jobs so coalitions with the same prefix hash run adjacently."""
    return sorted(jobs, key=key)
