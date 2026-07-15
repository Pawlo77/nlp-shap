"""Prefix-tree cache stub for prefix-aware coalition batching.

Reserved for future KV-cache grouping; concrete prefix-tree logic is not yet
implemented.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PrefixTreeStub:
    """Placeholder type marking the prefix-tree integration point."""

    enabled: bool = False
    """Whether prefix-tree grouping is active."""
