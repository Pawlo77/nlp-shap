"""Prefix-tree cache stub for future KV-cache grouping.

Phase 2 reserves this module for prefix-aware coalition batching. Concrete
prefix-tree logic arrives with the runtime scheduler in a later release.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PrefixTreeStub:
    """Placeholder type marking the prefix-tree integration point."""

    enabled: bool = False
    """Whether prefix-tree grouping is active."""
