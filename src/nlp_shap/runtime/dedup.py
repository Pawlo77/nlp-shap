"""Coalition deduplication keys and registry."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from ..masking.codec import MaskCodec
from ..pipeline.config import DedupConfig, GenerationConfig


def dedup_enabled(config: DedupConfig, generation: GenerationConfig) -> bool:
    """Return whether coalition deduplication should be active."""
    match config.enabled:
        case "on":
            return True
        case "off":
            return False
        case "auto":
            return generation.temperature == 0.0
        case _ as unreachable:
            raise AssertionError(f"unsupported dedup mode: {unreachable!r}")


def build_coalition_key(
    *,
    snapshot_id: str,
    player_ids: tuple[str, ...],
    mask_present: tuple[bool, ...],
    absence_policy: str,
    model_id: str,
    generation: GenerationConfig,
) -> str:
    """Build a stable SHA256 coalition key for deduplication."""
    packed = MaskCodec.pack(mask_present)
    payload = {
        "snapshot_id": snapshot_id,
        "player_ids": list(player_ids),
        "mask_words": packed.words.hex(),
        "mask_n_bits": packed.n_bits,
        "absence_policy": absence_policy,
        "model_id": model_id,
        "generation": {
            "max_new_tokens": generation.max_new_tokens,
            "temperature": generation.temperature,
            "top_k": generation.top_k,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass
class CoalitionDedupRegistry:
    """Track coalition keys that already required backend execution."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def observe(self, coalition_key: str) -> bool:
        """Record a coalition key and return whether it is newly observed."""
        is_new = coalition_key not in self._seen
        if is_new:
            self._seen.add(coalition_key)
        return is_new

    def contains(self, coalition_key: str) -> bool:
        """Return whether a coalition key was already observed."""
        return coalition_key in self._seen

    def __len__(self) -> int:
        """Return the number of unique coalition keys observed."""
        return len(self._seen)
