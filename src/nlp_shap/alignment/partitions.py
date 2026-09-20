"""SGPA-aligned segment player partition plugin."""

from dataclasses import dataclass, field

from ..domain.conversation import ConversationSnapshot
from ..domain.players import PlayerSet
from ..masking.filters import KeepAllTokens, TokenFilter
from .segments import AudioSegment


@dataclass(frozen=True, slots=True)
class SgpaSegmentPartitioner:
    """Partition a multimodal snapshot into SGPA audio-segment players."""

    segments: tuple[AudioSegment, ...] = ()
    """Pre-aligned segments in coalition-mask order."""

    token_filter: TokenFilter = field(default_factory=KeepAllTokens)
    """Filter applied before players are materialized."""

    @property
    def name(self) -> str:
        """Return the registered partition identifier."""
        return "sgpa_segments"

    def filtered_segments(self) -> tuple[AudioSegment, ...]:
        """Return segments retained after :attr:`token_filter`."""
        return tuple(
            segment
            for segment in self.segments
            if self.token_filter.keeps(segment.token)
        )

    def partition(self, snapshot: ConversationSnapshot) -> PlayerSet:
        """Derive ordered SGPA segment players from ``snapshot`` and segments."""
        if not snapshot.has_audio():
            msg = "sgpa_segments partition requires a snapshot with audio"
            raise ValueError(msg)
        kept = self.filtered_segments()
        if not kept:
            msg = "snapshot must contain at least one explainable SGPA segment"
            raise ValueError(msg)
        player_ids = tuple(
            f"{snapshot.snapshot_id}:sgpa:{index}" for index in range(len(kept))
        )
        return PlayerSet(player_ids=player_ids)
