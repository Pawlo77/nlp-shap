"""Player partition plugins for conversation snapshots."""

from dataclasses import dataclass

from ..domain.conversation import ConversationSnapshot
from ..domain.players import PlayerSet
from .tokens import player_id_for_span, tokenize_snapshot


@dataclass(frozen=True, slots=True)
class TokenPartitioner:
    """Partition a snapshot into whitespace-delimited text tokens."""

    @property
    def name(self) -> str:
        """Return the registered partition identifier."""
        return "tokens"

    def partition(self, snapshot: ConversationSnapshot) -> PlayerSet:
        """Derive ordered token players from the snapshot text."""
        spans = tokenize_snapshot(snapshot)
        if not spans:
            msg = "snapshot must contain at least one explainable token"
            raise ValueError(msg)
        player_ids = tuple(
            player_id_for_span(snapshot.snapshot_id, span) for span in spans
        )
        return PlayerSet(player_ids=player_ids)
