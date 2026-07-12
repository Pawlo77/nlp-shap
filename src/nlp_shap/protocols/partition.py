"""Player partition protocol."""

from typing import Protocol

from ..domain.conversation import ConversationSnapshot
from ..domain.players import PlayerSet


class PlayerPartitioner(Protocol):
    """Partition a conversation snapshot into explainability players."""

    @property
    def name(self) -> str:
        """Return the registered partition identifier."""

    def partition(self, snapshot: ConversationSnapshot) -> PlayerSet:
        """Derive ordered players from the snapshot."""
