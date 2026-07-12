"""Masking and absence policy protocols."""

from typing import Protocol

from ..domain.coalition import CoalitionMask
from ..domain.conversation import ConversationSnapshot


class AbsencePolicy(Protocol):
    """Render absent players when building a masked snapshot view."""

    @property
    def name(self) -> str:
        """Return the registered absence-policy identifier."""

    def apply(
        self,
        snapshot: ConversationSnapshot,
        mask: CoalitionMask,
    ) -> ConversationSnapshot:
        """Return a masked snapshot view for coalition evaluation."""
