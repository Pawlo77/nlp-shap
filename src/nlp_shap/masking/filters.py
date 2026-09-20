"""Token filters for explainability player selection."""

from dataclasses import dataclass
from typing import Protocol

DEFAULT_PUNCTUATION_PHRASES: frozenset[str] = frozenset({".", ",", "!", "?", ";", ":"})
"""Default punctuation phrases for :class:`ExcludePunctuationTokensFilter`."""


class TokenFilter(Protocol):
    """Strategy that marks transcript tokens as excluded from explainability."""

    @property
    def phrases_to_exclude(self) -> frozenset[str]:
        """Return phrases that must not become explainability players."""

    def keeps(self, token: str) -> bool:
        """Return whether ``token`` should remain a player or alignment target."""


@dataclass(frozen=True, slots=True)
class KeepAllTokens:
    """Token filter that excludes nothing."""

    phrases_to_exclude: frozenset[str] = frozenset()
    """Empty exclusion set."""

    def keeps(self, token: str) -> bool:
        """Always keep ``token``."""
        del token
        return True


@dataclass(frozen=True, slots=True)
class ExcludePunctuationTokensFilter:
    """Drop common punctuation tokens from player and SGPA target lists."""

    phrases_to_exclude: frozenset[str] = DEFAULT_PUNCTUATION_PHRASES
    """Configurable set of phrases excluded from SHAP calculations."""

    def keeps(self, token: str) -> bool:
        """Return ``False`` when ``token`` is in :attr:`phrases_to_exclude`."""
        return token not in self.phrases_to_exclude
