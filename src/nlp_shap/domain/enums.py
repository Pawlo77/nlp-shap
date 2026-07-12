"""Conversation and explanation enumerations for the text MVP."""

from enum import StrEnum


class Role(StrEnum):
    """Participant role attached to a message or token."""

    USER = "user"
    """End-user or human-provided input."""

    ASSISTANT = "assistant"
    """Model-generated output."""

    SYSTEM = "system"
    """System-level instruction or steering text."""


class SystemRolesSetup(StrEnum):
    """How system-role tokens participate in explainability."""

    NONE = "none"
    """All tokens, including system tokens, are explainable."""

    SYSTEM = "system"
    """System tokens are excluded from explainability."""

    SYSTEM_ASSISTANT = "system_assistant"
    """System and assistant tokens are excluded from explainability."""


class EmbeddingMode(StrEnum):
    """Whether value functions use static or contextual embeddings."""

    STATIC = "static"
    """Embeddings computed from final generated tokens only."""

    CONTEXTUAL = "contextual"
    """Embeddings derived from model internal states."""
