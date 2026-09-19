"""Conversation and explanation enumerations for text and multimodal inputs."""

from enum import StrEnum


class Role(StrEnum):
    """Participant role attached to a message or token."""

    USER = "user"
    """End-user or human-provided input."""

    ASSISTANT = "assistant"
    """Model-generated output."""

    SYSTEM = "system"
    """System-level instruction or steering text."""


class ModalityFlag(StrEnum):
    """Input or output modality attached to a message or token."""

    IGNORE = "ignore"
    """Skip this unit for modality-specific operations."""

    TEXT = "text"
    """Text content."""

    AUDIO = "audio"
    """Audio content."""


class SystemRolesSetup(StrEnum):
    """How system-role tokens participate in explainability."""

    NONE = "none"
    """All tokens, including system tokens, are explainable."""

    SYSTEM = "system"
    """System tokens are excluded from explainability."""

    SYSTEM_ASSISTANT = "system_assistant"
    """System and assistant tokens are excluded from explainability."""


class ModelHistoryTrackingMode(StrEnum):
    """Which generated modalities are retained in run history."""

    TEXT = "text"
    """Track text outputs only."""

    AUDIO = "audio"
    """Track audio outputs only."""

    TEXT_AUDIO = "text_audio"
    """Track both text and audio outputs."""


class EmbeddingMode(StrEnum):
    """Whether value functions use static or contextual embeddings."""

    STATIC = "static"
    """Embeddings computed from final generated tokens only."""

    CONTEXTUAL = "contextual"
    """Embeddings derived from model internal states."""
