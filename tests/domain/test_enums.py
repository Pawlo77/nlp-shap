"""Tests for conversation domain enums."""

import pytest

from nlp_shap.domain.enums import EmbeddingMode, Role, SystemRolesSetup


def test_role_wire_values_are_stable() -> None:
    """Role members serialize to stable lowercase strings."""
    assert Role.USER.value == "user"
    assert Role.ASSISTANT.value == "assistant"
    assert Role.SYSTEM.value == "system"


def test_system_roles_setup_values_are_stable() -> None:
    """System role setup members serialize to stable strings."""
    assert SystemRolesSetup.NONE.value == "none"
    assert SystemRolesSetup.SYSTEM.value == "system"
    assert SystemRolesSetup.SYSTEM_ASSISTANT.value == "system_assistant"


def test_embedding_mode_values_are_stable() -> None:
    """Embedding mode members serialize to stable strings."""
    assert EmbeddingMode.STATIC.value == "static"
    assert EmbeddingMode.CONTEXTUAL.value == "contextual"


def test_role_rejects_unknown_value() -> None:
    """Unknown role wire values raise a lookup error."""
    with pytest.raises(ValueError, match="not a valid"):
        Role("moderator")
