"""Shared fixtures for Text MVP sign-off tests."""

import pytest

from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
from nlp_shap.domain.enums import Role
from nlp_shap.domain.estimands import Estimand
from nlp_shap.pipeline.manifest import RunManifest


@pytest.fixture
def sample_snapshot() -> ConversationSnapshot:
    """Return a short conversation snapshot for sign-off scheduler tests."""
    turn = Turn(messages=(Message(role=Role.USER, text="Who are you?"),))
    return ConversationSnapshot.from_turns((turn,))


@pytest.fixture
def run_manifest() -> RunManifest:
    """Return a manifest used when opening run archives."""
    return RunManifest(estimand=Estimand.SHAPLEY, run_id="signoff-test")
