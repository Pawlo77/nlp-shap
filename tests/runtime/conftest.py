"""Shared fixtures for runtime tests."""

import pytest

from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
from nlp_shap.domain.enums import Role
from nlp_shap.domain.estimands import Estimand
from nlp_shap.masking.codec import MaskCodec
from nlp_shap.pipeline.manifest import RunManifest


@pytest.fixture
def sample_snapshot() -> ConversationSnapshot:
    """Return a short conversation snapshot for runtime tests."""
    turn = Turn(messages=(Message(role=Role.USER, text="Who are you?"),))
    return ConversationSnapshot.from_turns((turn,))


@pytest.fixture
def run_manifest() -> RunManifest:
    """Return a manifest used when opening run archives."""
    return RunManifest(estimand=Estimand.SHAPLEY, run_id="runtime-test")


@pytest.fixture
def packed_mask() -> tuple[bytes, int]:
    """Return packed mask bytes and bit length for one coalition."""
    packed = MaskCodec.pack((True, False, True))
    return packed.words, packed.n_bits
