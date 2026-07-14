"""Logprob-based value function."""

import hashlib

from ..domain.generation import GenerationRecord
from ..protocols.backend import GenerationResult


class LogprobValue:
    """Score generations from summed token log probabilities."""

    @property
    def name(self) -> str:
        """Return the registered value-function identifier."""
        return "logprob"

    def score(self, base: GenerationResult, candidate: GenerationResult) -> float:
        """Return utility from candidate logprobs relative to ``base``."""
        _ = base
        if isinstance(candidate, GenerationRecord) and candidate.logprobs:
            return float(sum(candidate.logprobs))
        return _stub_logprob(candidate)


def _stub_logprob(generation: GenerationResult) -> float:
    """Return a deterministic pseudo-logprob when the backend omits logprobs."""
    digest = hashlib.sha256(generation.text.encode()).digest()
    magnitude = int.from_bytes(digest[:8], byteorder="big", signed=False)
    normalized = (magnitude % 10_000) / 10_000.0
    return -1.0 - normalized
