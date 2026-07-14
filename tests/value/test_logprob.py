"""Tests for logprob value function."""

from nlp_shap.domain.generation import GenerationRecord
from nlp_shap.value.logprob import LogprobValue


def test_logprob_sums_candidate_logprobs() -> None:
    """Provided logprobs are summed for the candidate response."""
    base = GenerationRecord(text="base", logprobs=(-0.1,))
    candidate = GenerationRecord(text="candidate", logprobs=(-0.2, -0.3))
    value_fn = LogprobValue()
    assert value_fn.score(base, candidate) == -0.5


def test_logprob_stub_is_deterministic_without_logprobs() -> None:
    """Missing logprobs fall back to a deterministic pseudo score."""
    value_fn = LogprobValue()
    base = GenerationRecord(text="base")
    candidate = GenerationRecord(text="candidate")
    first = value_fn.score(base, candidate)
    second = value_fn.score(base, candidate)
    assert first == second
    assert first < 0.0
