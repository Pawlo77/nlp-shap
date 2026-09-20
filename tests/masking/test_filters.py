"""Tests for audio-text token filters (0.x parity)."""

import pytest

from nlp_shap.masking.filters import (
    DEFAULT_PUNCTUATION_PHRASES,
    ExcludePunctuationTokensFilter,
    KeepAllTokens,
)


def test_keep_all_tokens_has_empty_exclusion_set() -> None:
    """KeepAllTokens excludes nothing."""
    token_filter = KeepAllTokens()
    assert token_filter.phrases_to_exclude == frozenset()
    assert token_filter.keeps("hello") is True
    assert token_filter.keeps(".") is True


def test_exclude_punctuation_contains_expected_symbols() -> None:
    """Default punctuation set matches 0.x ExcludePunctuationTokensFilter."""
    token_filter = ExcludePunctuationTokensFilter()
    assert token_filter.phrases_to_exclude == DEFAULT_PUNCTUATION_PHRASES
    assert token_filter.phrases_to_exclude == frozenset({".", ",", "!", "?", ";", ":"})


@pytest.mark.parametrize("token", [".", ",", "!", "?", ";", ":"])
def test_exclude_punctuation_covers_each_symbol(token: str) -> None:
    """Each default punctuation phrase is excluded."""
    token_filter = ExcludePunctuationTokensFilter()
    assert token in token_filter.phrases_to_exclude
    assert token_filter.keeps(token) is False
    assert token_filter.keeps("word") is True


def test_exclude_punctuation_accepts_custom_exclude_set() -> None:
    """Exclude set is configurable per instance."""
    token_filter = ExcludePunctuationTokensFilter(
        phrases_to_exclude=frozenset({".", "..."})
    )
    assert token_filter.keeps(".") is False
    assert token_filter.keeps("...") is False
    assert token_filter.keeps(",") is True
