"""Tests for token-filter wiring in SGPA transcript preparation."""

from unittest.mock import MagicMock, patch

from nlp_shap.alignment.sgpa import SpectrogramGuidedAligner
from nlp_shap.masking.filters import ExcludePunctuationTokensFilter, KeepAllTokens


def _patched_transformers() -> tuple[MagicMock, MagicMock]:
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {"H": 0, "E": 1, "L": 2, "O": 3, "|": 4}
    tokenizer.pad_token_id = 5
    tokenizer.convert_tokens_to_ids.side_effect = lambda char: {
        "H": 0,
        "E": 1,
        "L": 2,
        "O": 3,
        "|": 4,
    }.get(char, 5)
    processor = MagicMock()
    processor.tokenizer = tokenizer
    processor_cls = MagicMock()
    processor_cls.from_pretrained.return_value = processor
    model = MagicMock()
    model.to.return_value = model
    model_cls = MagicMock()
    model_cls.from_pretrained.return_value = model
    return processor_cls, model_cls


@patch("nlp_shap.alignment.sgpa._require_transformers")
def test_prepare_transcript_drops_excluded_punctuation(mock_require: MagicMock) -> None:
    """SGPA target segments honor the configured token filter."""
    processor_cls, model_cls = _patched_transformers()
    mock_require.return_value = (processor_cls, model_cls)
    aligner = SpectrogramGuidedAligner(
        device="cpu",
        model_name="dummy",
        token_filter=ExcludePunctuationTokensFilter(),
    )
    _full, targets, _clean, _tokens = aligner._prepare_transcript("hello , world !")
    assert targets == ["hello", "world"]


@patch("nlp_shap.alignment.sgpa._require_transformers")
def test_prepare_transcript_keep_all_retains_punctuation(
    mock_require: MagicMock,
) -> None:
    """KeepAllTokens leaves punctuation tokens in the target segment list."""
    processor_cls, model_cls = _patched_transformers()
    mock_require.return_value = (processor_cls, model_cls)
    aligner = SpectrogramGuidedAligner(
        device="cpu",
        model_name="dummy",
        token_filter=KeepAllTokens(),
    )
    _full, targets, _clean, _tokens = aligner._prepare_transcript("hello !")
    assert targets == ["hello", "!"]
