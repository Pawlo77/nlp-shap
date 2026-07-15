"""Generation record helpers for archived coalition text."""

from ..domain.generation import GenerationRecord


def generation_record_from_text(text: str) -> GenerationRecord:
    """Build a generation record from archived text only."""
    tokens = text.split()
    rows = tuple((_stable_token_id(token),) for token in tokens)
    return GenerationRecord(text=text, text_token_rows=rows)


def _stable_token_id(token: str) -> int:
    digest = token.encode("utf-8")
    return int.from_bytes(digest[:4], byteorder="big", signed=False) % 10_000
