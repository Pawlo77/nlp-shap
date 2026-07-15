"""Token label helpers for attribution renderers."""

from ..domain.conversation import ConversationSnapshot
from ..domain.players import PlayerSet
from ..masking.tokens import player_id_for_span, tokenize_snapshot


def token_labels(
    snapshot: ConversationSnapshot,
    player_set: PlayerSet,
) -> tuple[str, ...]:
    """Map coalition player ids to whitespace token text."""
    spans = tokenize_snapshot(snapshot)
    labels_by_id = {
        player_id_for_span(snapshot.snapshot_id, span): span.text for span in spans
    }
    try:
        return tuple(labels_by_id[player_id] for player_id in player_set.player_ids)
    except KeyError as exc:
        msg = "player set does not align with snapshot token partition"
        raise ValueError(msg) from exc
