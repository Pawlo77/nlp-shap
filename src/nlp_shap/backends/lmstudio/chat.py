"""Convert conversation snapshots to LM Studio chat contexts."""

from typing import Any

from ...domain.conversation import ConversationSnapshot
from ...domain.enums import Role


def snapshot_to_chat(snapshot: ConversationSnapshot, chat_factory: Any) -> Any:
    """Build an LM Studio chat object from ``snapshot`` turns."""
    chat: Any | None = None
    for turn in snapshot.turns:
        for message in turn.messages:
            match message.role:
                case Role.SYSTEM:
                    if chat is None:
                        chat = chat_factory(message.text)
                    else:
                        chat.add_system_prompt(message.text)
                case Role.USER:
                    if chat is None:
                        chat = chat_factory()
                    chat.add_user_message(message.text)
                case Role.ASSISTANT:
                    if chat is None:
                        chat = chat_factory()
                    chat.add_assistant_response(message.text)
                case _ as unsupported:
                    msg = (
                        f"unsupported message role for LM Studio chat: {unsupported!r}"
                    )
                    raise ValueError(msg)
    if chat is None:
        chat = chat_factory()
    return chat
