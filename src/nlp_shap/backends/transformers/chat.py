"""Conversation snapshot helpers for Hugging Face chat templates."""

from typing import Any

from ...domain.conversation import ConversationSnapshot
from ...domain.enums import Role


def snapshot_has_audio(snapshot: ConversationSnapshot) -> bool:
    """Return whether ``snapshot`` carries audio payloads."""
    for turn in snapshot.turns:
        for message in turn.messages:
            if message.text.startswith("audio:"):
                return True
    return False


def snapshot_to_chat_messages(snapshot: ConversationSnapshot) -> list[dict[str, str]]:
    """Convert a conversation snapshot into OpenAI-style chat messages."""
    messages: list[dict[str, str]] = []
    for turn in snapshot.turns:
        for message in turn.messages:
            match message.role:
                case Role.SYSTEM:
                    role = "system"
                case Role.USER:
                    role = "user"
                case Role.ASSISTANT:
                    role = "assistant"
                case _ as unsupported:
                    msg = f"unsupported message role: {unsupported!r}"
                    raise ValueError(msg)
            messages.append({"role": role, "content": message.text})
    return messages


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Render chat messages to a model prompt string."""
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    chat_template = getattr(tokenizer, "chat_template", None)
    if apply_template is not None and chat_template:
        try:
            rendered = apply_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except ValueError:
            rendered = None
        if isinstance(rendered, str):
            return rendered
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)
