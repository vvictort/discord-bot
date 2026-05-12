from __future__ import annotations

import re

from src.scam_detector.models import MessageContext

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_message_text(text: str | None) -> str:
    if text is None:
        return ""
    return _WHITESPACE_RE.sub(" ", text.lower()).strip()


def is_eligible_message(message: MessageContext) -> bool:
    return bool(
        normalize_message_text(message.text)
        and not message.author_is_bot
        and message.guild_id is not None
    )
