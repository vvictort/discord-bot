from __future__ import annotations

import re

from src.scam_detector.models import MessageContext

_WHITESPACE_RE = re.compile(r"\s+")
_MASS_MENTION_RE = re.compile(r"@(?:everyone|here)\b", re.IGNORECASE)
_DASH_RE = re.compile(r"[-\u2010-\u2015]+")
_APOSTROPHE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\u0060": "'",
    }
)
_QUOTE_TRANSLATION = str.maketrans(
    {
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u2033": '"',
    }
)


def normalize_message_text(text: str | None) -> str:
    if text is None:
        return ""
    normalized = text.translate(_APOSTROPHE_TRANSLATION).translate(_QUOTE_TRANSLATION)
    normalized = normalized.lower()
    normalized = _MASS_MENTION_RE.sub(lambda match: f" {match.group(0).lower()} ", normalized)
    normalized = _DASH_RE.sub(" ", normalized)
    normalized = re.sub(r"\bmac\s+book\b", "macbook", normalized)
    normalized = normalized.replace("'", "")
    normalized = re.sub(r"[^a-z0-9@%]+", " ", normalized)
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def is_eligible_message(message: MessageContext) -> bool:
    return bool(
        normalize_message_text(message.text)
        and not message.author_is_bot
        and message.guild_id is not None
    )
