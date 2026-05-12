from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MessageContext:
    text: str | None
    author_id: int | str
    guild_id: int | str | None = 1
    author_is_bot: bool = False
    author_account_age_seconds: float | None = None
    member_join_age_seconds: float | None = None
    message_length: int | None = None
    word_count: int | None = None
    has_link: bool = False
    has_mention: bool = False
    num_roles: int | None = None


@dataclass(frozen=True)
class ScreeningResult:
    triggered: bool
    reasons: list[str]
