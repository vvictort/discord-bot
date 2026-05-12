from src.scam_detector.models import MessageContext
from src.scam_detector.preprocessing import is_eligible_message, normalize_message_text
from src.scam_detector.screening import cheap_trigger_screen


def test_normalize_message_text_collapses_whitespace_and_lowercases() -> None:
    assert normalize_message_text("  FREE   Nitro\nClaim NOW  ") == "free nitro claim now"


def test_ineligible_messages_are_ignored() -> None:
    assert not is_eligible_message(MessageContext(text="", author_id=1))
    assert not is_eligible_message(MessageContext(text="hello", author_id=1, author_is_bot=True))
    assert not is_eligible_message(MessageContext(text="hello", author_id=1, guild_id=None))


def test_eligible_message_requires_text_non_bot_and_guild() -> None:
    assert is_eligible_message(MessageContext(text="hello", author_id=1, guild_id=2))


def test_cheap_screening_ignores_plain_messages() -> None:
    result = cheap_trigger_screen(MessageContext(text="thanks for the update", author_id=1))

    assert not result.triggered
    assert result.reasons == []


def test_cheap_screening_detects_scam_keywords_links_and_mentions() -> None:
    result = cheap_trigger_screen(
        MessageContext(
            text="@everyone free nitro claim https://discord.example/gift",
            author_id=1,
            has_link=True,
            has_mention=True,
        )
    )

    assert result.triggered
    assert "keyword:free nitro" in result.reasons
    assert "keyword:claim" in result.reasons
    assert "has_link" in result.reasons
    assert "has_mention" in result.reasons
