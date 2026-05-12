from src.scam_detector.models import MessageContext
from src.scam_detector.preprocessing import is_eligible_message, normalize_message_text
from src.scam_detector.screening import cheap_trigger_screen


def test_normalize_message_text_collapses_whitespace_and_lowercases() -> None:
    assert normalize_message_text("  FREE   Nitro\nClaim NOW  ") == "free nitro claim now"


def test_normalize_message_text_handles_discord_scam_text_variants() -> None:
    assert (
        normalize_message_text("Hello@everyone I’m giving away a Mac Book Air first-come, first-served")
        == "hello @everyone im giving away a macbook air first come first served"
    )
    assert (
        normalize_message_text('@everyone"Just upgraded — DM me if you’re interested')
        == "@everyone just upgraded dm me if youre interested"
    )


def test_normalize_message_text_ignores_emojis() -> None:
    assert (
        normalize_message_text("🎉 Hello 👋 @everyone 🎁 Mac Book Air 💻 first-come, first-served 🚨")
        == "hello @everyone macbook air first come first served"
    )


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
    assert "mass_mention" in result.reasons


def test_cheap_screening_detects_giveaway_scam_phrase_signals() -> None:
    result = cheap_trigger_screen(
        MessageContext(
            text=(
                "Hello@everyone I want to give out my Mac Book Air 2021 for free. "
                "Strictly first-come, first-served! Text me on WhatsApp [PHONE]."
            ),
            author_id=1,
        )
    )

    assert result.triggered
    assert "mass_mention" in result.reasons
    assert "high_value_item" in result.reasons
    assert "giveaway_language" in result.reasons
    assert "free_offer" in result.reasons
    assert "urgency_phrase" in result.reasons
    assert "whatsapp_or_phone_contact" in result.reasons
