from __future__ import annotations

import re

from src.scam_detector.models import MessageContext, ScreeningResult
from src.scam_detector.preprocessing import normalize_message_text

SUSPICIOUS_KEYWORDS = (
    "free nitro",
    "claim",
    "airdrop",
    "giveaway",
    "steam gift",
    "verify wallet",
    "limited time",
    "discord gift",
)

GIVEAWAY_PHRASES = (
    "giving away",
    "giving out",
    "give out",
    "give it out",
    "give away",
    "pass on",
    "pass this on",
    "pass this one on",
    "letting go",
    "not using anymore",
    "free to a good home",
)

FREE_OFFER_PHRASES = (
    "for free",
    "100% free",
    "100 free",
    "no tricks no catch",
    "free to a good home",
)

DM_CONTACT_PHRASES = (
    "dm me",
    "dm if interested",
    "dm me if interested",
    "dm if you are interested",
    "dm if youre interested",
    "message me",
    "send me a dm",
    "send me a request",
    "send a request",
    "friend request",
    "if interested",
    "if you are interested",
    "if youre interested",
    "please respond via dm",
    "reach out first",
    "text me on whatsapp",
    "whatsapp number",
)

URGENCY_PHRASES = (
    "first come first served",
    "first come first serve",
    "strictly first come",
    "reach out first",
    "asap",
)

HIGH_VALUE_ITEM_TERMS = (
    "macbook",
    "macbook air",
    "macbook pro",
    "laptop",
    "ps5",
    "playstation",
    "xbox",
    "console",
    "camera",
    "canon camera",
    "canon powershot",
    "sony camera",
    "sony aziv",
    "smallrig",
    "sigma lens",
    "iphone",
    "ipad",
    "gaming pc",
    "gpu",
    "smart tv",
    "e bike",
)

RECENTLY_UPGRADED_PHRASES = (
    "recently upgraded",
    "just upgraded",
    "upgraded my gear",
    "got a new model",
    "recently got a new model",
    "just got a new model",
    "new gadgets",
    "got new gadgets",
    "just got new gadgets",
)

EMOTIONAL_NEED_PHRASES = (
    "cant afford",
    "in need of it",
    "someone who needs it",
    "someone who might need it",
    "someone wholl actually use it",
    "someone wholl enjoy it",
    "truly use it",
)

CONDITION_FRAMING_PHRASES = (
    "excellent condition",
    "fully functional",
    "like new",
    "perfect health",
    "brand new",
    "good as new",
    "well maintained",
    "well kept",
    "solid condition",
)

_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")


def _contains_any(normalized: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in normalized for phrase in phrases)


def _contains_phone_or_whatsapp(raw_text: str | None, normalized: str) -> bool:
    raw = raw_text or ""
    return bool(
        "[phone]" in raw.lower()
        or "whatsapp" in normalized
        or _PHONE_RE.search(raw)
    )


def detect_message_signals(message: MessageContext) -> list[str]:
    normalized = normalize_message_text(message.text)
    raw_lower = (message.text or "").lower()
    signals: list[str] = []

    def add(signal: str) -> None:
        if signal not in signals:
            signals.append(signal)

    if message.has_link or "http://" in raw_lower or "https://" in raw_lower:
        add("has_link")

    has_mass_mention = "@everyone" in normalized or "@here" in normalized
    if has_mass_mention:
        add("mass_mention")
    elif message.has_mention:
        add("has_mention")

    has_high_value_item = _contains_any(normalized, HIGH_VALUE_ITEM_TERMS)
    has_giveaway_language = _contains_any(normalized, GIVEAWAY_PHRASES)
    has_free_offer = _contains_any(normalized, FREE_OFFER_PHRASES)
    has_dm_request = _contains_any(normalized, DM_CONTACT_PHRASES)
    has_urgency = _contains_any(normalized, URGENCY_PHRASES)
    has_whatsapp_or_phone = _contains_phone_or_whatsapp(message.text, normalized)
    has_friend_or_external = (
        "friend request" in normalized
        or "whatsapp" in normalized
        or has_whatsapp_or_phone
    )
    has_emotional_need = _contains_any(normalized, EMOTIONAL_NEED_PHRASES)
    has_recently_upgraded = _contains_any(normalized, RECENTLY_UPGRADED_PHRASES)
    has_condition_framing = _contains_any(normalized, CONDITION_FRAMING_PHRASES)

    if has_high_value_item:
        add("high_value_item")
    if has_giveaway_language:
        add("giveaway_language")
    if has_free_offer:
        add("free_offer")
    if has_dm_request:
        add("dm_request")
    if has_urgency:
        add("urgency_phrase")
    if has_friend_or_external:
        add("friend_request_or_external_contact")
    if has_whatsapp_or_phone:
        add("whatsapp_or_phone_contact")
    if has_emotional_need:
        add("emotional_need_framing")
    if has_recently_upgraded:
        add("recently_upgraded_framing")
    if has_condition_framing:
        add("condition_framing")

    if has_high_value_item and has_giveaway_language and has_dm_request:
        add("high_value_giveaway_dm_pattern")
    if has_mass_mention and has_giveaway_language:
        add("mass_mention_giveaway_pattern")
    if has_high_value_item and has_free_offer:
        add("free_high_value_item_pattern")

    return signals


def cheap_trigger_screen(message: MessageContext) -> ScreeningResult:
    normalized = normalize_message_text(message.text)
    reasons: list[str] = []

    for keyword in SUSPICIOUS_KEYWORDS:
        if keyword in normalized:
            reasons.append(f"keyword:{keyword}")

    for signal in detect_message_signals(message):
        if signal not in reasons:
            reasons.append(signal)

    return ScreeningResult(triggered=bool(reasons), reasons=reasons)
