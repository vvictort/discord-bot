from __future__ import annotations

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


def cheap_trigger_screen(message: MessageContext) -> ScreeningResult:
    normalized = normalize_message_text(message.text)
    reasons: list[str] = []

    for keyword in SUSPICIOUS_KEYWORDS:
        if keyword in normalized:
            reasons.append(f"keyword:{keyword}")

    if message.has_link or "http://" in normalized or "https://" in normalized:
        reasons.append("has_link")

    if message.has_mention or "@everyone" in normalized or "@here" in normalized:
        reasons.append("has_mention")

    return ScreeningResult(triggered=bool(reasons), reasons=reasons)
