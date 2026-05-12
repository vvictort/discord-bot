from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.scam_detector.models import MessageContext
from src.scam_detector.preprocessing import normalize_message_text
from src.scam_detector.screening import SUSPICIOUS_KEYWORDS


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class RuleScore:
    score: int
    level: RiskLevel
    reasons: list[str]


def score_message(message: MessageContext) -> RuleScore:
    normalized = normalize_message_text(message.text)
    score = 0
    reasons: list[str] = []

    for keyword in SUSPICIOUS_KEYWORDS:
        if keyword in normalized:
            score += 2
            reasons.append(f"keyword:{keyword}")

    if message.has_link or "http://" in normalized or "https://" in normalized:
        score += 2
        reasons.append("has_link")

    if message.has_mention or "@everyone" in normalized or "@here" in normalized:
        score += 1
        reasons.append("has_mention")

    if message.member_join_age_seconds is not None and message.member_join_age_seconds < 3600:
        score += 2
        reasons.append("new_member")

    if message.num_roles == 0:
        score += 1
        reasons.append("no_roles")

    if score >= 7:
        level = RiskLevel.HIGH
    elif score >= 3:
        level = RiskLevel.MEDIUM
    else:
        level = RiskLevel.LOW

    return RuleScore(score=score, level=level, reasons=reasons)
