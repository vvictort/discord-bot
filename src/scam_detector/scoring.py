from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.scam_detector.models import MessageContext
from src.scam_detector.preprocessing import normalize_message_text
from src.scam_detector.screening import SUSPICIOUS_KEYWORDS, detect_message_signals


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


MEDIUM_RULE_SCORE = 3
HIGH_RULE_SCORE = 8
CRITICAL_RULE_SCORE = 16


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

    signal_reasons = detect_message_signals(message)
    signal_set = set(signal_reasons)
    reasons.extend(signal for signal in signal_reasons if signal not in reasons)

    score += _score_signals(signal_set)

    keyword_suspicious_content = any(reason.startswith("keyword:") for reason in reasons)
    suspicious_content_exists = keyword_suspicious_content or bool(
        signal_set.intersection(
            {
                "giveaway_language",
                "free_offer",
                "dm_request",
                "high_value_giveaway_dm_pattern",
                "mass_mention_giveaway_pattern",
                "free_high_value_item_pattern",
            }
        )
    )

    if (
        suspicious_content_exists
        and message.member_join_age_seconds is not None
        and message.member_join_age_seconds < 3600
    ):
        score += 2
        reasons.append("new_member")

    if suspicious_content_exists and message.num_roles == 0:
        score += 1
        reasons.append("no_roles")

    return RuleScore(score=score, level=risk_level_for_score(score), reasons=reasons)


def risk_level_for_score(score: int) -> RiskLevel:
    if score >= CRITICAL_RULE_SCORE:
        return RiskLevel.CRITICAL
    if score >= HIGH_RULE_SCORE:
        return RiskLevel.HIGH
    if score >= MEDIUM_RULE_SCORE:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _score_signals(signals: set[str]) -> int:
    score = 0

    if "has_link" in signals:
        score += 1
    if "has_mention" in signals:
        score += 1
    if "mass_mention" in signals:
        score += 3

    # These are weak alone, but become strong through composite patterns below.
    if "high_value_item" in signals:
        score += 1
    if "giveaway_language" in signals:
        score += 1
    if "free_offer" in signals:
        score += 1
    if "dm_request" in signals:
        score += 1
    if "urgency_phrase" in signals:
        score += 1
    if "friend_request_or_external_contact" in signals:
        score += 1
    if "whatsapp_or_phone_contact" in signals:
        score += 2

    has_high_value_giveaway = "high_value_item" in signals and (
        "giveaway_language" in signals or "free_offer" in signals
    )

    if "high_value_item" in signals and "giveaway_language" in signals:
        score += 5
    if "high_value_giveaway_dm_pattern" in signals:
        score += 4
    if "mass_mention_giveaway_pattern" in signals:
        score += 3
    if "free_high_value_item_pattern" in signals:
        score += 3
    if has_high_value_giveaway and "urgency_phrase" in signals:
        score += 2
    if has_high_value_giveaway and "whatsapp_or_phone_contact" in signals:
        score += 3
    if has_high_value_giveaway and "emotional_need_framing" in signals:
        score += 2
    if has_high_value_giveaway and "recently_upgraded_framing" in signals:
        score += 2
    if has_high_value_giveaway and "condition_framing" in signals:
        score += 1

    return score
