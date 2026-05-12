from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.scam_detector.scoring import CRITICAL_RULE_SCORE, HIGH_RULE_SCORE, MEDIUM_RULE_SCORE


class ActionBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Decision(str, Enum):
    ALLOW = "allow"
    LOG = "log"
    REVIEW = "review"
    DELETE = "delete"


@dataclass(frozen=True)
class DecisionThresholds:
    auto_delete: float = 0.90
    mod_review: float = 0.75
    log_only: float = 0.55
    auto_delete_critical: bool = True
    auto_delete_high: bool = False
    critical_rule_score_threshold: int = CRITICAL_RULE_SCORE
    high_rule_score_threshold: int = HIGH_RULE_SCORE
    medium_rule_score_threshold: int = MEDIUM_RULE_SCORE


@dataclass(frozen=True)
class DecisionResult:
    action: Decision
    reason: str
    band: ActionBand = ActionBand.LOW


def decide_action(
    rule_score: int,
    classifier_probability: float | None,
    thresholds: DecisionThresholds | None = None,
    user_report_count: int = 0,
) -> DecisionResult:
    thresholds = thresholds or DecisionThresholds()

    if rule_score >= thresholds.critical_rule_score_threshold:
        if thresholds.auto_delete_critical:
            return DecisionResult(
                Decision.DELETE,
                "critical_rule_score_auto_delete",
                ActionBand.CRITICAL,
            )
        return DecisionResult(Decision.REVIEW, "critical_rule_score_review", ActionBand.CRITICAL)

    if rule_score >= thresholds.high_rule_score_threshold:
        if thresholds.auto_delete_high:
            return DecisionResult(Decision.DELETE, "high_rule_score_auto_delete", ActionBand.HIGH)
        return DecisionResult(Decision.REVIEW, "high_rule_score_review", ActionBand.HIGH)

    if classifier_probability is not None:
        if classifier_probability >= thresholds.auto_delete:
            return DecisionResult(Decision.DELETE, "classifier_auto_delete_threshold", ActionBand.HIGH)
        if classifier_probability >= thresholds.mod_review:
            return DecisionResult(Decision.REVIEW, "classifier_mod_review_threshold", ActionBand.MEDIUM)
        if classifier_probability >= thresholds.log_only:
            return DecisionResult(Decision.LOG, "classifier_log_only_threshold", ActionBand.LOW)

    if rule_score >= thresholds.medium_rule_score_threshold:
        return DecisionResult(Decision.REVIEW, "medium_rule_score_review", ActionBand.MEDIUM)

    if user_report_count > 0:
        return DecisionResult(Decision.REVIEW, "user_report_review_priority", ActionBand.LOW)

    return DecisionResult(Decision.ALLOW, "low_rule_score", ActionBand.LOW)
