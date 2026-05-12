from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.scam_detector.scoring import CRITICAL_RULE_SCORE, HIGH_RULE_SCORE, MEDIUM_RULE_SCORE


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


@dataclass(frozen=True)
class DecisionResult:
    action: Decision
    reason: str


def decide_action(
    rule_score: int,
    classifier_probability: float | None,
    thresholds: DecisionThresholds | None = None,
) -> DecisionResult:
    thresholds = thresholds or DecisionThresholds()

    if rule_score >= CRITICAL_RULE_SCORE:
        return DecisionResult(Decision.REVIEW, "critical_rule_score")

    if classifier_probability is not None:
        if classifier_probability >= thresholds.auto_delete:
            return DecisionResult(Decision.DELETE, "classifier_auto_delete_threshold")
        if classifier_probability >= thresholds.mod_review:
            return DecisionResult(Decision.REVIEW, "classifier_mod_review_threshold")
        if classifier_probability >= thresholds.log_only:
            return DecisionResult(Decision.LOG, "classifier_log_only_threshold")

    if rule_score >= HIGH_RULE_SCORE:
        return DecisionResult(Decision.REVIEW, "high_rule_score")
    if rule_score >= MEDIUM_RULE_SCORE:
        return DecisionResult(Decision.LOG, "medium_rule_score")
    return DecisionResult(Decision.ALLOW, "low_rule_score")
