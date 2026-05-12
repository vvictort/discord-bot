from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.scam_detector.decisions import DecisionResult, decide_action
from src.scam_detector.models import MessageContext, ScreeningResult
from src.scam_detector.preprocessing import is_eligible_message
from src.scam_detector.scoring import RiskLevel, RuleScore, score_message
from src.scam_detector.screening import cheap_trigger_screen


class ClassifierProtocol(Protocol):
    def predict_probability(self, message: MessageContext) -> float | None:
        ...


@dataclass(frozen=True)
class DetectionResult:
    eligible: bool
    screening: ScreeningResult
    rule_score: RuleScore | None
    classifier_probability: float | None
    classifier_called: bool
    decision: DecisionResult


class DetectionPipeline:
    def __init__(
        self,
        classifier: ClassifierProtocol | None = None,
        whitelisted_role_ids: set[int] | frozenset[int] | None = None,
    ) -> None:
        self.classifier = classifier
        self.whitelisted_role_ids = frozenset(whitelisted_role_ids or set())

    def detect(
        self,
        message: MessageContext,
        whitelisted_role_ids: set[int] | frozenset[int] | None = None,
    ) -> DetectionResult:
        if not is_eligible_message(message):
            return DetectionResult(
                eligible=False,
                screening=ScreeningResult(triggered=False, reasons=[]),
                rule_score=None,
                classifier_probability=None,
                classifier_called=False,
                decision=decide_action(rule_score=0, classifier_probability=None),
            )

        active_whitelist = (
            frozenset(whitelisted_role_ids)
            if whitelisted_role_ids is not None
            else self.whitelisted_role_ids
        )
        if self._has_whitelisted_role(message, active_whitelist):
            return DetectionResult(
                eligible=True,
                screening=ScreeningResult(triggered=False, reasons=["whitelisted_role"]),
                rule_score=None,
                classifier_probability=None,
                classifier_called=False,
                decision=decide_action(rule_score=0, classifier_probability=None),
            )

        screening = cheap_trigger_screen(message)
        if not screening.triggered:
            return DetectionResult(
                eligible=True,
                screening=screening,
                rule_score=None,
                classifier_probability=None,
                classifier_called=False,
                decision=decide_action(rule_score=0, classifier_probability=None),
            )

        rule_score = score_message(message)
        classifier_probability = None
        classifier_called = False
        if self.classifier is not None and rule_score.level in {RiskLevel.MEDIUM, RiskLevel.HIGH}:
            classifier_called = True
            classifier_probability = self.classifier.predict_probability(message)

        return DetectionResult(
            eligible=True,
            screening=screening,
            rule_score=rule_score,
            classifier_probability=classifier_probability,
            classifier_called=classifier_called,
            decision=decide_action(rule_score=rule_score.score, classifier_probability=classifier_probability),
        )

    def _has_whitelisted_role(
        self,
        message: MessageContext,
        whitelisted_role_ids: frozenset[int],
    ) -> bool:
        if not whitelisted_role_ids:
            return False
        return bool(whitelisted_role_ids.intersection(message.author_role_ids))
