from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.detection.embedding_similarity import EmbeddingSimilarityResult
from src.scam_detector.decisions import DecisionResult, decide_action
from src.scam_detector.models import MessageContext, ScreeningResult
from src.scam_detector.preprocessing import is_eligible_message
from src.scam_detector.scoring import RiskLevel, RuleScore, risk_level_for_score, score_message
from src.scam_detector.screening import cheap_trigger_screen


class ClassifierProtocol(Protocol):
    def predict_probability(self, message: MessageContext) -> float | None:
        ...


class EmbeddingSimilarityProtocol(Protocol):
    """Interface for optional semantic-template matching."""

    def compare(self, message_text: str | None) -> EmbeddingSimilarityResult:
        ...


@dataclass(frozen=True)
class DetectionResult:
    """Full detector trace used for moderation logs and tests."""

    eligible: bool
    screening: ScreeningResult
    rule_score: RuleScore | None
    classifier_probability: float | None
    classifier_called: bool
    decision: DecisionResult
    classifier_skip_reason: str | None = None
    embedding_called: bool = False
    embedding_similarity: float | None = None
    embedding_matched_category: str | None = None
    embedding_skip_reason: str | None = None


class DetectionPipeline:
    """Cheap-first detection orchestrator.

    The ordering matters: rules run before optional expensive layers, and
    high/critical rule scores skip embeddings/classifier entirely.
    """

    def __init__(
        self,
        classifier: ClassifierProtocol | None = None,
        embedding_similarity: EmbeddingSimilarityProtocol | None = None,
        whitelisted_role_ids: set[int] | frozenset[int] | None = None,
    ) -> None:
        self.classifier = classifier
        self.embedding_similarity = embedding_similarity
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
                classifier_skip_reason="ineligible_message",
                embedding_skip_reason="ineligible_message",
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
                classifier_skip_reason="whitelisted_role",
                embedding_skip_reason="whitelisted_role",
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
                classifier_skip_reason="screening_not_triggered",
                embedding_skip_reason="screening_not_triggered",
            )

        rule_score = score_message(message)

        # Embeddings are only a helper for uncertain suspicious messages. Obvious
        # rule matches skip this layer so a low or unavailable semantic score
        # cannot weaken deterministic rule evidence.
        embedding_called = False
        embedding_similarity_score = None
        embedding_matched_category = None
        embedding_skip_reason = None

        if rule_score.level in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
            embedding_skip_reason = f"{rule_score.level.value}_rule_score"
        elif self.embedding_similarity is not None:
            embedding_called = True
            embedding_result = self.embedding_similarity.compare(message.text)
            embedding_similarity_score = embedding_result.max_similarity
            embedding_matched_category = embedding_result.matched_category
            if embedding_result.available:
                rule_score = _apply_embedding_similarity(rule_score, embedding_result)
            else:
                embedding_skip_reason = "embedding_similarity_unavailable"
        else:
            embedding_skip_reason = "embedding_similarity_disabled"

        classifier_probability = None
        classifier_called = False
        classifier_skip_reason = None
        # Classifier is last: it can help medium cases, but it must not override
        # high-confidence rule/template evidence.
        if rule_score.level in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
            classifier_skip_reason = f"{rule_score.level.value}_rule_score"
        elif self.classifier is not None and rule_score.level == RiskLevel.MEDIUM:
            classifier_called = True
            classifier_probability = self.classifier.predict_probability(message)
        elif self.classifier is None:
            classifier_skip_reason = "classifier_unavailable"

        return DetectionResult(
            eligible=True,
            screening=screening,
            rule_score=rule_score,
            classifier_probability=classifier_probability,
            classifier_called=classifier_called,
            decision=decide_action(rule_score=rule_score.score, classifier_probability=classifier_probability),
            classifier_skip_reason=classifier_skip_reason,
            embedding_called=embedding_called,
            embedding_similarity=embedding_similarity_score,
            embedding_matched_category=embedding_matched_category,
            embedding_skip_reason=embedding_skip_reason,
        )

    def _has_whitelisted_role(
        self,
        message: MessageContext,
        whitelisted_role_ids: frozenset[int],
    ) -> bool:
        if not whitelisted_role_ids:
            return False
        return bool(whitelisted_role_ids.intersection(message.author_role_ids))


def _apply_embedding_similarity(
    rule_score: RuleScore,
    embedding_result: EmbeddingSimilarityResult,
) -> RuleScore:
    if not embedding_result.reasons:
        return rule_score

    boost = 0
    if "highly_similar_to_known_scam_template" in embedding_result.reasons:
        boost = 5
    elif "similar_to_known_scam_template" in embedding_result.reasons:
        boost = 3

    if boost == 0:
        return rule_score

    reasons = list(rule_score.reasons)
    for reason in embedding_result.reasons:
        if reason not in reasons:
            reasons.append(reason)

    score = rule_score.score + boost
    return RuleScore(
        score=score,
        level=risk_level_for_score(score),
        reasons=reasons,
    )
