from src.scam_detector.decisions import Decision, DecisionThresholds, decide_action
from src.scam_detector.models import MessageContext
from src.scam_detector.scoring import RiskLevel, score_message


def test_rule_scoring_rates_plain_message_low() -> None:
    score = score_message(MessageContext(text="regular project update", author_id=1))

    assert score.level == RiskLevel.LOW
    assert score.score == 0


def test_rule_scoring_uses_content_metadata_and_behavior() -> None:
    score = score_message(
        MessageContext(
            text="@everyone free nitro claim https://example.test",
            author_id=1,
            has_link=True,
            has_mention=True,
            member_join_age_seconds=60,
            num_roles=0,
        )
    )

    assert score.level == RiskLevel.HIGH
    assert score.score >= 8
    assert "keyword:free nitro" in score.reasons
    assert "new_member" in score.reasons
    assert "no_roles" in score.reasons


def test_decision_allows_low_risk_messages() -> None:
    decision = decide_action(rule_score=0, classifier_probability=None)

    assert decision.action == Decision.ALLOW


def test_decision_logs_medium_rule_score_without_classifier() -> None:
    decision = decide_action(rule_score=3, classifier_probability=None)

    assert decision.action == Decision.LOG


def test_decision_flags_review_for_mod_review_probability_band() -> None:
    thresholds = DecisionThresholds(auto_delete=0.90, mod_review=0.75, log_only=0.55)
    decision = decide_action(rule_score=6, classifier_probability=0.80, thresholds=thresholds)

    assert decision.action == Decision.REVIEW


def test_decision_deletes_only_at_auto_delete_threshold() -> None:
    thresholds = DecisionThresholds(auto_delete=0.90, mod_review=0.75, log_only=0.55)
    decision = decide_action(rule_score=6, classifier_probability=0.95, thresholds=thresholds)

    assert decision.action == Decision.DELETE
