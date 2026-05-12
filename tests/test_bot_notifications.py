from src.scam_detector.bot import BotSettings, format_detection_summary, load_bot_settings_from_env
from src.scam_detector.decisions import Decision, DecisionResult
from src.scam_detector.models import ScreeningResult
from src.scam_detector.pipeline import DetectionResult
from src.scam_detector.scoring import RiskLevel, RuleScore


def test_load_bot_settings_from_env_parses_optional_review_channel() -> None:
    settings = load_bot_settings_from_env(
        {
            "MOD_REVIEW_CHANNEL_ID": "123456789",
            "BOT_DELETE_ENABLED": "false",
            "BOT_NOTIFY_LOG_ACTIONS": "true",
            "WHITELISTED_ROLE_IDS": "111, 222",
        }
    )

    assert settings == BotSettings(
        mod_review_channel_id=123456789,
        delete_enabled=False,
        notify_log_actions=True,
        whitelisted_role_ids=frozenset({111, 222}),
    )


def test_format_detection_summary_includes_action_reason_and_score() -> None:
    result = DetectionResult(
        eligible=True,
        screening=ScreeningResult(triggered=True, reasons=["keyword:free nitro"]),
        rule_score=RuleScore(score=4, level=RiskLevel.MEDIUM, reasons=["keyword:free nitro"]),
        classifier_probability=0.82,
        classifier_called=True,
        decision=DecisionResult(Decision.REVIEW, "classifier_mod_review_threshold"),
    )

    summary = format_detection_summary(
        author_id=1,
        channel_id=2,
        content="free nitro claim",
        result=result,
    )

    assert "Action: review" in summary
    assert "Reason: classifier_mod_review_threshold" in summary
    assert "Rule score: 4" in summary
    assert "Classifier probability: 0.820" in summary
