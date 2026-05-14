import pytest

from src.scam_detector.bot import (
    BotSettings,
    ScamDetectionBot,
    build_moderation_log_payload,
    build_default_guild_config,
    format_detection_summary,
    load_bot_settings_from_env,
)
from src.scam_detector.decisions import ActionBand, Decision, DecisionResult
from src.scam_detector.feedback import FeedbackRepository
from src.scam_detector.guild_config import GuildConfig
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
            "COMMAND_SYNC_GUILD_ID": "333",
            "EMBEDDING_SIMILARITY_ENABLED": "true",
            "SCAM_TEMPLATE_PATH": "templates/scams.json",
            "AUTO_DELETE_CRITICAL": "true",
            "AUTO_DELETE_HIGH": "true",
            "CRITICAL_RULE_SCORE_THRESHOLD": "18",
            "HIGH_RULE_SCORE_THRESHOLD": "9",
            "MOD_REVIEW_THRESHOLD": "0.8",
            "FEEDBACK_DB_PATH": "data/test-feedback.sqlite",
        }
    )

    assert settings == BotSettings(
        mod_review_channel_id=123456789,
        delete_enabled=False,
        notify_log_actions=True,
        whitelisted_role_ids=frozenset({111, 222}),
        command_sync_guild_id=333,
        embedding_similarity_enabled=True,
        scam_template_path="templates/scams.json",
        auto_delete_critical=True,
        auto_delete_high=True,
        critical_rule_score_threshold=18,
        high_rule_score_threshold=9,
        mod_review_threshold=0.8,
        feedback_database_path="data/test-feedback.sqlite",
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
    assert "Final score: 4" in summary
    assert "Classifier probability: 0.820" in summary


def test_build_default_guild_config_from_env_settings() -> None:
    config = build_default_guild_config(
        BotSettings(
            mod_review_channel_id=1,
            delete_enabled=False,
            notify_log_actions=False,
            whitelisted_role_ids=frozenset({2}),
        )
    )

    assert config.mod_review_channel_id == 1
    assert not config.delete_enabled
    assert not config.notify_log_actions
    assert config.whitelisted_role_ids == frozenset({2})


def test_bot_registers_core_scam_command_group() -> None:
    bot = ScamDetectionBot(settings=BotSettings())

    commands = {command.name: command for command in bot.tree.get_commands()}

    assert "scam" in commands
    assert "scam-config" not in commands
    subcommands = {command.name for command in commands["scam"].commands}
    assert subcommands == {"setup", "mode", "status", "trust", "untrust"}


class FakeAuthor:
    id = 10
    display_name = "Hav"


class FakeGuild:
    id = 20


class FakeChannel:
    id = 30
    mention = "<#30>"


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.id = 123
        self.content = content
        self.author = FakeAuthor()
        self.guild = FakeGuild()
        self.channel = FakeChannel()
        self.deleted = False
        self.jump_url = "https://discord.com/channels/20/30/123"

    async def delete(self) -> None:
        self.deleted = True


class FakeModChannel:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []

    async def send(self, *args: object, **kwargs: object) -> None:
        self.messages.append({"args": args, "kwargs": kwargs})


def test_build_moderation_log_payload_contains_compact_automod_style_alert() -> None:
    message = FakeMessage("Hello @everyone\nFree PS5 giveaway, DM me if interested.")
    result = DetectionResult(
        eligible=True,
        screening=ScreeningResult(triggered=True, reasons=["mass_mention"]),
        rule_score=RuleScore(
            score=20,
            level=RiskLevel.CRITICAL,
            reasons=["mass_mention", "high_value_item", "dm_request"],
        ),
        classifier_probability=0.94,
        classifier_called=True,
        decision=DecisionResult(Decision.DELETE, "classifier_auto_delete_threshold", ActionBand.HIGH),
    )

    payload = build_moderation_log_payload(message, result, action_taken="deleted")

    # Content is a single-line header like AutoMod.
    assert payload.content == "**Scam Bot** has blocked a message in <#30>"

    # Embed has no title (compact style).
    assert payload.embed.title is None

    # Embed has no separate fields — everything is in the description.
    assert len(payload.embed.fields) == 0

    # Description includes the author, quoted message, and metadata line.
    assert "**Hav**" in payload.embed.description
    assert "Free PS5 giveaway" in payload.embed.description
    assert "**Signals:** Mass Mention, High Value Item, DM Request" in payload.embed.description
    assert "**Rule:** Classifier Auto Delete Threshold" in payload.embed.description
    assert "**Action:** Blocked" in payload.embed.description

    assert payload.view is not None


@pytest.mark.asyncio
async def test_critical_detection_deletes_logs_and_stores_pending_candidate(tmp_path) -> None:
    repository = FeedbackRepository(tmp_path / "feedback.sqlite")
    repository.initialize()
    bot = ScamDetectionBot(
        settings=BotSettings(),
        feedback_repository=repository,
    )
    mod_channel = FakeModChannel()
    bot.get_channel = lambda channel_id: mod_channel
    message = FakeMessage("@everyone giving away my MacBook Air for free. DM me.")
    result = DetectionResult(
        eligible=True,
        screening=ScreeningResult(triggered=True, reasons=["mass_mention"]),
        rule_score=RuleScore(score=20, level=RiskLevel.CRITICAL, reasons=["mass_mention"]),
        classifier_probability=None,
        classifier_called=False,
        decision=DecisionResult(Decision.DELETE, "critical_rule_score_auto_delete", ActionBand.CRITICAL),
    )

    await bot._handle_detected_message(
        message=message,
        result=result,
        guild_config=GuildConfig(mod_review_channel_id=999, delete_enabled=True),
    )

    assert message.deleted
    assert mod_channel.messages
    log_kwargs = mod_channel.messages[0]["kwargs"]
    assert log_kwargs["content"] == "**Scam Bot** has blocked a message in <#30>"
    record = repository.list_records()[0]
    assert record.label is None
    assert record.label_source == "bot_flag"
    assert record.review_status == "pending"
    assert record.needs_review
    assert record.action_taken == "deleted"
