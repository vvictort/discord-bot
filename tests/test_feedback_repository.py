import pandas as pd
import pytest

from src.scam_detector.feedback import FeedbackRepository


def test_feedback_repository_stores_confirmed_labels(tmp_path) -> None:
    repository = FeedbackRepository(tmp_path / "feedback.sqlite")
    repository.initialize()

    repository.add_confirmed_label(
        message_id=123,
        guild_id=456,
        channel_id=789,
        moderator_id=42,
        text="free nitro claim",
        label=1,
        reason="confirmed scam",
    )

    records = repository.list_records()

    assert len(records) == 1
    assert records[0].label == 1
    assert records[0].source == "moderator_confirmed"
    assert records[0].review_status == "confirmed"
    assert not records[0].needs_review


def test_feedback_repository_stores_pending_bot_candidate_without_label(tmp_path) -> None:
    repository = FeedbackRepository(tmp_path / "feedback.sqlite")
    repository.initialize()

    repository.add_pending_candidate(
        message_id=123,
        guild_id=456,
        channel_id=789,
        text="@everyone giving away a MacBook",
        reason="critical_rule_score",
        action_taken="deleted",
    )

    record = repository.list_records()[0]
    assert record.label is None
    assert record.label_source == "bot_flag"
    assert record.review_status == "pending"
    assert record.needs_review
    assert record.action_taken == "deleted"


def test_moderator_confirmation_required_before_export_to_training_data(tmp_path) -> None:
    repository = FeedbackRepository(tmp_path / "feedback.sqlite")
    repository.initialize()
    repository.add_pending_candidate(
        message_id=1,
        guild_id=1,
        channel_id=1,
        text="pending bot flag",
        reason="critical_rule_score",
        action_taken="deleted",
    )

    assert repository.to_training_frame().empty

    repository.confirm_label(
        message_id=1,
        moderator_id=99,
        label=1,
        reason="confirmed scam",
    )

    frame = repository.to_training_frame()
    assert list(frame["text"]) == ["pending bot flag"]
    assert list(frame["label"]) == [1]


def test_moderator_can_confirm_false_positive(tmp_path) -> None:
    repository = FeedbackRepository(tmp_path / "feedback.sqlite")
    repository.initialize()
    repository.add_pending_candidate(
        message_id=1,
        guild_id=1,
        channel_id=1,
        text="not actually a scam",
        reason="high_rule_score_review",
        action_taken="review",
    )

    repository.confirm_label(
        message_id=1,
        moderator_id=99,
        label=0,
        reason="false positive",
    )

    record = repository.list_records()[0]
    assert record.label == 0
    assert record.label_source == "moderator_confirmed"
    assert record.review_status == "confirmed"
    assert not record.needs_review


def test_moderator_can_ignore_pending_candidate(tmp_path) -> None:
    repository = FeedbackRepository(tmp_path / "feedback.sqlite")
    repository.initialize()
    repository.add_pending_candidate(
        message_id=1,
        guild_id=1,
        channel_id=1,
        text="ignore this candidate",
        reason="medium_rule_score_review",
        action_taken="review",
    )

    repository.ignore_candidate(message_id=1, moderator_id=99, reason="not relevant")

    record = repository.list_records()[0]
    assert record.label is None
    assert record.review_status == "ignored"
    assert not record.needs_review
    assert repository.to_training_frame().empty


def test_user_report_does_not_create_training_label(tmp_path) -> None:
    repository = FeedbackRepository(tmp_path / "feedback.sqlite")
    repository.initialize()

    repository.add_user_report(
        message_id=123,
        guild_id=456,
        channel_id=789,
        reporter_id=42,
        text="normal message someone reported",
        reason="user report",
    )

    record = repository.list_records()[0]
    assert record.label is None
    assert record.label_source == "user_report"
    assert record.review_status == "pending"
    assert record.needs_review
    assert record.report_count == 1
    assert record.review_priority == 1
    assert repository.to_training_frame().empty


def test_feedback_repository_rejects_non_binary_labels(tmp_path) -> None:
    repository = FeedbackRepository(tmp_path / "feedback.sqlite")
    repository.initialize()

    with pytest.raises(ValueError):
        repository.add_confirmed_label(
            message_id=123,
            guild_id=456,
            channel_id=789,
            moderator_id=42,
            text="message",
            label=2,
        )


def test_feedback_repository_exports_training_frame(tmp_path) -> None:
    repository = FeedbackRepository(tmp_path / "feedback.sqlite")
    repository.initialize()
    repository.add_confirmed_label(
        message_id=1,
        guild_id=1,
        channel_id=1,
        moderator_id=1,
        text="safe message",
        label=0,
    )
    repository.add_confirmed_label(
        message_id=2,
        guild_id=1,
        channel_id=1,
        moderator_id=1,
        text="scam message",
        label=1,
    )

    frame = repository.to_training_frame()

    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == ["text", "label"]
    assert list(frame["label"]) == [0, 1]
