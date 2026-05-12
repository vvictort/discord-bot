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
