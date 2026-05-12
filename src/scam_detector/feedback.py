from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class FeedbackRecord:
    id: int
    message_id: int
    guild_id: int
    channel_id: int
    moderator_id: int
    text: str
    label: int
    reason: str | None
    source: str
    created_at: str


class FeedbackRepository:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS moderator_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id INTEGER NOT NULL,
                    guild_id INTEGER NOT NULL,
                    channel_id INTEGER NOT NULL,
                    moderator_id INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    label INTEGER NOT NULL CHECK (label IN (0, 1)),
                    reason TEXT,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_moderator_feedback_message
                ON moderator_feedback(message_id)
                """
            )

    def add_confirmed_label(
        self,
        message_id: int,
        guild_id: int,
        channel_id: int,
        moderator_id: int,
        text: str,
        label: int,
        reason: str | None = None,
    ) -> None:
        if label not in (0, 1):
            raise ValueError("label must be 0 for not scam or 1 for scam/phishing")
        if not text.strip():
            raise ValueError("text is required for confirmed labels")

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO moderator_feedback (
                    message_id, guild_id, channel_id, moderator_id, text, label, reason, source, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    guild_id,
                    channel_id,
                    moderator_id,
                    text,
                    label,
                    reason,
                    "moderator_confirmed",
                    datetime.now(UTC).isoformat(),
                ),
            )

    def list_records(self) -> list[FeedbackRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, message_id, guild_id, channel_id, moderator_id, text, label, reason, source, created_at
                FROM moderator_feedback
                ORDER BY id ASC
                """
            ).fetchall()
        return [FeedbackRecord(**dict(row)) for row in rows]

    def to_training_frame(self) -> pd.DataFrame:
        records = self.list_records()
        return pd.DataFrame([{"text": record.text, "label": record.label} for record in records])

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection
