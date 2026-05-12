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
    moderator_id: int | None
    text: str
    label: int | None
    reason: str | None
    label_source: str
    review_status: str
    needs_review: bool
    action_taken: str | None
    report_count: int
    review_priority: int
    created_at: str
    updated_at: str

    @property
    def source(self) -> str:
        return self.label_source


class FeedbackRepository:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            self._migrate_legacy_schema(connection)
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS moderator_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id INTEGER NOT NULL,
                    guild_id INTEGER NOT NULL,
                    channel_id INTEGER NOT NULL,
                    moderator_id INTEGER,
                    text TEXT NOT NULL,
                    label INTEGER CHECK (label IN (0, 1)),
                    reason TEXT,
                    label_source TEXT NOT NULL,
                    review_status TEXT NOT NULL,
                    needs_review INTEGER NOT NULL,
                    action_taken TEXT,
                    report_count INTEGER NOT NULL DEFAULT 0,
                    review_priority INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_moderator_feedback_message
                ON moderator_feedback(message_id)
                """
            )

    def add_pending_candidate(
        self,
        message_id: int,
        guild_id: int,
        channel_id: int,
        text: str,
        reason: str,
        action_taken: str,
    ) -> None:
        if not text.strip():
            raise ValueError("text is required for pending candidates")

        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO moderator_feedback (
                    message_id, guild_id, channel_id, moderator_id, text, label, reason,
                    label_source, review_status, needs_review, action_taken, report_count,
                    review_priority, created_at, updated_at
                )
                VALUES (?, ?, ?, NULL, ?, NULL, ?, ?, ?, ?, ?, 0, 1, ?, ?)
                """,
                (
                    message_id,
                    guild_id,
                    channel_id,
                    text,
                    reason,
                    "bot_flag",
                    "pending",
                    1,
                    action_taken,
                    now,
                    now,
                ),
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
                    message_id, guild_id, channel_id, moderator_id, text, label, reason,
                    label_source, review_status, needs_review, action_taken, report_count,
                    review_priority, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, 0, ?, ?)
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
                    "confirmed",
                    0,
                    datetime.now(UTC).isoformat(),
                    datetime.now(UTC).isoformat(),
                ),
            )

    def confirm_label(
        self,
        message_id: int,
        moderator_id: int,
        label: int,
        reason: str | None = None,
    ) -> None:
        if label not in (0, 1):
            raise ValueError("label must be 0 for not scam or 1 for scam/phishing")

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE moderator_feedback
                SET label = ?,
                    moderator_id = ?,
                    reason = COALESCE(?, reason),
                    label_source = 'moderator_confirmed',
                    review_status = 'confirmed',
                    needs_review = 0,
                    updated_at = ?
                WHERE message_id = ?
                """,
                (label, moderator_id, reason, datetime.now(UTC).isoformat(), message_id),
            )

    def ignore_candidate(
        self,
        message_id: int,
        moderator_id: int,
        reason: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE moderator_feedback
                SET moderator_id = ?,
                    reason = COALESCE(?, reason),
                    review_status = 'ignored',
                    needs_review = 0,
                    updated_at = ?
                WHERE message_id = ?
                """,
                (moderator_id, reason, datetime.now(UTC).isoformat(), message_id),
            )

    def add_user_report(
        self,
        message_id: int,
        guild_id: int,
        channel_id: int,
        reporter_id: int,
        text: str,
        reason: str | None = None,
    ) -> None:
        if not text.strip():
            raise ValueError("text is required for user reports")

        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT id, report_count, review_priority
                FROM moderator_feedback
                WHERE message_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (message_id,),
            ).fetchone()
            if existing:
                connection.execute(
                    """
                    UPDATE moderator_feedback
                    SET report_count = ?,
                        review_priority = ?,
                        needs_review = 1,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        int(existing["report_count"]) + 1,
                        int(existing["review_priority"]) + 1,
                        now,
                        existing["id"],
                    ),
                )
                return

            connection.execute(
                """
                INSERT INTO moderator_feedback (
                    message_id, guild_id, channel_id, moderator_id, text, label, reason,
                    label_source, review_status, needs_review, action_taken, report_count,
                    review_priority, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, 1, 1, ?, ?)
                """,
                (
                    message_id,
                    guild_id,
                    channel_id,
                    reporter_id,
                    text,
                    reason,
                    "user_report",
                    "pending",
                    1,
                    "reported",
                    now,
                    now,
                ),
            )

    def list_records(self) -> list[FeedbackRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, message_id, guild_id, channel_id, moderator_id, text, label, reason,
                       label_source, review_status, needs_review, action_taken, report_count,
                       review_priority, created_at, updated_at
                FROM moderator_feedback
                ORDER BY id ASC
                """
            ).fetchall()
        return [
            FeedbackRecord(
                **{
                    **dict(row),
                    "needs_review": bool(row["needs_review"]),
                }
            )
            for row in rows
        ]

    def to_training_frame(self) -> pd.DataFrame:
        records = [
            record
            for record in self.list_records()
            if record.label_source == "moderator_confirmed"
            and record.review_status == "confirmed"
            and record.label is not None
        ]
        return pd.DataFrame([{"text": record.text, "label": record.label} for record in records])

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _migrate_legacy_schema(self, connection: sqlite3.Connection) -> None:
        columns = connection.execute("PRAGMA table_info(moderator_feedback)").fetchall()
        if not columns:
            return

        column_names = {column["name"] for column in columns}
        if "label_source" in column_names:
            return

        connection.execute("ALTER TABLE moderator_feedback RENAME TO moderator_feedback_legacy")
        connection.execute(
            """
            CREATE TABLE moderator_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id INTEGER NOT NULL,
                guild_id INTEGER NOT NULL,
                channel_id INTEGER NOT NULL,
                moderator_id INTEGER,
                text TEXT NOT NULL,
                label INTEGER CHECK (label IN (0, 1)),
                reason TEXT,
                label_source TEXT NOT NULL,
                review_status TEXT NOT NULL,
                needs_review INTEGER NOT NULL,
                action_taken TEXT,
                report_count INTEGER NOT NULL DEFAULT 0,
                review_priority INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO moderator_feedback (
                id, message_id, guild_id, channel_id, moderator_id, text, label, reason,
                label_source, review_status, needs_review, action_taken, report_count,
                review_priority, created_at, updated_at
            )
            SELECT id, message_id, guild_id, channel_id, moderator_id, text, label, reason,
                   source, 'confirmed', 0, NULL, 0, 0, created_at, created_at
            FROM moderator_feedback_legacy
            """
        )
        connection.execute("DROP TABLE moderator_feedback_legacy")
