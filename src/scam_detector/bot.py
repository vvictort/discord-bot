from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

import discord
from dotenv import load_dotenv

from src.scam_detector.classifier import ScamClassifier
from src.scam_detector.decisions import Decision
from src.scam_detector.models import MessageContext
from src.scam_detector.pipeline import DetectionPipeline, DetectionResult


@dataclass(frozen=True)
class BotSettings:
    mod_review_channel_id: int | None = None
    delete_enabled: bool = True
    notify_log_actions: bool = True
    whitelisted_role_ids: frozenset[int] = frozenset()


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_role_ids(value: str | None) -> frozenset[int]:
    if not value:
        return frozenset()

    role_ids: set[int] = set()
    for raw_role_id in value.split(","):
        role_id = raw_role_id.strip()
        if not role_id:
            continue
        role_ids.add(int(role_id))
    return frozenset(role_ids)


def load_bot_settings_from_env(env: Mapping[str, str] | None = None) -> BotSettings:
    values = env or os.environ
    review_channel = values.get("MOD_REVIEW_CHANNEL_ID")
    return BotSettings(
        mod_review_channel_id=int(review_channel) if review_channel else None,
        delete_enabled=_parse_bool(values.get("BOT_DELETE_ENABLED"), default=True),
        notify_log_actions=_parse_bool(values.get("BOT_NOTIFY_LOG_ACTIONS"), default=True),
        whitelisted_role_ids=_parse_role_ids(values.get("WHITELISTED_ROLE_IDS")),
    )


def format_detection_summary(
    author_id: int | str,
    channel_id: int | str | None,
    content: str,
    result: DetectionResult,
) -> str:
    rule_score = result.rule_score.score if result.rule_score else 0
    rule_level = result.rule_score.level.value if result.rule_score else "none"
    probability = (
        f"{result.classifier_probability:.3f}"
        if result.classifier_probability is not None
        else "none"
    )
    preview = content.replace("`", "'").strip()
    if len(preview) > 300:
        preview = preview[:297] + "..."

    return "\n".join(
        [
            "**Scam detection event**",
            f"Action: {result.decision.action.value}",
            f"Reason: {result.decision.reason}",
            f"Author: {author_id}",
            f"Channel: {channel_id}",
            f"Rule score: {rule_score}",
            f"Rule level: {rule_level}",
            f"Screening reasons: {', '.join(result.screening.reasons) or 'none'}",
            f"Classifier called: {result.classifier_called}",
            f"Classifier probability: {probability}",
            f"Message: `{preview}`",
        ]
    )


def build_message_context(message: discord.Message) -> MessageContext:
    member = message.author if isinstance(message.author, discord.Member) else None
    now = discord.utils.utcnow()
    joined_at = member.joined_at if member else None
    member_join_age_seconds = (now - joined_at).total_seconds() if joined_at else None
    author_account_age_seconds = (now - message.author.created_at).total_seconds()
    return MessageContext(
        text=message.content,
        author_id=message.author.id,
        guild_id=message.guild.id if message.guild else None,
        author_is_bot=message.author.bot,
        author_account_age_seconds=author_account_age_seconds,
        member_join_age_seconds=member_join_age_seconds,
        message_length=len(message.content or ""),
        word_count=len((message.content or "").split()),
        has_link=("http://" in message.content.lower() or "https://" in message.content.lower()),
        has_mention=bool(message.mentions or message.mention_everyone),
        num_roles=max(len(member.roles) - 1, 0) if member else None,
        author_role_ids=tuple(role.id for role in member.roles) if member else (),
    )


class ScamDetectionBot(discord.Client):
    def __init__(
        self,
        pipeline: DetectionPipeline | None = None,
        settings: BotSettings | None = None,
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.messages = True
        super().__init__(intents=intents)
        self.settings = settings or BotSettings()
        self.pipeline = pipeline or DetectionPipeline(
            classifier=ScamClassifier(),
            whitelisted_role_ids=self.settings.whitelisted_role_ids,
        )

    async def on_ready(self) -> None:
        print(f"Logged in as {self.user} (id={self.user.id if self.user else 'unknown'})")
        if self.settings.mod_review_channel_id is None:
            print("MOD_REVIEW_CHANNEL_ID is not set; detections will only be printed to the console.")

    async def on_message(self, message: discord.Message) -> None:
        if message.author == self.user:
            return

        result = self.pipeline.detect(build_message_context(message))
        if result.decision.action != Decision.ALLOW:
            summary = format_detection_summary(
                author_id=message.author.id,
                channel_id=message.channel.id,
                content=message.content,
                result=result,
            )
            print(summary)
            await self._notify_moderators_if_needed(summary, result.decision.action)

        if result.decision.action == Decision.DELETE:
            if not self.settings.delete_enabled:
                print("Delete skipped because BOT_DELETE_ENABLED is false.")
                return
            try:
                await message.delete()
            except discord.Forbidden:
                print("Delete failed: missing Discord permission to manage/delete messages.")
            except discord.HTTPException as exc:
                print(f"Delete failed: {exc}")

    async def _notify_moderators_if_needed(self, summary: str, action: Decision) -> None:
        if action == Decision.LOG and not self.settings.notify_log_actions:
            return
        if action not in {Decision.LOG, Decision.REVIEW, Decision.DELETE}:
            return
        if self.settings.mod_review_channel_id is None:
            return

        channel = self.get_channel(self.settings.mod_review_channel_id)
        if channel is None:
            try:
                channel = await self.fetch_channel(self.settings.mod_review_channel_id)
            except (discord.Forbidden, discord.NotFound, discord.HTTPException) as exc:
                print(f"Could not access MOD_REVIEW_CHANNEL_ID: {exc}")
                return

        if not hasattr(channel, "send"):
            print("MOD_REVIEW_CHANNEL_ID does not point to a sendable channel.")
            return

        await channel.send(summary)


def main() -> None:
    load_dotenv()
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        raise RuntimeError("DISCORD_TOKEN is required")
    ScamDetectionBot(settings=load_bot_settings_from_env()).run(token)


if __name__ == "__main__":
    main()
