from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

import discord
from discord import app_commands
from dotenv import load_dotenv

from src.scam_detector.classifier import ScamClassifier
from src.scam_detector.decisions import Decision
from src.scam_detector.guild_config import GuildConfig, InMemoryGuildConfigStore
from src.scam_detector.models import MessageContext
from src.scam_detector.pipeline import DetectionPipeline, DetectionResult


@dataclass(frozen=True)
class BotSettings:
    mod_review_channel_id: int | None = None
    delete_enabled: bool = True
    notify_log_actions: bool = True
    whitelisted_role_ids: frozenset[int] = frozenset()
    command_sync_guild_id: int | None = None


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
        command_sync_guild_id=(
            int(values["COMMAND_SYNC_GUILD_ID"])
            if values.get("COMMAND_SYNC_GUILD_ID")
            else None
        ),
    )


def build_default_guild_config(settings: BotSettings) -> GuildConfig:
    return GuildConfig(
        mod_review_channel_id=settings.mod_review_channel_id,
        delete_enabled=settings.delete_enabled,
        notify_log_actions=settings.notify_log_actions,
        whitelisted_role_ids=settings.whitelisted_role_ids,
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
            f"Final score: {rule_score}",
            f"Rule level: {rule_level}",
            f"Screening reasons: {', '.join(result.screening.reasons) or 'none'}",
            f"Classifier called: {result.classifier_called}",
            f"Classifier skip reason: {result.classifier_skip_reason or 'none'}",
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
        config_store: InMemoryGuildConfigStore | None = None,
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.messages = True
        super().__init__(intents=intents)
        self.settings = settings or BotSettings()
        self.config_store = config_store or InMemoryGuildConfigStore(
            default_config=build_default_guild_config(self.settings)
        )
        self.pipeline = pipeline or DetectionPipeline(
            classifier=ScamClassifier(),
            whitelisted_role_ids=self.settings.whitelisted_role_ids,
        )
        self.tree = app_commands.CommandTree(self)
        self._register_config_commands()

    async def setup_hook(self) -> None:
        if self.settings.command_sync_guild_id is not None:
            guild = discord.Object(id=self.settings.command_sync_guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            return

        await self.tree.sync()

    async def on_ready(self) -> None:
        print(f"Logged in as {self.user} (id={self.user.id if self.user else 'unknown'})")
        if self.settings.mod_review_channel_id is None:
            print("MOD_REVIEW_CHANNEL_ID is not set; detections will only be printed to the console.")

    async def on_message(self, message: discord.Message) -> None:
        if message.author == self.user:
            return

        guild_config = self._get_message_guild_config(message)
        result = self.pipeline.detect(
            build_message_context(message),
            whitelisted_role_ids=guild_config.whitelisted_role_ids,
        )
        if result.decision.action != Decision.ALLOW:
            summary = format_detection_summary(
                author_id=message.author.id,
                channel_id=message.channel.id,
                content=message.content,
                result=result,
            )
            print(summary)
            await self._notify_moderators_if_needed(summary, result.decision.action, guild_config)

        if result.decision.action == Decision.DELETE:
            if not guild_config.delete_enabled:
                print("Delete skipped because BOT_DELETE_ENABLED is false.")
                return
            try:
                await message.delete()
            except discord.Forbidden:
                print("Delete failed: missing Discord permission to manage/delete messages.")
            except discord.HTTPException as exc:
                print(f"Delete failed: {exc}")

    def _get_message_guild_config(self, message: discord.Message) -> GuildConfig:
        if message.guild is None:
            return build_default_guild_config(self.settings)
        return self.config_store.get_config(message.guild.id)

    async def _notify_moderators_if_needed(
        self,
        summary: str,
        action: Decision,
        guild_config: GuildConfig,
    ) -> None:
        if action == Decision.LOG and not guild_config.notify_log_actions:
            return
        if action not in {Decision.LOG, Decision.REVIEW, Decision.DELETE}:
            return
        if guild_config.mod_review_channel_id is None:
            return

        channel = self.get_channel(guild_config.mod_review_channel_id)
        if channel is None:
            try:
                channel = await self.fetch_channel(guild_config.mod_review_channel_id)
            except (discord.Forbidden, discord.NotFound, discord.HTTPException) as exc:
                print(f"Could not access MOD_REVIEW_CHANNEL_ID: {exc}")
                return

        if not hasattr(channel, "send"):
            print("MOD_REVIEW_CHANNEL_ID does not point to a sendable channel.")
            return

        await channel.send(summary)

    def _register_config_commands(self) -> None:
        config_group = app_commands.Group(
            name="scam-config",
            description="Configure scam detection for this server.",
            guild_only=True,
        )
        whitelist_group = app_commands.Group(
            name="whitelist-role",
            description="Manage roles that bypass scam detection.",
            guild_only=True,
        )

        @config_group.command(name="review-channel", description="Set the channel for scam detection alerts.")
        @app_commands.checks.has_permissions(manage_guild=True)
        async def review_channel(
            interaction: discord.Interaction,
            channel: discord.TextChannel,
        ) -> None:
            guild_id = _require_interaction_guild_id(interaction)
            self.config_store.set_review_channel(guild_id, channel.id)
            await interaction.response.send_message(
                f"Scam detection alerts will be sent to {channel.mention}.",
                ephemeral=True,
            )

        @config_group.command(name="delete-enabled", description="Enable or disable automatic deletion.")
        @app_commands.checks.has_permissions(manage_guild=True)
        async def delete_enabled(interaction: discord.Interaction, enabled: bool) -> None:
            guild_id = _require_interaction_guild_id(interaction)
            self.config_store.set_delete_enabled(guild_id, enabled)
            state = "enabled" if enabled else "disabled"
            await interaction.response.send_message(
                f"Automatic deletion is now {state}.",
                ephemeral=True,
            )

        @whitelist_group.command(name="add", description="Add a trusted role that bypasses scam detection.")
        @app_commands.checks.has_permissions(manage_guild=True)
        async def whitelist_add(interaction: discord.Interaction, role: discord.Role) -> None:
            guild_id = _require_interaction_guild_id(interaction)
            self.config_store.add_whitelisted_role(guild_id, role.id)
            await interaction.response.send_message(
                f"{role.mention} now bypasses scam detection.",
                ephemeral=True,
            )

        @whitelist_group.command(name="remove", description="Remove a trusted role from the bypass list.")
        @app_commands.checks.has_permissions(manage_guild=True)
        async def whitelist_remove(interaction: discord.Interaction, role: discord.Role) -> None:
            guild_id = _require_interaction_guild_id(interaction)
            self.config_store.remove_whitelisted_role(guild_id, role.id)
            await interaction.response.send_message(
                f"{role.mention} no longer bypasses scam detection.",
                ephemeral=True,
            )

        @whitelist_group.command(name="list", description="List roles that bypass scam detection.")
        @app_commands.checks.has_permissions(manage_guild=True)
        async def whitelist_list(interaction: discord.Interaction) -> None:
            guild_id = _require_interaction_guild_id(interaction)
            role_mentions = [
                f"<@&{role_id}>"
                for role_id in self.config_store.list_whitelisted_roles(guild_id)
            ]
            message = (
                "Whitelisted roles: " + ", ".join(role_mentions)
                if role_mentions
                else "No whitelisted roles are configured."
            )
            await interaction.response.send_message(message, ephemeral=True)

        config_group.add_command(whitelist_group)
        self.tree.add_command(config_group)


def _require_interaction_guild_id(interaction: discord.Interaction) -> int:
    if interaction.guild_id is None:
        raise app_commands.AppCommandError("This command can only be used in a server.")
    return interaction.guild_id


def main() -> None:
    load_dotenv()
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        raise RuntimeError("DISCORD_TOKEN is required")
    ScamDetectionBot(settings=load_bot_settings_from_env()).run(token)


if __name__ == "__main__":
    main()
