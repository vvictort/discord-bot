from __future__ import annotations

import os

import discord
from dotenv import load_dotenv

from src.scam_detector.classifier import ScamClassifier
from src.scam_detector.decisions import Decision
from src.scam_detector.models import MessageContext
from src.scam_detector.pipeline import DetectionPipeline


def build_message_context(message: discord.Message) -> MessageContext:
    member = message.author if isinstance(message.author, discord.Member) else None
    return MessageContext(
        text=message.content,
        author_id=message.author.id,
        guild_id=message.guild.id if message.guild else None,
        author_is_bot=message.author.bot,
        member_join_age_seconds=None,
        message_length=len(message.content or ""),
        word_count=len((message.content or "").split()),
        has_link=("http://" in message.content.lower() or "https://" in message.content.lower()),
        has_mention=bool(message.mentions or message.mention_everyone),
        num_roles=max(len(member.roles) - 1, 0) if member else None,
    )


class ScamDetectionBot(discord.Client):
    def __init__(self, pipeline: DetectionPipeline | None = None) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.messages = True
        super().__init__(intents=intents)
        self.pipeline = pipeline or DetectionPipeline(classifier=ScamClassifier())

    async def on_message(self, message: discord.Message) -> None:
        if message.author == self.user:
            return

        result = self.pipeline.detect(build_message_context(message))
        if result.decision.action == Decision.DELETE:
            await message.delete()
        elif result.decision.action == Decision.REVIEW:
            # Moderation channel routing will be added when server config exists.
            return


def main() -> None:
    load_dotenv()
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        raise RuntimeError("DISCORD_TOKEN is required")
    ScamDetectionBot().run(token)


if __name__ == "__main__":
    main()
