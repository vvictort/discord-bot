from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class GuildConfig:
    mod_review_channel_id: int | None = None
    delete_enabled: bool = True
    notify_log_actions: bool = True
    whitelisted_role_ids: frozenset[int] = frozenset()
    auto_delete_critical: bool = True
    auto_delete_high: bool = False
    critical_rule_score_threshold: int | None = None
    high_rule_score_threshold: int | None = None
    mod_review_threshold: float = 0.75


class InMemoryGuildConfigStore:
    def __init__(self, default_config: GuildConfig | None = None) -> None:
        self.default_config = default_config or GuildConfig()
        self._configs: dict[int, GuildConfig] = {}

    def get_config(self, guild_id: int) -> GuildConfig:
        return self._configs.get(guild_id, self.default_config)

    def set_review_channel(self, guild_id: int, channel_id: int | None) -> GuildConfig:
        config = replace(self.get_config(guild_id), mod_review_channel_id=channel_id)
        self._configs[guild_id] = config
        return config

    def set_delete_enabled(self, guild_id: int, enabled: bool) -> GuildConfig:
        config = replace(self.get_config(guild_id), delete_enabled=enabled)
        self._configs[guild_id] = config
        return config

    def add_whitelisted_role(self, guild_id: int, role_id: int) -> GuildConfig:
        current = set(self.get_config(guild_id).whitelisted_role_ids)
        current.add(role_id)
        config = replace(self.get_config(guild_id), whitelisted_role_ids=frozenset(current))
        self._configs[guild_id] = config
        return config

    def remove_whitelisted_role(self, guild_id: int, role_id: int) -> GuildConfig:
        current = set(self.get_config(guild_id).whitelisted_role_ids)
        current.discard(role_id)
        config = replace(self.get_config(guild_id), whitelisted_role_ids=frozenset(current))
        self._configs[guild_id] = config
        return config

    def list_whitelisted_roles(self, guild_id: int) -> list[int]:
        return sorted(self.get_config(guild_id).whitelisted_role_ids)
