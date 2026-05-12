from src.scam_detector.guild_config import GuildConfig, InMemoryGuildConfigStore


def test_in_memory_guild_config_starts_from_defaults() -> None:
    store = InMemoryGuildConfigStore(
        default_config=GuildConfig(
            mod_review_channel_id=10,
            delete_enabled=False,
            whitelisted_role_ids=frozenset({1}),
        )
    )

    config = store.get_config(guild_id=123)

    assert config.mod_review_channel_id == 10
    assert not config.delete_enabled
    assert config.whitelisted_role_ids == frozenset({1})


def test_in_memory_guild_config_updates_review_channel_and_delete_flag() -> None:
    store = InMemoryGuildConfigStore()

    store.set_review_channel(guild_id=123, channel_id=456)
    store.set_delete_enabled(guild_id=123, enabled=False)

    config = store.get_config(guild_id=123)
    assert config.mod_review_channel_id == 456
    assert not config.delete_enabled


def test_in_memory_guild_config_adds_removes_and_lists_whitelisted_roles() -> None:
    store = InMemoryGuildConfigStore()

    store.add_whitelisted_role(guild_id=123, role_id=1)
    store.add_whitelisted_role(guild_id=123, role_id=2)
    store.remove_whitelisted_role(guild_id=123, role_id=1)

    assert store.list_whitelisted_roles(guild_id=123) == [2]
    assert store.get_config(guild_id=123).whitelisted_role_ids == frozenset({2})
