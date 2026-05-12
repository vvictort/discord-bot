from __future__ import annotations

import json

import pandas as pd

from src.training import dataset_loader


class MockSplit:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)


class MockDatasetDict(dict):
    pass


def test_hugging_face_loader_uses_mocked_dataset_and_cleans_columns(monkeypatch) -> None:
    rows = [
        {
            "lable": "scam",
            "msg_content": "Free Nitro",
            "msg_timestamp": "2024-01-01T00:00:00Z",
            "message_length": 10,
            "has_link": True,
        },
        {"lable": "not scam", "msg_content": "hello", "num_roles": 2},
        {"lable": None, "msg_content": "missing label"},
        {"lable": "scam", "msg_content": None},
    ]
    mock_dataset = MockDatasetDict(train=MockSplit(rows))
    monkeypatch.setattr(dataset_loader, "load_dataset", lambda name: mock_dataset)

    prepared = dataset_loader.load_and_prepare_dataset("example/dataset")

    assert list(prepared.frame["label"]) == [1, 0]
    assert list(prepared.frame["text"]) == ["Free Nitro", "hello"]
    assert "msg_timestamp" in prepared.frame.columns
    assert "message_length" in prepared.frame.columns
    assert "has_link" in prepared.frame.columns
    assert "num_roles" in prepared.frame.columns
    assert prepared.stats.total_rows_before_cleaning == 4
    assert prepared.stats.total_rows_after_cleaning == 2


def test_normalize_text_for_deduplication() -> None:
    assert dataset_loader.normalize_text_for_deduplication("  FREE   Nitro\nNow ") == "free nitro now"


def test_deduplication_removes_only_identical_normalized_text() -> None:
    frame = pd.DataFrame(
        [
            {"text": "FREE Nitro", "label": 1},
            {"text": " free   nitro ", "label": 1},
            {"text": "FREE Nitro today", "label": 1},
            {"text": "hello", "label": 0},
        ]
    )

    deduped, removed = dataset_loader.remove_duplicate_text_rows(frame)

    assert removed == 1
    assert list(deduped["text"]) == ["FREE Nitro", "FREE Nitro today", "hello"]


def test_stratified_split_creates_three_natural_distribution_splits() -> None:
    frame = pd.DataFrame(
        [{"text": f"scam {i}", "label": 1} for i in range(20)]
        + [{"text": f"safe {i}", "label": 0} for i in range(80)]
    )

    splits = dataset_loader.create_stratified_splits(frame, random_state=7)

    assert len(splits.train) == 70
    assert len(splits.validation) == 15
    assert len(splits.test) == 15
    for split in (splits.train, splits.validation, splits.test):
        assert set(split["label"]) == {0, 1}
    assert splits.validation["label"].mean() == 0.2
    assert splits.test["label"].mean() == 0.2


def test_downsampling_keeps_all_positives_and_samples_negatives_only_for_train() -> None:
    train = pd.DataFrame(
        [{"text": f"scam {i}", "label": 1} for i in range(10)]
        + [{"text": f"safe {i}", "label": 0} for i in range(100)]
    )
    validation = pd.DataFrame([{"text": "safe validation", "label": 0}, {"text": "scam validation", "label": 1}])
    test = pd.DataFrame([{"text": "safe test", "label": 0}, {"text": "scam test", "label": 1}])

    sampled = dataset_loader.apply_imbalance_strategy(
        train,
        strategy="downsample",
        negative_positive_ratio=3,
        random_state=7,
    )

    assert int((sampled["label"] == 1).sum()) == 10
    assert int((sampled["label"] == 0).sum()) == 30
    assert len(validation) == 2
    assert len(test) == 2


def test_save_processed_dataset_writes_splits_and_stats(tmp_path) -> None:
    frame = pd.DataFrame(
        [{"text": f"scam {i}", "label": 1} for i in range(20)]
        + [{"text": f"safe {i}", "label": 0} for i in range(80)]
    )
    splits = dataset_loader.create_stratified_splits(frame, random_state=7)
    stats = dataset_loader.compute_dataset_stats(frame, before_count=100, duplicate_rows_removed=0)

    dataset_loader.save_processed_dataset(splits, stats, tmp_path)

    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "validation.csv").exists()
    assert (tmp_path / "test.csv").exists()
    stats_payload = json.loads((tmp_path / "dataset_stats.json").read_text())
    assert stats_payload["positive_count"] == 20
    assert stats_payload["negative_count"] == 80
