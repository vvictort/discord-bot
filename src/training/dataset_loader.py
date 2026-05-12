from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from sklearn.model_selection import train_test_split

DEFAULT_DATASET_NAME = "wangyuancheng/discord-phishing-scam"
DEFAULT_OUTPUT_DIR = Path("data/processed")

COLUMN_RENAMES = {
    "lable": "label",
    "msg_content": "text",
}

METADATA_COLUMNS = [
    "msg_timestamp",
    "usr_joined_at",
    "time_since_join",
    "message_length",
    "word_count",
    "has_link",
    "has_mention",
    "num_roles",
]

REQUIRED_COLUMNS = ["text", "label"]
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class DatasetStats:
    total_rows_before_cleaning: int
    total_rows_after_cleaning: int
    duplicate_rows_removed: int
    positive_count: int
    negative_count: int
    positive_percentage: float
    negative_percentage: float


@dataclass(frozen=True)
class PreparedDataset:
    frame: pd.DataFrame
    stats: DatasetStats


@dataclass(frozen=True)
class DatasetSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def normalize_text_for_deduplication(text: str | None) -> str:
    if text is None:
        return ""
    return WHITESPACE_RE.sub(" ", str(text).lower()).strip()


def load_huggingface_dataframe(dataset_name: str = DEFAULT_DATASET_NAME) -> pd.DataFrame:
    try:
        dataset = load_dataset(dataset_name)
        return _dataset_to_dataframe(dataset)
    except Exception as exc:
        return _load_huggingface_csv_fallback(dataset_name, exc)


def _dataset_to_dataframe(dataset: Any) -> pd.DataFrame:
    if hasattr(dataset, "to_pandas"):
        return dataset.to_pandas()

    if isinstance(dataset, dict):
        frames = []
        for split_name, split in dataset.items():
            frame = split.to_pandas() if hasattr(split, "to_pandas") else pd.DataFrame(split)
            frame = frame.copy()
            frame["_source_split"] = split_name
            frames.append(frame)
        if not frames:
            raise ValueError("Hugging Face dataset returned no splits")
        return pd.concat(frames, ignore_index=True)

    return pd.DataFrame(dataset)


def _load_huggingface_csv_fallback(dataset_name: str, original_error: Exception) -> pd.DataFrame:
    try:
        csv_files = [name for name in list_repo_files(dataset_name, repo_type="dataset") if name.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in dataset repository {dataset_name}")
        frames = []
        for csv_file in csv_files:
            local_path = hf_hub_download(repo_id=dataset_name, filename=csv_file, repo_type="dataset")
            frame = pd.read_csv(local_path)
            frame["_source_file"] = csv_file
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)
    except Exception as fallback_error:
        raise RuntimeError(
            f"Could not load Hugging Face dataset {dataset_name} directly or via CSV fallback"
        ) from fallback_error or original_error


def standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    standardized = frame.rename(columns=COLUMN_RENAMES).copy()
    keep_columns = [column for column in REQUIRED_COLUMNS + METADATA_COLUMNS if column in standardized.columns]
    missing = [column for column in REQUIRED_COLUMNS if column not in standardized.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns after standardization: {missing}")
    return standardized[keep_columns]


def convert_label_to_binary(value: object) -> int | None:
    if pd.isna(value):
        return None

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if int(value) in (0, 1):
            return int(value)

    normalized = str(value).strip().lower()
    positive_values = {"1", "true", "yes", "y", "scam", "phishing", "phish", "malicious"}
    negative_values = {"0", "false", "no", "n", "not scam", "not_scam", "safe", "ham", "benign", "normal"}

    if normalized in positive_values:
        return 1
    if normalized in negative_values:
        return 0
    return None


def clean_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    cleaned = standardize_columns(frame)
    cleaned = cleaned.dropna(subset=["text", "label"]).copy()
    cleaned["text"] = cleaned["text"].astype(str)
    cleaned["label"] = cleaned["label"].map(convert_label_to_binary)
    cleaned = cleaned.dropna(subset=["label"]).copy()
    cleaned["label"] = cleaned["label"].astype(int)
    cleaned = cleaned[cleaned["text"].map(normalize_text_for_deduplication).astype(bool)].copy()
    return remove_duplicate_text_rows(cleaned)


def remove_duplicate_text_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    with_normalized = frame.copy()
    with_normalized["_normalized_text"] = with_normalized["text"].map(normalize_text_for_deduplication)
    before = len(with_normalized)
    deduped = with_normalized.drop_duplicates(subset=["_normalized_text"], keep="first").copy()
    removed = before - len(deduped)
    return deduped.drop(columns=["_normalized_text"]).reset_index(drop=True), removed


def compute_dataset_stats(
    frame: pd.DataFrame,
    before_count: int,
    duplicate_rows_removed: int,
) -> DatasetStats:
    positive_count = int((frame["label"] == 1).sum())
    negative_count = int((frame["label"] == 0).sum())
    total = len(frame)
    return DatasetStats(
        total_rows_before_cleaning=before_count,
        total_rows_after_cleaning=total,
        duplicate_rows_removed=duplicate_rows_removed,
        positive_count=positive_count,
        negative_count=negative_count,
        positive_percentage=round((positive_count / total) * 100, 4) if total else 0.0,
        negative_percentage=round((negative_count / total) * 100, 4) if total else 0.0,
    )


def load_and_prepare_dataset(dataset_name: str = DEFAULT_DATASET_NAME) -> PreparedDataset:
    raw_frame = load_huggingface_dataframe(dataset_name)
    before_count = len(raw_frame)
    cleaned, duplicate_rows_removed = clean_dataset(raw_frame)
    stats = compute_dataset_stats(cleaned, before_count, duplicate_rows_removed)
    return PreparedDataset(frame=cleaned, stats=stats)


def create_stratified_splits(
    frame: pd.DataFrame,
    random_state: int = 42,
    train_size: float = 0.70,
    validation_size: float = 0.15,
    test_size: float = 0.15,
) -> DatasetSplits:
    if round(train_size + validation_size + test_size, 6) != 1.0:
        raise ValueError("train_size, validation_size, and test_size must sum to 1.0")

    train, holdout = train_test_split(
        frame,
        train_size=train_size,
        random_state=random_state,
        stratify=frame["label"],
        shuffle=True,
    )
    relative_validation_size = validation_size / (validation_size + test_size)
    validation, test = train_test_split(
        holdout,
        train_size=relative_validation_size,
        random_state=random_state,
        stratify=holdout["label"],
        shuffle=True,
    )
    return DatasetSplits(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def apply_imbalance_strategy(
    train: pd.DataFrame,
    strategy: str = "none",
    negative_positive_ratio: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    if strategy == "none":
        return train.reset_index(drop=True)
    if strategy != "downsample":
        raise ValueError(f"Unsupported imbalance strategy: {strategy}")
    if negative_positive_ratio is None or negative_positive_ratio <= 0:
        raise ValueError("negative_positive_ratio must be a positive integer for downsampling")

    positives = train[train["label"] == 1]
    negatives = train[train["label"] == 0]
    max_negatives = len(positives) * negative_positive_ratio
    sampled_negatives = negatives.sample(
        n=min(len(negatives), max_negatives),
        random_state=random_state,
        replace=False,
    )
    sampled = pd.concat([positives, sampled_negatives], ignore_index=True)
    return sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def save_processed_dataset(
    splits: DatasetSplits,
    stats: DatasetStats,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    splits.train.to_csv(output_path / "train.csv", index=False)
    splits.validation.to_csv(output_path / "validation.csv", index=False)
    splits.test.to_csv(output_path / "test.csv", index=False)
    (output_path / "dataset_stats.json").write_text(json.dumps(asdict(stats), indent=2) + "\n")


def prepare_and_save_dataset(
    dataset_name: str = DEFAULT_DATASET_NAME,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    random_state: int = 42,
) -> PreparedDataset:
    prepared = load_and_prepare_dataset(dataset_name)
    splits = create_stratified_splits(prepared.frame, random_state=random_state)
    save_processed_dataset(splits, prepared.stats, output_dir)
    return prepared


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pull and preprocess the Discord scam dataset.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    prepared = prepare_and_save_dataset(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        random_state=args.random_state,
    )
    print(json.dumps(asdict(prepared.stats), indent=2))


if __name__ == "__main__":
    main()
