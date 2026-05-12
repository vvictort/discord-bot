from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.scam_detector.features import METADATA_FEATURES
from src.training.dataset_loader import (
    DEFAULT_DATASET_NAME,
    DEFAULT_OUTPUT_DIR,
    DatasetSplits,
    apply_imbalance_strategy,
    create_stratified_splits,
    load_and_prepare_dataset,
    save_processed_dataset,
)
from src.training.evaluation import DEFAULT_THRESHOLDS, evaluate_model, evaluate_thresholds, recommend_thresholds

MODEL_OUTPUT = Path("models/scam_classifier.joblib")
THRESHOLD_OUTPUT = Path("models/thresholds.json")
METRICS_OUTPUT = Path("models/metrics.json")

ClassWeight = Literal["balanced"] | None


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    imbalance_strategy: str
    negative_positive_ratio: int | None
    class_weight: str | None
    include_metadata: bool = True


def default_experiment_configs() -> list[ExperimentConfig]:
    return [
        ExperimentConfig("natural_balanced_weight", "none", None, "balanced"),
        ExperimentConfig("downsample_1_to_1", "downsample", 1, None),
        ExperimentConfig("downsample_3_to_1", "downsample", 3, None),
        ExperimentConfig("downsample_5_to_1", "downsample", 5, None),
        ExperimentConfig("downsample_3_to_1_balanced_weight", "downsample", 3, "balanced"),
    ]


def _available_metadata_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in METADATA_FEATURES if column in frame.columns]


def build_classifier_pipeline(
    train_frame: pd.DataFrame,
    class_weight: ClassWeight = "balanced",
    include_metadata: bool = True,
) -> Pipeline:
    transformers = [
        (
            "word_tfidf",
            TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True),
            "text",
        ),
        (
            "char_tfidf",
            TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=True),
            "text",
        ),
    ]

    metadata_columns = _available_metadata_columns(train_frame) if include_metadata else []
    if metadata_columns:
        transformers.append(
            (
                "metadata",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                metadata_columns,
            )
        )

    return Pipeline(
        [
            ("features", ColumnTransformer(transformers=transformers)),
            (
                "classifier",
                LogisticRegression(
                    class_weight=class_weight,
                    max_iter=1000,
                    solver="liblinear",
                ),
            ),
        ]
    )


def _prepare_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["text"] = prepared["text"].fillna("").astype(str)
    for column in _available_metadata_columns(prepared):
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    return prepared


def train_classifier(
    train_frame: pd.DataFrame,
    class_weight: ClassWeight = "balanced",
    include_metadata: bool = True,
) -> Pipeline:
    prepared = _prepare_training_frame(train_frame)
    model = build_classifier_pipeline(prepared, class_weight=class_weight, include_metadata=include_metadata)
    model.fit(prepared, prepared["label"].astype(int))
    return model


def train_and_save_classifier(
    train_frame: pd.DataFrame,
    model_output: str | Path = MODEL_OUTPUT,
    class_weight: ClassWeight = "balanced",
    include_metadata: bool = True,
) -> Pipeline:
    model = train_classifier(train_frame, class_weight=class_weight, include_metadata=include_metadata)
    output_path = Path(model_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return model


def run_experiment(
    splits: DatasetSplits,
    config: ExperimentConfig,
    random_state: int = 42,
    thresholds: list[float] | None = None,
) -> tuple[Pipeline, dict[str, object]]:
    train_frame = apply_imbalance_strategy(
        splits.train,
        strategy=config.imbalance_strategy,
        negative_positive_ratio=config.negative_positive_ratio,
        random_state=random_state,
    )
    model = train_classifier(
        train_frame,
        class_weight=config.class_weight,
        include_metadata=config.include_metadata,
    )
    validation_probability = model.predict_proba(splits.validation)[:, 1]
    test_probability = model.predict_proba(splits.test)[:, 1]
    threshold_rows = evaluate_thresholds(
        splits.validation["label"],
        validation_probability,
        thresholds=thresholds or DEFAULT_THRESHOLDS,
        review_threshold=0.75,
    )
    recommended_thresholds = recommend_thresholds(threshold_rows)
    metrics = {
        "config": asdict(config),
        "training_rows": len(train_frame),
        "training_positive_count": int((train_frame["label"] == 1).sum()),
        "training_negative_count": int((train_frame["label"] == 0).sum()),
        "validation": evaluate_model(model, splits.validation),
        "test": evaluate_model(model, splits.test),
        "validation_thresholds": threshold_rows,
        "recommended_thresholds": recommended_thresholds,
        "test_at_auto_delete_threshold": evaluate_model(
            model,
            splits.test,
            threshold=recommended_thresholds["auto_delete"],
        ),
        "validation_probability_sample": [float(value) for value in validation_probability[:10]],
        "test_probability_sample": [float(value) for value in test_probability[:10]],
    }
    return model, metrics


def _selection_key(metrics: dict[str, object]) -> tuple[float, float, float, float]:
    thresholds = metrics["validation_thresholds"]
    rows = [row for row in thresholds if float(row["threshold"]) >= 0.90] or thresholds
    best_row = sorted(
        rows,
        key=lambda row: (
            float(row["false_positive_rate"]),
            -float(row["precision"]),
            -float(row["recall"]),
            -float(row["f1"]),
        ),
    )[0]
    return (
        -float(best_row["false_positive_rate"]),
        float(best_row["precision"]),
        float(best_row["recall"]),
        float(best_row["f1"]),
    )


def train_from_huggingface(
    dataset_name: str = DEFAULT_DATASET_NAME,
    processed_output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    model_output: str | Path = MODEL_OUTPUT,
    thresholds_output: str | Path = THRESHOLD_OUTPUT,
    metrics_output: str | Path = METRICS_OUTPUT,
    imbalance_strategy: str = "none",
    negative_positive_ratio: int | None = None,
    class_weight: str | None = "balanced",
    include_metadata: bool = True,
    run_default_experiments: bool = False,
    random_state: int = 42,
) -> dict[str, object]:
    prepared = load_and_prepare_dataset(dataset_name)
    splits = create_stratified_splits(prepared.frame, random_state=random_state)
    save_processed_dataset(splits, prepared.stats, processed_output_dir)

    configs = (
        default_experiment_configs()
        if run_default_experiments
        else [
            ExperimentConfig(
                name="configured_experiment",
                imbalance_strategy=imbalance_strategy,
                negative_positive_ratio=negative_positive_ratio,
                class_weight=class_weight,
                include_metadata=include_metadata,
            )
        ]
    )

    experiment_results = []
    best_model = None
    best_metrics = None
    for config in configs:
        model, metrics = run_experiment(splits, config, random_state=random_state)
        experiment_results.append(metrics)
        if best_metrics is None or _selection_key(metrics) > _selection_key(best_metrics):
            best_model = model
            best_metrics = metrics

    assert best_model is not None
    assert best_metrics is not None

    model_path = Path(model_output)
    threshold_path = Path(thresholds_output)
    metrics_path = Path(metrics_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, model_path)
    threshold_path.write_text(
        json.dumps(best_metrics["recommended_thresholds"], indent=2) + "\n",
        encoding="utf-8",
    )

    payload = {
        "dataset_name": dataset_name,
        "dataset_stats": asdict(prepared.stats),
        "selected_experiment": best_metrics["config"],
        "selection_priority": [
            "lowest false positive rate at auto-delete threshold",
            "highest precision at auto-delete threshold",
            "reasonable recall",
            "F1 tie-breaker",
        ],
        "experiments": experiment_results,
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _parse_class_weight(value: str) -> str | None:
    if value.lower() in {"none", "null", "off"}:
        return None
    if value != "balanced":
        raise argparse.ArgumentTypeError("--class-weight must be 'balanced' or 'none'")
    return value


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Discord scam classifier.")
    parser.add_argument("--dataset-source", default="huggingface", choices=["huggingface"])
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--processed-output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--imbalance-strategy", default="none", choices=["none", "downsample"])
    parser.add_argument("--negative-positive-ratio", type=int, default=None)
    parser.add_argument("--class-weight", type=_parse_class_weight, default="balanced")
    parser.add_argument("--no-metadata", action="store_true")
    parser.add_argument("--run-default-experiments", action="store_true")
    parser.add_argument("--model-output", default=str(MODEL_OUTPUT))
    parser.add_argument("--thresholds-output", default=str(THRESHOLD_OUTPUT))
    parser.add_argument("--metrics-output", default=str(METRICS_OUTPUT))
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    payload = train_from_huggingface(
        dataset_name=args.dataset_name,
        processed_output_dir=args.processed_output_dir,
        model_output=args.model_output,
        thresholds_output=args.thresholds_output,
        metrics_output=args.metrics_output,
        imbalance_strategy=args.imbalance_strategy,
        negative_positive_ratio=args.negative_positive_ratio,
        class_weight=args.class_weight,
        include_metadata=not args.no_metadata,
        run_default_experiments=args.run_default_experiments,
        random_state=args.random_state,
    )
    print(json.dumps(payload["selected_experiment"], indent=2))


if __name__ == "__main__":
    main()
