from __future__ import annotations

import joblib
import pandas as pd

from src.training.evaluation import evaluate_predictions, evaluate_thresholds
from src.training.train_model import train_classifier, train_and_save_classifier


def _toy_training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"text": "free nitro claim now", "label": 1, "message_length": 20, "word_count": 4, "has_link": 1},
            {"text": "claim discord gift link", "label": 1, "message_length": 24, "word_count": 4, "has_link": 1},
            {"text": "verify wallet for prize", "label": 1, "message_length": 23, "word_count": 4, "has_link": 1},
            {"text": "limited time steam gift", "label": 1, "message_length": 23, "word_count": 4, "has_link": 0},
            {"text": "team meeting moved tomorrow", "label": 0, "message_length": 27, "word_count": 4, "has_link": 0},
            {"text": "thanks for the update", "label": 0, "message_length": 21, "word_count": 4, "has_link": 0},
            {"text": "deploy finished successfully", "label": 0, "message_length": 28, "word_count": 3, "has_link": 0},
            {"text": "please review the pull request", "label": 0, "message_length": 30, "word_count": 5, "has_link": 0},
        ]
    )


def test_training_saves_and_loads_joblib_model(tmp_path) -> None:
    frame = _toy_training_frame()
    output_path = tmp_path / "scam_classifier.joblib"

    model = train_and_save_classifier(
        train_frame=frame,
        model_output=output_path,
        class_weight=None,
        include_metadata=True,
    )
    loaded = joblib.load(output_path)
    probabilities = loaded.predict_proba(frame[["text", "message_length", "word_count", "has_link"]])[:, 1]

    assert output_path.exists()
    assert model is not None
    assert probabilities.min() >= 0
    assert probabilities.max() <= 1


def test_train_classifier_supports_balanced_class_weight() -> None:
    frame = _toy_training_frame()

    model = train_classifier(frame, class_weight="balanced", include_metadata=False)
    probabilities = model.predict_proba(frame[["text"]])[:, 1]

    assert len(probabilities) == len(frame)


def test_evaluation_computes_core_metrics_and_false_positive_rate() -> None:
    metrics = evaluate_predictions(
        y_true=[0, 0, 1, 1],
        y_probability=[0.10, 0.80, 0.90, 0.40],
        threshold=0.50,
    )

    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["false_positive_rate"] == 0.5
    assert metrics["false_negative_rate"] == 0.5
    assert "precision_recall_auc" in metrics
    assert "roc_auc" in metrics


def test_threshold_evaluation_reports_multiple_thresholds_and_review_band() -> None:
    rows = evaluate_thresholds(
        y_true=[0, 0, 1, 1],
        y_probability=[0.10, 0.80, 0.90, 0.40],
        thresholds=[0.5, 0.8, 0.9],
        review_threshold=0.55,
    )

    assert [row["threshold"] for row in rows] == [0.5, 0.8, 0.9]
    assert rows[0]["auto_delete_candidates"] == 2
    assert rows[2]["auto_delete_candidates"] == 1
    assert rows[2]["review_candidates"] == 1
