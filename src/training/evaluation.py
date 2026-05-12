from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)


def evaluate_predictions(
    y_true: Iterable[int],
    y_probability: Iterable[float],
    threshold: float = 0.50,
) -> dict[str, object]:
    truth = np.asarray(list(y_true), dtype=int)
    probability = np.asarray(list(y_probability), dtype=float)
    prediction = (probability >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) else 0.0

    metrics: dict[str, object] = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(truth, prediction)),
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
    }

    if len(set(truth)) == 2:
        metrics["precision_recall_auc"] = float(average_precision_score(truth, probability))
        metrics["roc_auc"] = float(roc_auc_score(truth, probability))
    else:
        metrics["precision_recall_auc"] = None
        metrics["roc_auc"] = None

    return metrics


def evaluate_thresholds(
    y_true: Iterable[int],
    y_probability: Iterable[float],
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    review_threshold: float = 0.75,
) -> list[dict[str, object]]:
    probability = np.asarray(list(y_probability), dtype=float)
    rows: list[dict[str, object]] = []

    for threshold in thresholds:
        metrics = evaluate_predictions(y_true, probability, threshold)
        metrics["auto_delete_candidates"] = int((probability >= threshold).sum())
        if review_threshold < threshold:
            metrics["review_candidates"] = int(((probability >= review_threshold) & (probability < threshold)).sum())
        else:
            metrics["review_candidates"] = 0
        rows.append(metrics)

    return rows


def evaluate_model(model: object, frame: pd.DataFrame, threshold: float = 0.50) -> dict[str, object]:
    probabilities = model.predict_proba(frame)[:, 1]
    return evaluate_predictions(frame["label"], probabilities, threshold=threshold)


def recommend_thresholds(
    threshold_rows: list[dict[str, object]],
    auto_delete_floor: float = 0.90,
    mod_review: float = 0.75,
    log_only: float = 0.55,
) -> dict[str, float]:
    auto_delete_candidates = [
        row for row in threshold_rows if float(row["threshold"]) >= auto_delete_floor
    ] or threshold_rows
    best = sorted(
        auto_delete_candidates,
        key=lambda row: (
            float(row["false_positive_rate"]),
            -float(row["precision"]),
            -float(row["recall"]),
            -float(row["f1"]),
        ),
    )[0]
    return {
        "auto_delete": float(best["threshold"]),
        "mod_review": mod_review,
        "log_only": log_only,
    }
