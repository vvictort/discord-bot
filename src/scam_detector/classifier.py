from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.scam_detector.models import MessageContext
from src.training.train_model import METADATA_FEATURES


class ScamClassifier:
    def __init__(self, model_path: str | Path = "models/scam_classifier.joblib", enabled: bool = True) -> None:
        self.model_path = Path(model_path)
        self.enabled = enabled
        self._model: Any | None = None
        self._load_attempted = False

    def predict_probability(self, message: MessageContext) -> float | None:
        if not self.enabled:
            return None

        model = self._load_model()
        if model is None:
            return None

        frame = self._message_to_frame(message)
        try:
            probability = model.predict_proba(frame)[:, 1][0]
        except Exception:
            return None
        return float(probability)

    def _load_model(self) -> Any | None:
        if self._load_attempted:
            return self._model

        self._load_attempted = True
        if not self.model_path.exists():
            return None

        try:
            self._model = joblib.load(self.model_path)
        except Exception:
            self._model = None
        return self._model

    @staticmethod
    def _message_to_frame(message: MessageContext) -> pd.DataFrame:
        payload: dict[str, object] = {
            "text": message.text or "",
            "message_length": message.message_length,
            "word_count": message.word_count,
            "has_link": int(message.has_link),
            "has_mention": int(message.has_mention),
            "num_roles": message.num_roles,
            "time_since_join": message.member_join_age_seconds,
        }
        return pd.DataFrame([{key: value for key, value in payload.items() if key == "text" or key in METADATA_FEATURES}])
