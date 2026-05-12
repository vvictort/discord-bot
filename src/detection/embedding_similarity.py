from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.scam_detector.preprocessing import normalize_message_text

# This layer is intentionally lightweight: it uses local TF-IDF vectors as the
# default "embedding" backend so the bot can run without downloading model files.
# A future encoder can be injected through EmbeddingSimilarityMatcher.
SIMILARITY_THRESHOLD = 0.18
HIGH_SIMILARITY_THRESHOLD = 0.24


@dataclass(frozen=True)
class ScamTemplate:
    """An anonymized known scam pattern grouped by moderation category."""

    category: str
    text: str


@dataclass(frozen=True)
class EmbeddingSimilarityResult:
    """Similarity result that callers can safely ignore when unavailable."""

    max_similarity: float
    matched_template: str | None
    matched_category: str | None
    reasons: list[str]
    available: bool = True
    error: str | None = None


DEFAULT_SCAM_TEMPLATES = [
    ScamTemplate(
        category="macbook_giveaway",
        text=(
            "@everyone I upgraded to a new model and I am giving away my old MacBook Air "
            "with charger for free. It is in excellent condition. First come first served. "
            "DM me if interested."
        ),
    ),
    ScamTemplate(
        category="camera_giveaway",
        text=(
            "@everyone I just upgraded my Canon camera and I am letting go of the old one. "
            "It is fully working and well maintained. Message me if you want to claim it."
        ),
    ),
    ScamTemplate(
        category="console_giveaway",
        text=(
            "@everyone I am giving away a PS5 console because I got a new one. "
            "First come first served. DM if interested."
        ),
    ),
    ScamTemplate(
        category="moving_away_free_items",
        text=(
            "@everyone I am moving soon and cannot take my items. MacBook, Xbox, PS5, "
            "smart TV, and other items are free to a good home. DM me for details."
        ),
    ),
    ScamTemplate(
        category="external_contact_giveaway",
        text=(
            "@everyone I am giving out a high value item for free because I upgraded. "
            "Text me on WhatsApp if interested."
        ),
    ),
]


def load_scam_templates(path: str | Path | None) -> list[ScamTemplate]:
    """Load anonymized templates.

    Supported JSON shapes:
    - [{"category": "macbook_giveaway", "text": "..."}]
    - {"macbook_giveaway": ["...", "..."]}
    """

    if path is None:
        return list(DEFAULT_SCAM_TEMPLATES)

    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict):
        templates = []
        for category, texts in payload.items():
            for text in texts:
                templates.append(ScamTemplate(category=category, text=text))
        return templates

    return [
        ScamTemplate(category=item["category"], text=item["text"])
        for item in payload
    ]


def normalize_template_text(text: str | None) -> str:
    return normalize_message_text(text)


def compute_embedding_similarity(
    message_text: str | None,
    templates: Sequence[ScamTemplate],
) -> EmbeddingSimilarityResult:
    """Return the best semantic-template match for one message.

    The current implementation uses character n-gram TF-IDF similarity. It is
    robust to small wording changes and spelling/spacing variants, but it is
    still cheap enough to run only on already suspicious messages.
    """

    if not templates:
        return EmbeddingSimilarityResult(
            max_similarity=0.0,
            matched_template=None,
            matched_category=None,
            reasons=[],
        )

    normalized_message = normalize_template_text(message_text)
    normalized_templates = [normalize_template_text(template.text) for template in templates]
    if not normalized_message:
        return EmbeddingSimilarityResult(
            max_similarity=0.0,
            matched_template=None,
            matched_category=None,
            reasons=[],
        )

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    matrix = vectorizer.fit_transform([normalized_message, *normalized_templates])
    similarities = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
    best_index = int(similarities.argmax())
    max_similarity = float(similarities[best_index])
    matched_template = templates[best_index]
    reasons = _similarity_reasons(max_similarity)

    return EmbeddingSimilarityResult(
        max_similarity=max_similarity,
        matched_template=matched_template.text,
        matched_category=matched_template.category,
        reasons=reasons,
    )


def _similarity_reasons(max_similarity: float) -> list[str]:
    if max_similarity >= HIGH_SIMILARITY_THRESHOLD:
        return [
            "similar_to_known_scam_template",
            "highly_similar_to_known_scam_template",
        ]
    if max_similarity >= SIMILARITY_THRESHOLD:
        return ["similar_to_known_scam_template"]
    return []


class EmbeddingSimilarityMatcher:
    """Optional, fail-closed wrapper around the template similarity backend."""

    def __init__(
        self,
        template_path: str | Path | None = None,
        enabled: bool = True,
        encoder_factory: Callable[[], object] | None = None,
    ) -> None:
        self.enabled = enabled
        self.templates: list[ScamTemplate] = []
        self.available = False
        self.error: str | None = None

        if not enabled:
            self.error = "embedding_similarity_disabled"
            return

        try:
            # The injected factory gives us a future hook for heavier embedding
            # backends while keeping the runtime safe when dependencies are absent.
            if encoder_factory is not None:
                encoder_factory()
            self.templates = load_scam_templates(template_path)
            self.available = True
        except Exception as exc:
            self.error = str(exc)
            self.available = False

    def compare(self, message_text: str | None) -> EmbeddingSimilarityResult:
        if not self.enabled or not self.available:
            return EmbeddingSimilarityResult(
                max_similarity=0.0,
                matched_template=None,
                matched_category=None,
                reasons=[],
                available=False,
                error=self.error,
            )

        try:
            return compute_embedding_similarity(message_text, self.templates)
        except Exception as exc:
            return EmbeddingSimilarityResult(
                max_similarity=0.0,
                matched_template=None,
                matched_category=None,
                reasons=[],
                available=False,
                error=str(exc),
            )
