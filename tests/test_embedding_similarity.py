from src.detection.embedding_similarity import (
    HIGH_SIMILARITY_THRESHOLD,
    EmbeddingSimilarityMatcher,
    EmbeddingSimilarityResult,
    compute_embedding_similarity,
    load_scam_templates,
    normalize_template_text,
)
from src.scam_detector.models import MessageContext
from src.scam_detector.pipeline import DetectionPipeline
from src.scam_detector.scoring import RiskLevel


class CountingSimilarity:
    def __init__(self, result: EmbeddingSimilarityResult | None = None) -> None:
        self.calls = 0
        self.result = result or EmbeddingSimilarityResult(
            max_similarity=0.0,
            matched_template=None,
            matched_category=None,
            reasons=[],
        )

    def compare(self, message_text: str | None) -> EmbeddingSimilarityResult:
        self.calls += 1
        return self.result


def test_normalize_template_text_ignores_formatting_noise() -> None:
    assert normalize_template_text("🎁 Mac Book Air — first-come, first-served!") == (
        "macbook air first come first served"
    )


def test_macbook_giveaway_paraphrase_is_similar_to_known_template() -> None:
    result = compute_embedding_similarity(
        "I upgraded recently and want my old MacBook Air to go to someone who needs it. DM me.",
        load_scam_templates(None),
    )

    assert result.max_similarity >= HIGH_SIMILARITY_THRESHOLD
    assert result.matched_category == "macbook_giveaway"
    assert "highly_similar_to_known_scam_template" in result.reasons


def test_canon_camera_giveaway_paraphrase_is_similar_to_known_template() -> None:
    result = compute_embedding_similarity(
        "Just upgraded my Canon camera. It works well and I want to give it away. Message me.",
        load_scam_templates(None),
    )

    assert result.max_similarity >= HIGH_SIMILARITY_THRESHOLD
    assert result.matched_category == "camera_giveaway"


def test_ps5_giveaway_paraphrase_is_similar_to_known_template() -> None:
    result = compute_embedding_similarity(
        "Giving away my PS5 console because I got a new one. First come first served, DM if interested.",
        load_scam_templates(None),
    )

    assert result.max_similarity >= HIGH_SIMILARITY_THRESHOLD
    assert result.matched_category == "console_giveaway"


def test_normal_laptop_discussion_is_not_highly_similar() -> None:
    result = compute_embedding_similarity(
        "Does anyone have laptop recommendations for CPSC classes?",
        load_scam_templates(None),
    )

    assert result.max_similarity < HIGH_SIMILARITY_THRESHOLD
    assert "highly_similar_to_known_scam_template" not in result.reasons


def test_normal_marketplace_sale_is_not_highly_similar() -> None:
    result = compute_embedding_similarity(
        "Selling my old keyboard in the marketplace channel for pickup near campus.",
        load_scam_templates(None),
    )

    assert result.max_similarity < HIGH_SIMILARITY_THRESHOLD
    assert "highly_similar_to_known_scam_template" not in result.reasons


def test_missing_embedding_dependency_does_not_crash_matcher() -> None:
    matcher = EmbeddingSimilarityMatcher(
        encoder_factory=lambda: (_ for _ in ()).throw(ImportError("missing optional encoder"))
    )

    result = matcher.compare("Giving away a MacBook, DM me.")

    assert not result.available
    assert result.max_similarity == 0.0
    assert result.reasons == []


def test_embeddings_run_only_for_uncertain_suspicious_messages() -> None:
    similarity = CountingSimilarity()
    pipeline = DetectionPipeline(classifier=None, embedding_similarity=similarity)

    safe = pipeline.detect(MessageContext(text="regular project update", author_id=1))
    uncertain = pipeline.detect(MessageContext(text="DM me about this MacBook", author_id=1))
    critical = pipeline.detect(
        MessageContext(
            text="@everyone giving away my MacBook Air for free. First come first served. DM me.",
            author_id=1,
        )
    )

    assert safe.embedding_called is False
    assert uncertain.embedding_called is True
    assert critical.embedding_called is False
    assert critical.embedding_skip_reason in {"high_rule_score", "critical_rule_score"}
    assert similarity.calls == 1


def test_high_embedding_similarity_increases_uncertain_rule_score() -> None:
    similarity = CountingSimilarity(
        EmbeddingSimilarityResult(
            max_similarity=0.91,
            matched_template="macbook template",
            matched_category="macbook_giveaway",
            reasons=[
                "similar_to_known_scam_template",
                "highly_similar_to_known_scam_template",
            ],
        )
    )
    pipeline = DetectionPipeline(classifier=None, embedding_similarity=similarity)

    result = pipeline.detect(MessageContext(text="DM me about this MacBook", author_id=1))

    assert result.rule_score is not None
    assert result.rule_score.level in {RiskLevel.MEDIUM, RiskLevel.HIGH}
    assert "highly_similar_to_known_scam_template" in result.rule_score.reasons
    assert result.embedding_similarity == 0.91
    assert result.embedding_matched_category == "macbook_giveaway"
