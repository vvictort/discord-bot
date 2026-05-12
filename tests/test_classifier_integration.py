from src.scam_detector.classifier import ScamClassifier
from src.scam_detector.decisions import Decision
from src.scam_detector.models import MessageContext
from src.scam_detector.pipeline import DetectionPipeline


class CountingClassifier:
    def __init__(self, probability: float | None = 0.80) -> None:
        self.probability = probability
        self.calls = 0

    def predict_probability(self, message: MessageContext) -> float | None:
        self.calls += 1
        return self.probability


def test_missing_model_does_not_crash_classifier(tmp_path) -> None:
    classifier = ScamClassifier(model_path=tmp_path / "missing.joblib", enabled=True)

    assert classifier.predict_probability(MessageContext(text="free nitro claim", author_id=1)) is None


def test_disabled_classifier_returns_none_safely(tmp_path) -> None:
    classifier = ScamClassifier(model_path=tmp_path / "missing.joblib", enabled=False)

    assert classifier.predict_probability(MessageContext(text="free nitro claim", author_id=1)) is None


def test_detection_pipeline_skips_classifier_for_untriggered_message() -> None:
    classifier = CountingClassifier()
    pipeline = DetectionPipeline(classifier=classifier)

    result = pipeline.detect(MessageContext(text="normal update", author_id=1, guild_id=1))

    assert result.decision.action == Decision.ALLOW
    assert result.classifier_probability is None
    assert classifier.calls == 0


def test_detection_pipeline_calls_classifier_for_medium_or_high_rule_score() -> None:
    classifier = CountingClassifier(probability=0.80)
    pipeline = DetectionPipeline(classifier=classifier)

    result = pipeline.detect(
        MessageContext(
            text="free nitro claim",
            author_id=1,
            guild_id=1,
        )
    )

    assert classifier.calls == 1
    assert result.classifier_probability == 0.80
    assert result.decision.action == Decision.REVIEW
