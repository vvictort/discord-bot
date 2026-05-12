from src.scam_detector.decisions import Decision, DecisionThresholds, decide_action
from src.scam_detector.models import MessageContext
from src.scam_detector.pipeline import DetectionPipeline
from src.scam_detector.scoring import HIGH_RULE_SCORE, RiskLevel, score_message


KNOWN_GIVEAWAY_SCAM_MESSAGES = [
    "@everyone Sorry to distract your attention. I’m here because I will be giving out my old gadgets which are getting more old day by day since I have gotten new ones. I have given out some and still have a 2022 MacBook Air M2 to give away. So if you’re interested pls dm text me on WhatsApp [PHONE].",
    "Hello @everyone Just upgraded! Giving away my old Canon camera. It's still functional and in good shape. Perfect for photography enthusiasts or anyone wanting to start! DM me if interested in picking it up. Here is my WhatsApp number: [PHONE].",
    "Hello@everyone I want to give out my MacBook Air 2020 & Charger for free, it's in perfect health and good as new. I want to give it out because I just got a new model and thought of giving out the old one to someone who can't afford one and is in need of it. Strictly first come first serve! SEND ME A REQUEST IF YOU ARE INTERESTED.",
    "Hey @everyone, I’ve got a clean, well-kept MacBook Air with charger that I’m not using anymore. It’s super fast, battery is great, and it still looks brand new. If this sounds like something you’d love to have, just send me a DM!",
    "Hello @everyone I want to give out my Mac Book Air 2021 and charger for free. It's as new. I want to give it out because I just got a new model and thought of giving out the old one to someone who can't afford one and is in need of it. Strictly first come first serve! DM IF YOU ARE INTERESTED.",
    "Hey @everyone, I just got new gadgets and a new MacBook M4 couple weeks ago, so I will be giving away my old MacBook Air M2 2022 to any lucky person that reach out first. Excellent condition. Fully functional. 100% FREE, no tricks, no catch. First come, first serve! SEND ME FRIEND REQUEST AND DM ME ASAP IF YOU WANT IT!",
    "Hello @everyone Giving away a PS5 to anyone who's interested! I was given a new one as an incentive in my office and I thought of making someone happy by giving it away and want to pass on my old console to someone who'll enjoy it. First come, first served! DM if you are interested.",
    "Hello @everyone, I’m giving away my MacBook Air 2022 M2 and its charger for free. It’s in perfect condition and looks brand new. I’m giving it away because I recently got a new model and thought I’d share the old one with someone who might need it. Please respond via DM if you’re interested.",
    "@everyone am giving out my Sony AZIV with SmallRig Cage and Sigma 24-70mm F2.8 lens. Very low shutter count. Used by commercial company for headshots and no longer need. It's in perfect health and good as new and I am willing to give it out to someone who can't afford one and is in need of it. Strictly first come first serve!",
    "@everyone I’ve just upgraded my gear, so I’m letting go of my previous Canon camera. It’s fully working and well maintained, great for beginners or photography lovers. Message me if you’d like to claim it.",
    "Hello @everyone. I’m giving away my 2021 MacBook for FREE! It’s in amazing condition and works perfectly. I recently upgraded, so instead of letting this one collect dust, I want it to go to someone who’ll actually use it. Excellent condition. Fully functional. 100% FREE, no tricks, no catch. First come, first serve! DM ME IF YOU’RE INTERESTED OR FRIEND REQUEST.",
    "Hello @everyone. I’m giving away my MacBook AIR 2022 M2 with charger 100% free. Excellent condition, almost like new. I recently upgraded and want this to go to someone who’ll truly use it. First come, first served. DM me if interested!",
    "Hey @everyone! I’m moving overseas very soon and unfortunately can’t take most of my stuff with me. Everything below is FREE to a good home, all items in excellent condition: 2020 MacBook, Xbox, PS5, Smart TV, e-bike, treadmill, and other items. First come, first served! If you’re interested in any of these, please DM me here for photos, more details, or to arrange pickup.",
    "Hello @everyone. Just upgraded some of my gear, so I’m giving away two items: Pro Evolution Soccer 2 and my old Canon PowerShot G7 X Mark II digital camera. Both are in solid condition and ready for a new owner. If you’re interested in either one, DM me.",
    "@Everyone am giving out my MacBook Pro 16 2019. The laptop is fully functional, runs fast, keyboard and trackpad work perfectly, speakers are great, and all ports WiFi Bluetooth work normally. Includes charger. It's in perfect health and good as new and I am willing to give it out to someone who can't afford one and is in need of it. Strictly first come first serve! DM IF YOU ARE INTERESTED.",
]


NORMAL_MESSAGES = [
    "I got a new laptop for school and I’m setting it up tonight.",
    "Selling my old keyboard in the marketplace channel, pickup near campus.",
    "Does anyone know if the library is open late today?",
    "Thanks everyone, see you at the meeting.",
    "Anyone want to play Valorant later?",
    "I upgraded my camera recently and I’m testing it at the club event.",
    "Does anyone have recommendations for a good laptop for CPSC courses?",
    "Giving away some old notes from last term if anyone wants them.",
]


class CountingClassifier:
    def __init__(self) -> None:
        self.calls = 0

    def predict_probability(self, message: MessageContext) -> float | None:
        self.calls += 1
        return 0.01


def test_rule_scoring_rates_plain_message_low() -> None:
    score = score_message(MessageContext(text="regular project update", author_id=1))

    assert score.level == RiskLevel.LOW
    assert score.score == 0


def test_rule_scoring_uses_content_metadata_and_behavior() -> None:
    score = score_message(
        MessageContext(
            text="@everyone free nitro claim https://example.test",
            author_id=1,
            has_link=True,
            has_mention=True,
            member_join_age_seconds=60,
            num_roles=0,
        )
    )

    assert score.level == RiskLevel.HIGH
    assert score.score >= 8
    assert "keyword:free nitro" in score.reasons
    assert "new_member" in score.reasons
    assert "no_roles" in score.reasons


def test_known_giveaway_scam_messages_score_high_or_critical_without_ml() -> None:
    all_reasons = set()

    for text in KNOWN_GIVEAWAY_SCAM_MESSAGES:
        score = score_message(MessageContext(text=text, author_id=1))
        all_reasons.update(score.reasons)

        assert score.level in {RiskLevel.HIGH, RiskLevel.CRITICAL}, text
        assert score.score >= HIGH_RULE_SCORE, text
        assert "mass_mention" in score.reasons
        assert "high_value_item" in score.reasons
        assert "giveaway_language" in score.reasons
        assert any(
            reason in score.reasons
            for reason in {
                "dm_request",
                "friend_request_or_external_contact",
                "whatsapp_or_phone_contact",
                "urgency_phrase",
            }
        )
        assert any(
            reason in score.reasons
            for reason in {
                "high_value_giveaway_dm_pattern",
                "mass_mention_giveaway_pattern",
                "free_high_value_item_pattern",
            }
        )

    assert {
        "free_offer",
        "dm_request",
        "urgency_phrase",
        "friend_request_or_external_contact",
        "whatsapp_or_phone_contact",
        "emotional_need_framing",
        "recently_upgraded_framing",
        "high_value_giveaway_dm_pattern",
        "mass_mention_giveaway_pattern",
        "free_high_value_item_pattern",
    }.issubset(all_reasons)


def test_known_giveaway_scam_message_with_emojis_still_scores_high() -> None:
    score = score_message(
        MessageContext(
            text=(
                "🎉 Hello @everyone 🎁 I’m giving away my MacBook Air 2022 M2 for FREE 💻. "
                "First-come, first-served 🚨. DM me if interested!"
            ),
            author_id=1,
        )
    )

    assert score.level in {RiskLevel.HIGH, RiskLevel.CRITICAL}
    assert "mass_mention" in score.reasons
    assert "high_value_item" in score.reasons
    assert "giveaway_language" in score.reasons
    assert "dm_request" in score.reasons


def test_normal_messages_do_not_score_as_critical_or_auto_delete() -> None:
    for text in NORMAL_MESSAGES:
        score = score_message(MessageContext(text=text, author_id=1))
        decision = decide_action(rule_score=score.score, classifier_probability=None)

        assert score.level in {RiskLevel.LOW, RiskLevel.MEDIUM}, text
        assert decision.action != Decision.DELETE


def test_decision_allows_low_risk_messages() -> None:
    decision = decide_action(rule_score=0, classifier_probability=None)

    assert decision.action == Decision.ALLOW


def test_decision_logs_medium_rule_score_without_classifier() -> None:
    decision = decide_action(rule_score=3, classifier_probability=None)

    assert decision.action == Decision.LOG


def test_decision_flags_review_for_mod_review_probability_band() -> None:
    thresholds = DecisionThresholds(auto_delete=0.90, mod_review=0.75, log_only=0.55)
    decision = decide_action(rule_score=6, classifier_probability=0.80, thresholds=thresholds)

    assert decision.action == Decision.REVIEW


def test_decision_deletes_only_at_auto_delete_threshold() -> None:
    thresholds = DecisionThresholds(auto_delete=0.90, mod_review=0.75, log_only=0.55)
    decision = decide_action(rule_score=6, classifier_probability=0.95, thresholds=thresholds)

    assert decision.action == Decision.DELETE


def test_high_rule_score_is_not_weakened_by_low_classifier_probability() -> None:
    decision = decide_action(rule_score=HIGH_RULE_SCORE, classifier_probability=0.01)

    assert decision.action == Decision.REVIEW


def test_obvious_rule_based_scam_skips_classifier() -> None:
    classifier = CountingClassifier()
    pipeline = DetectionPipeline(classifier=classifier)

    result = pipeline.detect(MessageContext(text=KNOWN_GIVEAWAY_SCAM_MESSAGES[0], author_id=1))

    assert result.rule_score is not None
    assert result.rule_score.level in {RiskLevel.HIGH, RiskLevel.CRITICAL}
    assert result.classifier_called is False
    assert result.classifier_skip_reason in {"high_rule_score", "critical_rule_score"}
    assert classifier.calls == 0
    assert result.decision.action == Decision.REVIEW
