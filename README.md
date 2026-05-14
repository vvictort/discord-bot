# Discord Scam Detection Bot

Python 3 Discord moderation bot focused on the core server-admin workflow:

- Watch chat for common scam and giveaway patterns.
- Send compact alerts to a moderator channel.
- Run in `Monitor` mode or `Protect` mode.
- Let admins trust staff roles that should bypass checks.
- Store moderator review candidates for future training.

The project is built around a cheap-first detection pipeline:

1. Ignore ineligible messages.
2. Run lightweight trigger screening.
3. Score suspicious messages with deterministic rules.
4. Optionally compare uncertain suspicious messages against known scam-template embeddings.
5. Optionally call an ML classifier only for suspicious messages.
6. Decide whether to allow, log, send to review, or delete.
7. Store moderator-confirmed labels for future training.

## Tech stack

- Python 3.10+ with modern type hints and dataclasses.
- `discord.py` for Discord gateway events and slash commands.
- `python-dotenv` for local `.env` configuration.
- `scikit-learn` for TF-IDF features, template similarity, and Logistic Regression.
- `pandas`, `numpy`, and `joblib` for training data, metrics, and model artifacts.
- Hugging Face `datasets` and `huggingface-hub` for bootstrap dataset loading.
- SQLite through Python's standard `sqlite3` module for feedback storage.
- `pytest` and `pytest-asyncio` for unit and async bot tests.

## Local setup

Run commands from the repository root. Use `python3` to create the virtualenv;
after activation, the virtualenv exposes `python`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run tests:

```bash
python -m pytest
```

Without activating the virtualenv, use:

```bash
.venv/bin/python -m pytest
```

## Project layout

- `src/scam_detector/`: bot runtime, rules, scoring, decisions, config, and feedback.
- `src/detection/`: optional template similarity layer.
- `src/training/`: Hugging Face dataset loading, model training, and evaluation.
- `tests/`: unit and integration-style tests for runtime and training behavior.
- `data/`: local raw/processed data, ignored by git.
- `models/`: local trained model artifacts, ignored by git.

## Runtime architecture

The runtime path intentionally avoids expensive ML on ordinary chat traffic:

- Tier 0 eligibility: empty messages, bot authors, and non-guild messages are ignored.
- Tier 1 screening: cheap keyword, link, and mention checks run on eligible messages.
- Tier 2 scoring: suspicious messages receive deterministic risk scoring from content and metadata.
- Tier 3 embedding similarity: optional known-template similarity runs only for cheap-screened medium/uncertain messages.
- Tier 4 classifier: the optional scikit-learn model is called only for uncertain messages that still need help after rules and embeddings.
- Decision engine: low risk is allowed, medium confidence goes to review, and critical/high-confidence detections can take immediate action.
- Feedback loop: moderator-confirmed scam and false-positive labels are stored for future training.

Rules are the primary safety layer. They are deterministic, explainable, and catch obvious scam templates without needing ML. Embedding similarity is a secondary optional layer for wording changes that are close to known giveaway scams. The trained classifier is a separate optional model trained from labeled data.

Embeddings do not replace rules and are not run on every message. The bot only calls embedding similarity after cheap screening has triggered and the rule score is still low/medium. Critical or high rule scores skip embeddings and classifier calls because the rule evidence is already strong. This keeps normal chat cheap and avoids using semantic similarity as a broad surveillance step.

## Action bands

The bot uses action bands so high-confidence scam messages do not stay visible while waiting for review:

- `CRITICAL`: delete immediately when `AUTO_DELETE_CRITICAL=true`, log evidence to the mod channel, and store a pending review candidate.
- `HIGH`: delete when `AUTO_DELETE_HIGH=true`; otherwise send to mod review.
- `MEDIUM`: send to mod review without auto-delete by default.
- `LOW`: allow unless manually reported.

Bot-flagged candidates are stored as pending review items, not training labels:

- `label = null`
- `label_source = bot_flag`
- `review_status = pending`
- `needs_review = true`
- `action_taken = deleted`, `review`, or another action outcome

Moderator outcomes determine training eligibility:

- Confirm Scam: `label = scam`, `label_source = moderator_confirmed`
- False Positive: `label = not_scam`, `label_source = moderator_confirmed`
- Ignore: `review_status = ignored`

User reports are weak signals. They increase review priority, but they do not directly delete messages and do not become training labels. Server members can report an unflagged message from Discord's message app menu with **Apps > Report scam**. The bot stores that message as a pending `user_report` review item and sends it to the mod alert channel.

## Dataset preparation

The bootstrap dataset is Hugging Face only:

- `wangyuancheng/discord-phishing-scam`

Pull, clean, deduplicate, stratify, and write processed splits:

```bash
python -m src.training.dataset_loader \
  --dataset-name wangyuancheng/discord-phishing-scam \
  --output-dir data/processed
```

This writes:

- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`
- `data/processed/dataset_stats.json`

Raw and processed data are ignored by git.

## Training

Train one configured model:

```bash
python -m src.training.train_model \
  --dataset-source huggingface \
  --dataset-name wangyuancheng/discord-phishing-scam \
  --imbalance-strategy downsample \
  --negative-positive-ratio 3 \
  --class-weight balanced \
  --model-output models/scam_classifier.joblib
```

Run the default imbalance experiments:

```bash
python -m src.training.train_model \
  --dataset-source huggingface \
  --dataset-name wangyuancheng/discord-phishing-scam \
  --run-default-experiments
```

Default experiments include:

- Natural distribution with `class_weight="balanced"`
- 1:1 negative downsampling without class weight
- 3:1 negative downsampling without class weight
- 5:1 negative downsampling without class weight
- 3:1 negative downsampling with `class_weight="balanced"`

The model is a scikit-learn pipeline using TF-IDF word n-grams, TF-IDF character n-grams, optional metadata features, and Logistic Regression.

Saved training outputs:

- `models/scam_classifier.joblib`
- `models/thresholds.json`
- `models/metrics.json`

Model binaries are ignored by git.

## Imbalance policy

Do not balance the full dataset before splitting. The cleaned dataset is split first with stratified 70/15/15 train/validation/test splits.

Only the training split may be resampled. Validation and test keep the natural class imbalance so metrics reflect production-like moderation traffic. Downsampling validation or test would make false-positive and precision estimates look better than they are.

Supported training imbalance options:

- No resampling, optionally with Logistic Regression `class_weight="balanced"`
- Negative downsampling at 1:1, 3:1, or 5:1 negative-to-positive ratios
- Downsampling with or without class weighting

## Evaluation and thresholds

Accuracy is not the optimization target because scam datasets are imbalanced. A classifier can look accurate by predicting the majority class.

For moderation safety, auto-delete should prioritize:

1. Lowest false-positive rate at the auto-delete threshold
2. Highest precision at the auto-delete threshold
3. Reasonable recall
4. F1 as a secondary tie-breaker

Starting thresholds:

- Auto-delete: `0.90` or `0.95`
- Mod review: `0.75`
- Log only: `0.55`

Medium-confidence messages should go to moderator review instead of being deleted automatically. Critical rule detections are different: they can be deleted immediately because the rule evidence is already high confidence.

## Feedback data

The Hugging Face dataset is only a bootstrap seed. The long-term training set should come from:

- Moderator-confirmed scam messages
- Moderator-confirmed false positives
- Manually added new scam templates
- Future server-specific examples

Never train on the model's own predictions as ground truth. Only train on imported labeled data or human-confirmed labels.

## Run the bot

Create a local `.env` file with the core settings:

```bash
DISCORD_TOKEN=your-token-here
MOD_REVIEW_CHANNEL_ID=your-mod-review-channel-id
WHITELISTED_ROLE_IDS=admin-role-id,moderator-role-id
COMMAND_SYNC_GUILD_ID=your-private-server-id
BOT_DELETE_ENABLED=true
FEEDBACK_DB_PATH=data/feedback.sqlite
```

Run the bot:

```bash
python -m src.scam_detector.bot
```

For a private server, also check Discord setup:

1. In the Discord Developer Portal, enable the bot's privileged **Message Content Intent**.
2. Invite the bot with permissions to view channels, read message history, send messages, and manage messages if you want auto-delete.
3. Put `MOD_REVIEW_CHANNEL_ID` in `.env` or run `/scam setup #mod-alerts` so suspicious messages are visible.
4. Add trusted admin/mod roles with `WHITELISTED_ROLE_IDS` or `/scam trust @Admin`.
5. Restart the bot after changing `.env` or Developer Portal settings.

The bot has two admin-facing modes:

- `Monitor`: alert moderators, but do not delete messages.
- `Protect`: alert moderators and allow critical scam messages to be deleted when permissions allow it.

Critical rule-only confidence can delete immediately when `AUTO_DELETE_CRITICAL=true`. High rule confidence deletes only when `AUTO_DELETE_HIGH=true`.

## Discord commands

Admins with **Manage Server** can configure the bot in Discord:

```text
/scam setup #mod-alerts
/scam mode protect
/scam mode monitor
/scam status
/scam trust @Admin
/scam untrust @Admin
```

Server members can report missed messages from Discord's message app menu:

```text
Right-click message > Apps > Report scam
```

These command settings are currently stored in memory and reset when the bot restarts. The `.env` values still act as startup defaults.

Advanced optional settings:

- `BOT_DELETE_ENABLED=false`: start in Monitor mode so detections are reported but not deleted.
- `BOT_NOTIFY_LOG_ACTIONS=false`: only review/delete events go to the mod channel.
- `EMBEDDING_SIMILARITY_ENABLED=true`: enable optional known-template similarity for medium/uncertain suspicious messages.
- `SCAM_TEMPLATE_PATH`: optional JSON template file. If omitted, anonymized local giveaway templates are used.
- `AUTO_DELETE_CRITICAL`: delete critical rule detections immediately when possible.
- `AUTO_DELETE_HIGH`: delete high rule detections instead of sending them to review.
- `CRITICAL_RULE_SCORE_THRESHOLD`: rule score needed for the critical action band.
- `HIGH_RULE_SCORE_THRESHOLD`: rule score needed for the high action band.
- `MOD_REVIEW_THRESHOLD`: classifier probability threshold for mod review.

To copy a role ID, enable Discord Developer Mode, open **Server Settings > Roles**, right-click the role, and choose **Copy Role ID**. Avoid trusting `@everyone`, because that bypasses detection for the whole server.

For local testing, set `COMMAND_SYNC_GUILD_ID` to your server ID. Without it, commands are synced globally and may take longer to appear in Discord.
