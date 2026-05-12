# Discord Scam Detection Bot

Python Discord moderation bot for tiered scam detection.

The project is built around a cheap-first detection pipeline:

1. Ignore ineligible messages.
2. Run lightweight trigger screening.
3. Score suspicious messages with deterministic rules.
4. Optionally compare uncertain suspicious messages against known scam-template embeddings.
5. Optionally call an ML classifier only for suspicious messages.
6. Decide whether to allow, log, send to review, or delete.
7. Store moderator-confirmed labels for future training.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

## Architecture

The runtime path intentionally avoids expensive ML on ordinary chat traffic:

- Tier 0 eligibility: empty messages, bot authors, and non-guild messages are ignored.
- Tier 1 screening: cheap keyword, link, and mention checks run on eligible messages.
- Tier 2 scoring: suspicious messages receive deterministic risk scoring from content and metadata.
- Tier 3 embedding similarity: optional known-template similarity runs only for cheap-screened medium/uncertain messages.
- Tier 4 classifier: the optional sklearn model is called only for uncertain messages that still need help after rules and embeddings.
- Decision engine: low risk is allowed, medium confidence can be logged or reviewed, and only high-confidence classifier output can auto-delete.
- Feedback loop: moderator-confirmed scam and false-positive labels are stored for future training.

Rules are the primary safety layer. They are deterministic, explainable, and catch obvious scam templates without needing ML. Embedding similarity is a secondary optional layer for wording changes that are close to known giveaway scams. The trained classifier is a separate optional model trained from labeled data.

Embeddings do not replace rules and are not run on every message. The bot only calls embedding similarity after cheap screening has triggered and the rule score is still low/medium. Critical or high rule scores skip embeddings and classifier calls because the rule evidence is already strong. This keeps normal chat cheap and avoids using semantic similarity as a broad surveillance step.

## Dataset

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

The model is a sklearn pipeline using TF-IDF word n-grams, TF-IDF character n-grams, optional metadata features, and Logistic Regression.

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

Medium-confidence messages should go to moderator review instead of being deleted automatically.

## Feedback data

The Hugging Face dataset is only a bootstrap seed. The long-term training set should come from:

- Moderator-confirmed scam messages
- Moderator-confirmed false positives
- Manually added new scam templates
- Future server-specific examples

Never train on the model's own predictions as ground truth. Only train on imported labeled data or human-confirmed labels.

## Bot skeleton

Create a local `.env` file:

```bash
DISCORD_TOKEN=your-token-here
MOD_REVIEW_CHANNEL_ID=your-mod-review-channel-id
WHITELISTED_ROLE_IDS=admin-role-id,moderator-role-id
COMMAND_SYNC_GUILD_ID=your-private-server-id
EMBEDDING_SIMILARITY_ENABLED=false
# SCAM_TEMPLATE_PATH=optional/path/to/templates.json
```

Run the bot:

```bash
python -m src.scam_detector.bot
```

Optional bot settings:

- `MOD_REVIEW_CHANNEL_ID`: channel where log/review/delete events are posted.
- `WHITELISTED_ROLE_IDS`: comma-separated default role IDs that bypass scam detection.
- `BOT_DELETE_ENABLED=false`: dry-run mode; detections are reported but messages are not deleted.
- `BOT_NOTIFY_LOG_ACTIONS=false`: only review/delete events go to the mod channel.
- `COMMAND_SYNC_GUILD_ID`: sync slash commands to one server immediately during testing.
- `EMBEDDING_SIMILARITY_ENABLED=true`: enable optional known-template similarity for medium/uncertain suspicious messages.
- `SCAM_TEMPLATE_PATH`: optional JSON template file. If omitted, anonymized local giveaway templates are used.

For a private server, also check Discord configuration:

1. In the Discord Developer Portal, enable the bot's privileged **Message Content Intent**.
2. Invite the bot with permissions to view channels, read message history, send messages, and manage messages if you want auto-delete.
3. Put `MOD_REVIEW_CHANNEL_ID` in `.env` while testing so suspicious messages are visible even when they are not deleted.
4. Add trusted admin/mod roles to `WHITELISTED_ROLE_IDS` so their messages bypass detection.
5. Restart the bot after changing `.env` or Developer Portal settings.

Most test scam messages will not auto-delete immediately. Medium confidence becomes `log`, high rule-only confidence becomes `review`, and auto-delete requires a classifier probability above the auto-delete threshold.

Embedding similarity is disabled by default. If enabled, missing template files or optional embedding backends fail closed: the bot logs the unavailable layer and continues with rule/classifier behavior.

To copy a role ID, enable Discord Developer Mode, open **Server Settings > Roles**, right-click the role, and choose **Copy Role ID**. Avoid whitelisting `@everyone`, because that bypasses detection for the whole server.

## Server configuration commands

Admins with **Manage Server** can configure the bot in Discord:

```text
/scam-config review-channel #mod-alerts
/scam-config delete-enabled false
/scam-config whitelist-role add @Admin
/scam-config whitelist-role remove @Admin
/scam-config whitelist-role list
```

These command settings are currently stored in memory and reset when the bot restarts. The `.env` values still act as startup defaults. Persistent per-server storage is the next step.

For local testing, set `COMMAND_SYNC_GUILD_ID` to your server ID. Without it, commands are synced globally and may take longer to appear in Discord.
