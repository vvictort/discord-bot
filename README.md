# Discord Scam Detection Bot

Python Discord moderation bot for tiered scam detection.

The project is built around a cheap-first detection pipeline:

1. Ignore ineligible messages.
2. Run lightweight trigger screening.
3. Score suspicious messages with deterministic rules.
4. Optionally call an ML classifier only for suspicious messages.
5. Decide whether to allow, log, send to review, or delete.
6. Store moderator-confirmed labels for future training.
