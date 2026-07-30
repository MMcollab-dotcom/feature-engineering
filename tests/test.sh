#!/bin/sh
set -eu

mkdir -p /logs/verifier
test -f /app/submission/feature_engineering/submissions/*/manifest.json
feature-engineering-verify \
  --task-config /app/task_config.yaml \
  --task-root /app \
  --submission-root /app/submission \
  --reward-path /logs/verifier/reward.json \
  --metrics-path /logs/verifier/metrics.json
test -f /logs/verifier/reward.json
python - <<'PY'
import json
from pathlib import Path

reward = json.loads(Path("/logs/verifier/reward.json").read_text())
required = {
    "primary_score",
    "reward",
    "sharpe",
    "cagr",
    "max_drawdown",
    "pearson_ic",
}
missing = sorted(required.difference(reward))
if missing:
    raise SystemExit(f"Missing reward fields: {', '.join(missing)}")
if not all(isinstance(reward[name], (int, float)) for name in required):
    raise SystemExit("All reward fields must be numeric.")
PY
