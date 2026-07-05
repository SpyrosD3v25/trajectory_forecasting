#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

created=()
for yaml in "$REPO_ROOT"/experiments/baselines/*.yaml; do
  model_name="$(python - <<PY
import yaml
from pathlib import Path
payload = yaml.safe_load(Path(r"$yaml").read_text()) or {}
print(payload["model"]["name"])
PY
)"
  if [[ "$model_name" == "kineroute_nvp" ]]; then
    output="$(python scripts/train.py --config "${yaml#"$REPO_ROOT"/}")"
  else
    output="$(python scripts/train_baseline.py --config "${yaml#"$REPO_ROOT"/}")"
  fi
  created+=("$output")
done

