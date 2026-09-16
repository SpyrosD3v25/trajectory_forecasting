#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Production launch is verification-only: drift aborts the batch. Repair is a
# separate, explicit operator action and never part of experiment execution.
python scripts/preflight_compact_data.py

created=()
for yaml in "$REPO_ROOT"/experiments/preprocessing_ablation/*.yaml; do
  output="$(python scripts/train.py --config "${yaml#"$REPO_ROOT"/}" "$@")"
  created+=("$output")
done

printf '%s\n' "${created[@]}"
