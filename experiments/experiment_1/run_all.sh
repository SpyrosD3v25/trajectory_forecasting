#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

created=()
for yaml in "$REPO_ROOT"/experiments/experiment_1/*.yaml; do
  output="$(python scripts/train.py --config "${yaml#"$REPO_ROOT"/}")"
  created+=("$output")
done

printf '%s\n' "${created[@]}"
