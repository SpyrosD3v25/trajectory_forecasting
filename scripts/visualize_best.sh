#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -eq 0 ]]; then
  printf 'Usage: %s RESULT_DIR_OR_CHECKPOINT [...]\n' "$0" >&2
  exit 2
fi

for supplied in "$@"; do
  checkpoint="$supplied"
  if [[ -d "$checkpoint" ]]; then
    checkpoint="$checkpoint/checkpoints/best.pt"
  fi
  if [[ ! -f "$checkpoint" ]]; then
    printf 'Warning: checkpoint not found, skipping: %s\n' "$checkpoint" >&2
    continue
  fi
  python scripts/visualize_trajectories_from_pt.py --pt "$checkpoint" --n 20 --random --seed 42
done
