#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

python scripts/preflight_compact_data.py

# Only the manifest-backed canonical experiment is production-valid.  The v1
# file is retained as a historical record and intentionally is not launched.
python scripts/train.py --config experiments/experiment_1/kineroute_paper.yaml "$@"
