from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    if not argv:
        raise SystemExit("Usage: python scripts/evaluate.py <result_dir>")
    result_dir = REPO_ROOT / argv[0]
    metrics = json.loads((result_dir / "evaluation" / "metrics.json").read_text())
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
