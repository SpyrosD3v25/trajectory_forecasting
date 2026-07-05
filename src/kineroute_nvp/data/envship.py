from __future__ import annotations

import json
from pathlib import Path


def load_processed_metadata(processed_root: Path) -> dict:
    return json.loads((Path(processed_root) / "metadata.json").read_text())
