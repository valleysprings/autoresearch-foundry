from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.bench.livecodebench_prepare import prepare_livecodebench_collection


ROOT = Path(__file__).resolve().parent


def main() -> None:
    prepare_livecodebench_collection(
        task_root=ROOT,
        task_id="livecodebench",
        releases=[
            ("v1", 400),
            ("v2", 111),
            ("v3", 101),
            ("v4", 101),
            ("v5", 167),
            ("v6", 175),
        ],
    )


if __name__ == "__main__":
    main()
