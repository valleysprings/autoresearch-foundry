from __future__ import annotations

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "planning_verified" / "planbench" / "prepare.py"),
        run_name="__main__",
    )
