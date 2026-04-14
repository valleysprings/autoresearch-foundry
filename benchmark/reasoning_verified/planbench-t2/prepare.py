from __future__ import annotations

import argparse
from pathlib import Path

from app.bench.planbench_prepare import write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize PlanBench Task 2 optimal-planning rows.")
    parser.add_argument("--output", default=str(Path(__file__).with_name("data") / "questions.json"))
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    count = write_manifest("planbench-t2", Path(args.output), split=args.split)
    print(f"Wrote {count} PlanBench Task 2 rows to {args.output}.")


if __name__ == "__main__":
    main()
