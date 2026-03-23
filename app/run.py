from __future__ import annotations

import sys

from app.entries.runner import main as runner_main
from app.entries.server import main as server_main


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "serve":
        server_main(args[1:])
        return
    runner_main(args)


if __name__ == "__main__":
    main()
