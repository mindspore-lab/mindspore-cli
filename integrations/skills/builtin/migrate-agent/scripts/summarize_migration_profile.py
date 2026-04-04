#!/usr/bin/env python3
"""Print a compact migration profile placeholder from CLI inputs."""

from __future__ import annotations

import json
import sys


def main(argv: list[str]) -> None:
    data = {
        "source_repo": argv[1] if len(argv) > 1 else "",
        "route": argv[2] if len(argv) > 2 else "",
        "goal": argv[3] if len(argv) > 3 else "",
    }
    print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv)
