#!/usr/bin/env python3
"""Print a compact traceback summary from stdin or a file path."""

from __future__ import annotations

import sys
from pathlib import Path


def _read_text(argv: list[str]) -> str:
    if len(argv) > 1:
        return Path(argv[1]).read_text(encoding="utf-8")
    return sys.stdin.read()


def main(argv: list[str]) -> None:
    text = _read_text(argv).strip().splitlines()
    for line in text[:30]:
        print(line)


if __name__ == "__main__":
    main(sys.argv)
