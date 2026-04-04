#!/usr/bin/env python3
"""Collect basic feature-adaptation context."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    payload = {
        "cwd": str(Path.cwd()),
        "argv": sys.argv[1:],
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
