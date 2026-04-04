#!/usr/bin/env python3
"""Emit a placeholder feature-spec summary."""

from __future__ import annotations

import json
import sys


def main() -> int:
    payload = {
        "summary": "feature-spec placeholder",
        "inputs": sys.argv[1:],
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
