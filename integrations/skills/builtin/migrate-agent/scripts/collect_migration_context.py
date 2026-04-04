#!/usr/bin/env python3
"""Collect a minimal migration-context snapshot."""

from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> None:
    cwd = Path(os.getcwd()).resolve()
    data = {
        "working_dir": str(cwd),
        "files": sorted(p.name for p in cwd.iterdir())[:80] if cwd.exists() else [],
    }
    print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
