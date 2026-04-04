import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _fake_python_source() -> str:
    real_python = json.dumps(sys.executable)
    return f"""#!/usr/bin/env python3
import json
import subprocess
import sys

REAL_PYTHON = {real_python}

if len(sys.argv) >= 3 and sys.argv[1] == "-c":
    code = sys.argv[2]
    if "platform.python_version" in code and "version_info" in code:
        print(json.dumps({{"version_info": [3, 10, 0], "version": "3.10.0"}}))
        raise SystemExit(0)
    if "importlib.util" in code and len(sys.argv) >= 5:
        mode = sys.argv[3]
        payload = json.loads(sys.argv[4])
        if mode == "import":
            packages = payload.get("packages", [])
            print(json.dumps({{name: True for name in packages}}))
            raise SystemExit(0)
        if mode == "package_versions":
            packages = payload.get("packages", [])
            print(json.dumps({{"versions": {{name: "1.0.0" for name in packages}}, "errors": {{}}}}))
            raise SystemExit(0)
        if mode == "framework_smoke":
            print(json.dumps({{"success": True, "details": ["fake framework smoke ok"], "error": None}}))
            raise SystemExit(0)
    completed = subprocess.run([REAL_PYTHON, *sys.argv[1:]])
    raise SystemExit(completed.returncode)

if len(sys.argv) >= 3 and sys.argv[1] == "-m" and sys.argv[2] == "py_compile":
    raise SystemExit(0)

completed = subprocess.run([REAL_PYTHON, *sys.argv[1:]])
raise SystemExit(completed.returncode)
"""


def _make_fake_selected_python(tmp_path: Path) -> Path:
    script = tmp_path / "fake-python.py"
    script.write_text(_fake_python_source(), encoding="utf-8")
    if os.name == "nt":
        launcher = tmp_path / "fake-python.cmd"
        launcher.write_text(f'@echo off\r\n"{sys.executable}" "%~dp0fake-python.py" %*\r\n', encoding="utf-8")
        return launcher
    script.chmod(script.stat().st_mode | 0o111)
    return script


@pytest.fixture
def fake_selected_python(tmp_path: Path) -> Path:
    return _make_fake_selected_python(tmp_path)
