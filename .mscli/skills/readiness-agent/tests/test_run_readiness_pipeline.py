import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
READINESS_VERDICT_REF = Path("meta/readiness-verdict.json")


def run_pipeline(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "run_readiness_pipeline.py"), *args],
        cwd=str(cwd),
        check=True,
        text=True,
        capture_output=True,
    )


def load_report_pair(report_json: Path) -> tuple[dict, dict]:
    envelope = json.loads(report_json.read_text(encoding="utf-8"))
    verdict_json = report_json.parent / READINESS_VERDICT_REF
    verdict = json.loads(verdict_json.read_text(encoding="utf-8"))
    return envelope, verdict


def fake_uv_source() -> str:
    return f"""#!/usr/bin/env python3
import os
import shutil
import sys
from pathlib import Path

REAL_PYTHON = r'''{sys.executable}'''

def main() -> int:
    args = sys.argv[1:]
    if not args:
        return 1
    if args[0] == "venv":
        env_root = None
        for item in args[1:]:
            if item.startswith("-"):
                continue
            env_root = Path(item)
            break
        if env_root is None:
            return 2
        target = env_root / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REAL_PYTHON, target)
        return 0
    if len(args) >= 2 and args[0] == "pip" and args[1] == "install":
        return 0
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
"""


def install_fake_uv(tmp_path: Path, monkeypatch) -> Path:
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()

    uv_py = bin_dir / "uv"
    uv_py.write_text(fake_uv_source(), encoding="utf-8")
    uv_py.chmod(uv_py.stat().st_mode | 0o111)

    uv_cmd = bin_dir / "uv.cmd"
    uv_cmd.write_text(f'@echo off\r\n"{sys.executable}" "%~dp0uv" %*\r\n', encoding="utf-8")
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))
    return bin_dir


def make_workspace(tmp_path: Path, script_name: str = "infer.py", body: str = "print('infer')\n") -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / script_name).write_text(body, encoding="utf-8")
    (workspace / "model").mkdir()
    return workspace


def test_run_readiness_pipeline_check_blocks_without_workspace_env(tmp_path: Path):
    workspace = make_workspace(tmp_path)
    output_dir = tmp_path / "out"

    run_pipeline(
        "--working-dir",
        str(workspace),
        "--output-dir",
        str(output_dir),
        "--target",
        "inference",
        "--model-path",
        "model",
        "--check",
        cwd=workspace,
    )

    _, verdict = load_report_pair(output_dir / "report.json")
    fix_applied = verdict["fix_applied"]
    readiness_env = (workspace / ".readiness.env").read_text(encoding="utf-8")

    assert verdict["status"] == "BLOCKED"
    assert verdict["can_run"] is False
    assert fix_applied["execute"] is False
    assert fix_applied["planned_actions"]
    assert "READINESS_WORKING_DIR" in readiness_env


def test_run_readiness_pipeline_ready_uses_runtime_smoke_and_prompts_to_run_model_script(tmp_path: Path, fake_selected_python: Path):
    workspace = make_workspace(
        tmp_path,
        body="import torch\nimport torch_npu\nimport transformers\nprint('infer')\n",
    )
    output_dir = tmp_path / "out"

    run_pipeline(
        "--working-dir",
        str(workspace),
        "--output-dir",
        str(output_dir),
        "--target",
        "inference",
        "--framework-hint",
        "pta",
        "--selected-python",
        str(fake_selected_python),
        "--model-path",
        "model",
        "--check",
        cwd=workspace,
    )

    _, verdict = load_report_pair(output_dir / "report.json")
    assert verdict["status"] == "READY"
    assert verdict["can_run"] is True
    assert verdict["evidence_level"] == "runtime_smoke"
    assert "Do you want me to run the real model script now?" in verdict["next_action"]


def test_run_readiness_pipeline_warns_on_unmapped_cann_and_still_prompts_to_run_model_script(tmp_path: Path, fake_selected_python: Path):
    workspace = make_workspace(
        tmp_path,
        body="import torch\nimport torch_npu\nimport transformers\nprint('infer')\n",
    )
    output_dir = tmp_path / "out"

    run_pipeline(
        "--working-dir",
        str(workspace),
        "--output-dir",
        str(output_dir),
        "--target",
        "inference",
        "--framework-hint",
        "pta",
        "--selected-python",
        str(fake_selected_python),
        "--model-path",
        "model",
        "--cann-path",
        str(tmp_path / "custom-cann-9.9.9"),
        "--check",
        cwd=workspace,
    )

    _, verdict = load_report_pair(output_dir / "report.json")
    assert verdict["status"] == "WARN"
    assert "Do you want me to run the real model script now?" in verdict["next_action"]
    assert any(item["id"] == "framework-compatibility" for item in verdict["warnings_detailed"])


def test_run_readiness_pipeline_fix_creates_default_env_and_reruns(tmp_path: Path, monkeypatch):
    install_fake_uv(tmp_path, monkeypatch)
    workspace = make_workspace(tmp_path)
    output_dir = tmp_path / "out"

    run_pipeline(
        "--working-dir",
        str(workspace),
        "--output-dir",
        str(output_dir),
        "--target",
        "inference",
        "--model-path",
        "model",
        "--fix",
        cwd=workspace,
    )

    _, verdict = load_report_pair(output_dir / "report.json")
    env_json = json.loads((output_dir / "meta" / "env.json").read_text(encoding="utf-8"))

    assert (workspace / ".venv").exists()
    assert verdict["status"] == "WARN"
    assert verdict["can_run"] is True
    assert "Do you want me to run the real model script now?" in verdict["next_action"]
    assert env_json["pipeline_passes"] == 2
    assert "create-workspace-env" in verdict["fix_applied"]["executed_actions"]


def test_run_readiness_pipeline_tolerates_missing_and_unknown_cli_args(tmp_path: Path):
    workspace = make_workspace(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "run_readiness_pipeline.py"),
            "--check",
            "--verbose",
            "--unknown-flag",
            "mystery",
            "--model-path",
        ],
        cwd=str(workspace),
        check=True,
        text=True,
        capture_output=True,
    )

    summary = json.loads(completed.stdout)
    inputs = json.loads((workspace / "readiness-output" / "meta" / "inputs.json").read_text(encoding="utf-8"))

    assert summary["status"] == "BLOCKED"
    assert inputs["ignored_cli_args"] == [
        {"token": "--unknown-flag", "reason": "unknown_flag"},
        {"token": "mystery", "reason": "unknown_flag_value"},
        {"token": "--model-path", "reason": "missing_value"},
    ]


def test_run_readiness_pipeline_rejects_removed_auto_mode(tmp_path: Path):
    workspace = make_workspace(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "run_readiness_pipeline.py"),
            "--auto",
        ],
        cwd=str(workspace),
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert json.loads(completed.stderr) == {
        "error": "auto mode was removed; use --fix for readiness remediation."
    }
