import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
READINESS_VERDICT_REF = Path("meta/readiness-verdict.json")


def test_shared_envelope_and_readiness_verdict_validate_against_their_schemas(tmp_path: Path):
    jsonschema = pytest.importorskip("jsonschema")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "infer.py").write_text("print('hello')\n", encoding="utf-8")
    (workspace / "model").mkdir()

    report_dir = tmp_path / "out"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "run_readiness_pipeline.py"),
            "--working-dir",
            str(workspace),
            "--output-dir",
            str(report_dir),
            "--target",
            "inference",
            "--model-path",
            "model",
            "--check",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert completed.stdout
    shared_schema = json.loads((ROOT.parent / "_shared" / "contract" / "report.schema.json").read_text(encoding="utf-8"))
    verdict_schema = json.loads((ROOT / "contract" / "readiness-verdict.schema.json").read_text(encoding="utf-8"))
    report = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    verdict = json.loads((report_dir / READINESS_VERDICT_REF).read_text(encoding="utf-8"))

    jsonschema.validate(instance=report, schema=shared_schema)
    jsonschema.validate(instance=verdict, schema=verdict_schema)
