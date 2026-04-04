import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_test_utils import write_sample_profiler_export, write_validation_metrics


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
VERDICT_REF = Path("meta/performance-verdict.json")


def run_script(script_name: str, *args: str) -> subprocess.CompletedProcess[str]:
    script = SCRIPTS / script_name
    return subprocess.run(
        [sys.executable, str(script), *args],
        check=True,
        text=True,
        capture_output=True,
    )


def test_shared_envelope_and_performance_verdict_validate_against_their_schemas(tmp_path: Path):
    jsonschema = pytest.importorskip("jsonschema")

    profiler_root = write_sample_profiler_export(tmp_path)
    before_json, after_json = write_validation_metrics(tmp_path)

    locate_json = tmp_path / "locate.json"
    step_json = tmp_path / "step.json"
    comm_json = tmp_path / "comm.json"
    memory_json = tmp_path / "memory.json"
    input_json = tmp_path / "input.json"
    trace_gaps_json = tmp_path / "trace-gaps.json"
    hotspot_json = tmp_path / "hotspot-summary.json"
    profile_json = tmp_path / "profile.json"
    bottlenecks_json = tmp_path / "bottlenecks.json"
    validation_json = tmp_path / "validation.json"
    report_json = tmp_path / "out" / "report.json"
    report_md = tmp_path / "out" / "report.md"

    run_script("locate_profiler_output.py", "--trace-path", str(profiler_root), "--output-json", str(locate_json))
    run_script("summarize_step_breakdown.py", "--trace-root", str(profiler_root), "--output-json", str(step_json))
    run_script("summarize_communication.py", "--trace-root", str(profiler_root), "--output-json", str(comm_json))
    run_script("summarize_memory_pressure.py", "--trace-root", str(profiler_root), "--output-json", str(memory_json))
    run_script("summarize_input_pipeline.py", "--trace-root", str(profiler_root), "--output-json", str(input_json))
    run_script("summarize_trace_gaps.py", "--trace-root", str(profiler_root), "--output-json", str(trace_gaps_json))
    run_script(
        "summarize_msprof_hotspots.py",
        "--input-dir",
        str(profiler_root),
        "--output-md",
        str(tmp_path / "hotspot-summary.md"),
        "--output-json",
        str(hotspot_json),
    )
    run_script(
        "build_performance_profile.py",
        "--working-dir",
        str(tmp_path),
        "--user-problem",
        "Memory pressure and communication overhead are both visible, but communication is dominant.",
        "--locate-json",
        str(locate_json),
        "--step-json",
        str(step_json),
        "--communication-json",
        str(comm_json),
        "--memory-json",
        str(memory_json),
        "--input-json",
        str(input_json),
        "--trace-gaps-json",
        str(trace_gaps_json),
        "--hotspot-json",
        str(hotspot_json),
        "--output-json",
        str(profile_json),
    )
    run_script(
        "classify_bottlenecks.py",
        "--profile-json",
        str(profile_json),
        "--step-json",
        str(step_json),
        "--communication-json",
        str(comm_json),
        "--memory-json",
        str(memory_json),
        "--input-json",
        str(input_json),
        "--trace-gaps-json",
        str(trace_gaps_json),
        "--hotspot-json",
        str(hotspot_json),
        "--output-json",
        str(bottlenecks_json),
    )
    run_script(
        "compare_validation_metrics.py",
        "--before-json",
        str(before_json),
        "--after-json",
        str(after_json),
        "--output-json",
        str(validation_json),
    )
    run_script(
        "build_performance_report.py",
        "--working-dir",
        str(tmp_path),
        "--user-problem",
        "Memory pressure and communication overhead are both visible, but communication is dominant.",
        "--locate-json",
        str(locate_json),
        "--profile-json",
        str(profile_json),
        "--bottlenecks-json",
        str(bottlenecks_json),
        "--validation-json",
        str(validation_json),
        "--output-json",
        str(report_json),
        "--output-md",
        str(report_md),
    )

    shared_schema = json.loads((ROOT.parent / "_shared" / "contract" / "report.schema.json").read_text(encoding="utf-8"))
    verdict_schema = json.loads((ROOT / "contract" / "performance-verdict.schema.json").read_text(encoding="utf-8"))
    report = json.loads(report_json.read_text(encoding="utf-8"))
    verdict = json.loads((tmp_path / "out" / VERDICT_REF).read_text(encoding="utf-8"))

    jsonschema.validate(instance=report, schema=shared_schema)
    jsonschema.validate(instance=verdict, schema=verdict_schema)
