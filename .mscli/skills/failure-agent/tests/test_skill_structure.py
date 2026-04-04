from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
SKILL_MD = SKILL_ROOT / "SKILL.md"


def test_skill_markers_present():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "## Workflow" in text
    assert "## Stage 1. Failure Analyzer" in text
    assert "## Stage 2. Root-Cause Validator" in text
    assert "## Stage 3. Snapshot Builder" in text
    assert "## Stage 4. Report Builder" in text


def test_reference_and_script_files_exist():
    assert (SKILL_ROOT / "reference" / "failure-taxonomy.md").exists()
    assert (SKILL_ROOT / "reference" / "root-cause-validation.md").exists()
    assert (SKILL_ROOT / "scripts" / "collect_failure_context.py").exists()
    assert (SKILL_ROOT / "scripts" / "summarize_traceback.py").exists()
