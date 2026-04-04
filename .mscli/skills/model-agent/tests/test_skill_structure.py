from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
SKILL_MD = SKILL_ROOT / "SKILL.md"


def test_skill_markers_present():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "## Workflow" in text
    assert "## Stage 1. Migration Analyzer" in text
    assert "## Stage 2. Route Selector" in text
    assert "## Stage 3. Migration Builder" in text
    assert "## Stage 4. Verification and Report" in text


def test_reference_and_script_files_exist():
    assert (SKILL_ROOT / "references" / "migration-routing.md").exists()
    assert (SKILL_ROOT / "references" / "verification.md").exists()
    assert (SKILL_ROOT / "references" / "hf-transformers-guardrails.md").exists()
    assert (SKILL_ROOT / "references" / "hf-transformers-env.md").exists()
    assert (SKILL_ROOT / "scripts" / "collect_migration_context.py").exists()
    assert (SKILL_ROOT / "scripts" / "summarize_migration_profile.py").exists()
    assert (SKILL_ROOT / "scripts" / "hf_transformers_auto_convert.py").exists()
    assert (SKILL_ROOT / "scripts" / "hf_transformers_auto_convert.requirements.txt").exists()
