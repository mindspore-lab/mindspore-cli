from pathlib import Path


SKILL_MD = Path(__file__).resolve().parents[1] / "SKILL.md"


def test_workflow_stages_are_present():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "1. `feature-analyzer`" in text
    assert "2. `integration-planner`" in text
    assert "3. `patch-builder`" in text
    assert "4. `readiness-handoff-and-report`" in text


def test_route_selection_and_mhc_pack_are_declared():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "Choose exactly one integration route:" in text
    assert "- `generic-feature`" in text
    assert "- `mhc`" in text
    assert "- `attnres`" in text
    assert "`integration_route`" in text
    assert "`route_evidence`" in text
    assert "`references/mhc/mhc-implementation-pattern.md`" in text
    assert "`references/mhc/mhc-validation-checklist.md`" in text
    assert "`references/mhc/mhc-qwen3-case-study.md`" in text
    assert "`references/attnres/attnres-implementation-pattern.md`" in text
    assert "`references/attnres/attnres-validation-checklist.md`" in text
    assert "`references/attnres/attnres-qwen3-case-study.md`" in text


def test_algorithm_agent_remains_top_level_entry():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "This skill is the top-level algorithm feature entry." in text
    assert "The user should not need to choose up front" in text
    assert "Do not turn route selection into a fifth workflow stage." in text
