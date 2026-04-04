from pathlib import Path


SKILL_MD = Path(__file__).resolve().parents[1] / "SKILL.md"


def test_behavior_rules_require_evidence_and_validation():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "Collect evidence before diagnosis." in text
    assert "Prefer the first real failure point over downstream noise." in text
    assert "Do not claim a fix is confirmed until the user verifies it." in text
    assert "Do not auto-edit code, configs, or the environment in this skill." in text
    assert "Do not auto-submit or mutate Factory content." in text


def test_failure_profile_and_root_cause_validation_are_present():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "Build a `FailureProfile`" in text
    assert "Return ranked root-cause candidates with:" in text
    assert "- confidence" in text
    assert "- evidence" in text
    assert "- validation checks" in text
    assert "- fix hints" in text


def test_references_and_scripts_are_declared():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "`reference/failure-taxonomy.md`" in text
    assert "`reference/root-cause-validation.md`" in text
    assert "`scripts/collect_failure_context.py`" in text
    assert "`scripts/summarize_traceback.py`" in text
