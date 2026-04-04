from pathlib import Path


SKILL_MD = Path(__file__).resolve().parents[1] / "SKILL.md"


def test_behavior_rules_require_baseline_and_divergence_reasoning():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "Establish a comparable baseline before making root-cause claims." in text
    assert "Find the earliest meaningful divergence before suggesting fixes." in text
    assert "If there is no trusted baseline, say so explicitly" in text
    assert "Do not claim a fix is confirmed until the user verifies it." in text
    assert "Do not skip workflow stages." in text
    assert "one confirmed accuracy issue per invocation" in text


def test_accuracy_profile_and_consistency_validation_are_present():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "Build an `AccuracyProfile`" in text
    assert "Return ranked root-cause candidates with:" in text
    assert "- dtype, precision, API parameter, and device-placement consistency" in text
    assert "- api parameters" in text
    assert "- device placement" in text
    assert "- confidence" in text
    assert "- evidence" in text
    assert "- validation checks" in text
    assert "- fix hints" in text


def test_references_and_scripts_are_declared():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "`references/consistency-validation.md`" in text
    assert "`references/debug-script-hygiene.md`" in text
    assert "`references/operator-accuracy-triage.md`" in text
    assert "`scripts/collect_accuracy_context.py`" in text
    assert "`scripts/summarize_metric_diff.py`" in text


def test_debug_script_hygiene_reference_covers_generic_stack_and_determinism():
    text = (SKILL_MD.parent / "references" / "debug-script-hygiene.md").read_text(
        encoding="utf-8"
    )
    assert "backend-specific extension package" in text
    assert "determin" in text
    assert "identity or no-op" in text


def test_operator_triage_reference_tracks_official_api_mapping():
    text = (SKILL_MD.parent / "references" / "operator-accuracy-triage.md").read_text(
        encoding="utf-8"
    )
    assert "https://www.mindspore.cn/docs/zh-CN/master/note/api_mapping/pytorch_api_mapping.html" in text
    assert "official PyTorch-to-MindSpore API" in text


def test_collect_accuracy_context_tracks_framework_versions():
    text = (SKILL_MD.parent / "scripts" / "collect_accuracy_context.py").read_text(
        encoding="utf-8"
    )
    assert "framework_versions" in text
    assert "torch_npu" in text
    assert "HCCL_DETERMINISTIC" in text
