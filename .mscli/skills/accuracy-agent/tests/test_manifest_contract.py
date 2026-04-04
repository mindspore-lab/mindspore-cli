from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = SKILL_ROOT / "skill.yaml"
SKILL = SKILL_ROOT / "SKILL.md"


def _manifest_text() -> str:
    return MANIFEST.read_text(encoding="utf-8")


def test_manifest_contract_fields_present():
    text = _manifest_text()
    assert 'name: "accuracy-agent"' in text
    assert 'display_name: "Accuracy Agent"' in text
    assert 'version: "0.2.0"' in text
    assert 'type: "manual"' in text
    assert 'path: "SKILL.md"' in text
    assert 'network: "none"' in text
    assert 'filesystem: "workspace-write"' in text


def test_manifest_declares_accuracy_inputs_and_outputs():
    text = _manifest_text()
    assert 'name: "user_problem"' in text
    assert 'name: "baseline_path"' in text
    assert 'name: "current_result_path"' in text
    assert 'name: "env_lock"' in text
    assert 'name: "factory_root"' in text
    assert 'report_schema' in text
    assert 'out_dir_layout' in text


def test_skill_describes_four_stage_accuracy_workflow():
    text = SKILL.read_text(encoding="utf-8")
    assert "# Accuracy Agent" in text
    assert "1. `accuracy-analyzer`" in text
    assert "2. `consistency-validator`" in text
    assert "3. `snapshot-builder`" in text
    assert "4. `report-builder`" in text
