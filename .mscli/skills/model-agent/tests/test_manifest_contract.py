from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = SKILL_ROOT / "skill.yaml"
SKILL = SKILL_ROOT / "SKILL.md"


def _manifest_text() -> str:
    return MANIFEST.read_text(encoding="utf-8")


def test_manifest_contract_fields_present():
    text = _manifest_text()
    assert 'name: "model-agent"' in text
    assert 'display_name: "Model Migrate"' in text
    assert 'version: "0.2.0"' in text
    assert 'type: "manual"' in text
    assert 'path: "SKILL.md"' in text
    assert 'network: "none"' in text
    assert 'filesystem: "workspace-write"' in text


def test_manifest_declares_route_inputs_and_outputs():
    text = _manifest_text()
    assert 'name: "route_preference"' in text
    assert 'choices: ["hf-transformers", "hf-diffusers", "generic-pytorch-repo"]' in text
    assert 'name: "migration_goal"' in text
    assert 'report_schema' in text
    assert 'out_dir_layout' in text


def test_skill_describes_four_stage_migration_workflow():
    text = SKILL.read_text(encoding="utf-8")
    assert "# Model Agent" in text
    assert "1. `migration-analyzer`" in text
    assert "2. `route-selector`" in text
    assert "3. `migration-builder`" in text
    assert "4. `verification-and-report`" in text
