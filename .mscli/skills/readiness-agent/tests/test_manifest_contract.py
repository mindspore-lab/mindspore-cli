from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = SKILL_ROOT / "skill.yaml"
SKILL = SKILL_ROOT / "SKILL.md"


def _manifest_text() -> str:
    return MANIFEST.read_text(encoding="utf-8")


def test_manifest_contract_fields_present():
    text = _manifest_text()
    assert 'name: "readiness-agent"' in text
    assert 'display_name: "Readiness Agent"' in text
    assert 'version: "0.2.0"' in text
    assert 'type: "manual"' in text
    assert 'path: "SKILL.md"' in text
    assert 'network: "optional"' in text
    assert 'filesystem: "workspace-write"' in text


def test_manifest_keeps_core_inputs_and_drops_low_value_ones():
    text = _manifest_text()
    for token in (
        'name: "working_dir"',
        'name: "target"',
        'choices: ["training", "inference", "auto"]',
        'name: "framework_hint"',
        'choices: ["mindspore", "pta", "mixed", "auto"]',
        'name: "cann_path"',
        'name: "mode"',
        'choices: ["check", "fix"]',
        'name: "selected_python"',
        'name: "model_hub_id"',
        'name: "dataset_hub_id"',
        'name: "dataset_split"',
        'name: "task_smoke_cmd"',
        'name: "allow_network"',
    ):
        assert token in text
    for removed in ('name: "selected_env_root"', 'name: "fix_scope"', 'name: "factory_root"'):
        assert removed not in text


def test_skill_describes_streamlined_runtime_smoke_workflow():
    text = SKILL.read_text(encoding="utf-8")
    assert text.startswith("---\nname: readiness-agent\ndescription:")
    assert "Use when Codex needs" not in text
    assert "# Readiness Agent" in text
    assert "## Scope" in text
    assert "## Hard Rules" in text
    assert "## Workflow" in text
    assert "## References" in text
    assert "## Scripts" in text
    assert "runtime_smoke" in text
    assert "`scripts/run_readiness_pipeline.py`" in text
    assert "`scripts/readiness_core.py`" not in text
    assert "`scripts/readiness_report.py`" not in text
    assert "`scripts/ascend_compat.py`" not in text
    assert "Do you want me to run the real model script now?" in text
    assert "`references/product-contract.md`" in text
    assert "`references/decision-rules.md`" in text
    assert "`references/env-fix-policy.md`" in text
    assert "`references/ascend-compat.md`" in text
