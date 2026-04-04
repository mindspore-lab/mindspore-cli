from pathlib import Path


SKILL_YAML = Path(__file__).resolve().parents[1] / "skill.yaml"
SKILL_MD = Path(__file__).resolve().parents[1] / "SKILL.md"


def test_manifest_contract_fields_present():
    text = SKILL_YAML.read_text(encoding="utf-8")
    assert 'schema_version: "1.1.0"' in text
    assert 'name: "algorithm-agent"' in text
    assert 'display_name: "Algorithm Agent"' in text
    assert 'version: "0.3.0"' in text
    assert 'type: "manual"' in text
    assert 'path: "SKILL.md"' in text
    assert 'network: "none"' in text
    assert 'filesystem: "workspace-write"' in text


def test_manifest_declares_route_input_and_output_contract():
    text = SKILL_YAML.read_text(encoding="utf-8")
    assert 'name: "working_dir"' in text
    assert 'name: "source_text"' in text
    assert 'name: "reference_code_path"' in text
    assert 'name: "route_preference"' in text
    assert 'choices: ["generic-feature", "mhc", "attnres"]' in text
    assert 'report_schema' in text
    assert 'out_dir_layout' in text


def test_skill_declares_route_specific_plan_fields():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "`FeatureSpec` that includes `integration_route` and" in text
    assert "`route_specific_constraints`" in text
    assert "`route_specific_validations`" in text
