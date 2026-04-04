import socketserver
import sys
import threading
from http.server import BaseHTTPRequestHandler
from os import name as os_name
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "scripts").resolve()))

import readiness_core
from readiness_core import build_fix_actions, build_state, discover_execution_target, install_packages, probe_hf_endpoint, selected_python_for_execution


class _RetryProbeHandler(BaseHTTPRequestHandler):
    api_attempts = 0

    def do_HEAD(self):
        if self.path == "/":
            self.send_response(405)
        elif self.path == "/api/models/Qwen/Qwen3-0.6B":
            type(self).api_attempts += 1
            self.send_response(200 if type(self).api_attempts >= 3 else 503)
        else:
            self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        return


def _args(**overrides):
    payload = {
        "target": "auto",
        "framework_hint": "auto",
        "entry_script": None,
        "selected_python": None,
        "config_path": None,
        "model_path": None,
        "model_hub_id": None,
        "dataset_path": None,
        "dataset_hub_id": None,
        "dataset_split": None,
        "checkpoint_path": None,
        "task_smoke_cmd": None,
        "cann_path": None,
        "allow_network": False,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def _workspace_python_path(env_root: Path) -> Path:
    if os_name == "nt":
        python_path = env_root / "Scripts" / "python.exe"
    else:
        python_path = env_root / "bin" / "python"
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_text("", encoding="utf-8")
    return python_path


def test_discover_execution_target_matches_qwen_recipe_from_remote_assets(tmp_path: Path):
    target = discover_execution_target(
        tmp_path,
        _args(
            model_hub_id="Qwen/Qwen3-0.6B",
            dataset_hub_id="karthiksagarn/astro_horoscope",
        ),
    )
    assert target["target_type"] == "training"
    assert target["example_recipe_id"] == "qwen3-training"
    assert target["entry_script"].endswith("train.py")


def test_discover_execution_target_stays_inside_current_workspace(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    unrelated_repo = tmp_path / "other-project"
    unrelated_repo.mkdir()
    (unrelated_repo / "train_qwen3.py").write_text("import torch\nimport torch_npu\n", encoding="utf-8")
    (unrelated_repo / "model").mkdir()
    (unrelated_repo / ".venv").mkdir()

    target = discover_execution_target(workspace, _args())

    assert target["working_dir"] == str(workspace)
    assert target["entry_script"] is None
    assert target["model_path"] is None
    assert "example_recipe_id" not in target


def test_discover_execution_target_ignores_hidden_dirs(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    hidden_dir = workspace / ".hidden-tools" / "scripts"
    hidden_dir.mkdir(parents=True)
    (hidden_dir / "tooling.py").write_text("import mindspore\n", encoding="utf-8")
    hidden_other = workspace / ".agent-cache"
    hidden_other.mkdir(parents=True)
    (hidden_other / "helper.py").write_text("print('tooling only')\n", encoding="utf-8")

    target = discover_execution_target(workspace, _args())

    assert target["entry_script"] is None
    assert target["framework_path"] is None
    assert target["framework_evidence"] == []


def test_build_state_uses_workspace_pta_evidence_without_mindspore_probe(tmp_path: Path, fake_selected_python: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "train_qwen3.py").write_text(
        "import torch\nimport torch_npu\nfrom transformers import Trainer, TrainingArguments\nfrom datasets import load_dataset\n",
        encoding="utf-8",
    )
    (workspace / "model").mkdir()

    state = build_state(
        _args(
            target="auto",
            framework_hint="auto",
            selected_python=str(fake_selected_python),
            model_path="model",
        ),
        workspace,
    )

    target = state["target"]
    framework_layer = state["closure"]["layers"]["framework"]
    runtime_layer = state["closure"]["layers"]["runtime_dependencies"]

    assert target["target_type"] == "training"
    assert target["framework_path"] == "pta"
    assert framework_layer["required_packages"] == ["torch", "torch_npu"]
    assert "mindspore" not in framework_layer["required_packages"]
    assert "mindspore" not in (framework_layer.get("import_probes") or {})
    assert "mindspore" not in (runtime_layer.get("required_imports") or [])


def test_build_state_ignores_hidden_dirs_when_inferring_framework(tmp_path: Path, fake_selected_python: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "train_qwen3.py").write_text(
        "import torch\nimport torch_npu\nfrom transformers import Trainer\n",
        encoding="utf-8",
    )
    hidden_dir = workspace / ".hidden-tools" / "scripts"
    hidden_dir.mkdir(parents=True)
    (hidden_dir / "tooling.py").write_text("import mindspore\n", encoding="utf-8")
    (workspace / "model").mkdir()

    state = build_state(
        _args(
            target="auto",
            framework_hint="auto",
            selected_python=str(fake_selected_python),
            model_path="model",
        ),
        workspace,
    )

    assert Path(state["target"]["entry_script"]).name == "train_qwen3.py"
    assert state["target"]["framework_path"] == "pta"
    assert state["target"]["framework_evidence"] == ["pta imports detected"]


def test_runtime_smoke_blocks_when_script_parse_prerequisites_are_missing(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "infer.py").write_text("print('infer')\n", encoding="utf-8")
    (workspace / "model").mkdir()

    state = build_state(
        _args(
            target="inference",
            framework_hint="auto",
            model_path="model",
        ),
        workspace,
    )

    runtime_smoke = next(item for item in state["checks"] if item["id"] == "runtime-smoke")
    assert runtime_smoke["status"] == "block"
    assert "prerequisites are unresolved" in runtime_smoke["summary"]


def test_build_fix_actions_creates_workspace_uv_env_before_installing_missing_packages(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external_env = tmp_path / "external-env"
    external_python = _workspace_python_path(external_env)
    monkeypatch.setattr(readiness_core, "resolve_uv_executable", lambda: None)

    target = {
        "working_dir": str(workspace),
        "framework_path": "pta",
    }
    closure = {
        "layers": {
            "python_environment": {
                "selection_status": "selected",
                "selected_env_root": str(external_env),
                "probe_python_path": str(external_python),
            },
            "framework": {
                "framework_path": "pta",
                "import_probes": {"torch": False, "torch_npu": False},
                "recommended_package_specs": ["torch==2.9.0", "torch_npu==2.9.0"],
            },
            "runtime_dependencies": {"import_probes": {}},
            "workspace_assets": {"entry_script": {"exists": True}},
            "remote_assets": {"assets": {}},
        }
    }

    actions = build_fix_actions(target, closure, {"blockers_detailed": []}, allow_network=False)
    action_types = [item["action_type"] for item in actions]

    assert "install_uv" in action_types
    assert "create_or_select_env" in action_types
    assert "install_framework_packages" in action_types
    assert action_types.index("install_uv") < action_types.index("install_framework_packages")
    assert action_types.index("create_or_select_env") < action_types.index("install_framework_packages")


def test_build_fix_actions_plans_uv_env_and_package_installs_when_workspace_env_is_missing(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(readiness_core, "resolve_uv_executable", lambda: None)

    target = {
        "working_dir": str(workspace),
        "framework_path": "pta",
    }
    closure = {
        "layers": {
            "python_environment": {
                "selection_status": "missing",
            },
            "framework": {
                "framework_path": "pta",
                "required_packages": ["torch", "torch_npu"],
                "recommended_package_specs": ["torch==2.9.0", "torch_npu==2.9.0"],
                "import_probes": {},
            },
            "runtime_dependencies": {
                "required_imports": ["datasets", "transformers"],
                "import_probes": {},
            },
            "workspace_assets": {"entry_script": {"exists": True}},
            "remote_assets": {"assets": {}},
        }
    }

    actions = build_fix_actions(target, closure, {"blockers_detailed": []}, allow_network=False)
    action_types = [item["action_type"] for item in actions]

    assert "install_uv" in action_types
    assert "create_or_select_env" in action_types
    assert "install_framework_packages" in action_types
    assert "install_runtime_dependencies" in action_types
    assert action_types.index("install_uv") < action_types.index("install_framework_packages")
    assert action_types.index("install_uv") < action_types.index("install_runtime_dependencies")
    assert action_types.index("create_or_select_env") < action_types.index("install_framework_packages")
    assert action_types.index("create_or_select_env") < action_types.index("install_runtime_dependencies")


def test_selected_python_for_execution_prefers_workspace_uv_env_over_external_selection(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace_python = _workspace_python_path(workspace / ".venv")
    external_env = tmp_path / "external-env"
    external_python = _workspace_python_path(external_env)

    selected = selected_python_for_execution(
        workspace,
        {"selected_python": str(external_python)},
        {
            "layers": {
                "python_environment": {
                    "selected_env_root": str(external_env),
                    "probe_python_path": str(external_python),
                }
            }
        },
    )

    assert selected == workspace_python


def test_install_packages_uses_uv_pip_with_selected_python(tmp_path: Path, monkeypatch):
    python_path = _workspace_python_path(tmp_path / ".venv")
    uv_path = tmp_path / "uv"
    uv_path.write_text("", encoding="utf-8")
    commands = []

    monkeypatch.setattr(readiness_core, "ensure_uv_available", lambda: (True, "uv is already available", uv_path))
    monkeypatch.setattr(readiness_core, "preferred_pip_index_urls", lambda: ["https://mirror.example/simple"])

    def _record_command(command):
        commands.append(command)
        return True, ""

    monkeypatch.setattr(readiness_core, "run_install_command", _record_command)

    ok, message = install_packages(python_path, ["torch==2.9.0", "torch_npu==2.9.0"])

    assert ok is True
    assert "installed packages via https://mirror.example/simple" == message
    assert commands == [[
        str(uv_path),
        "pip",
        "install",
        "--python",
        str(python_path),
        "--index-url",
        "https://mirror.example/simple",
        "torch==2.9.0",
        "torch_npu==2.9.0",
    ]]


def test_build_fix_actions_gates_remote_downloads_on_allow_network(tmp_path: Path):
    target = {
        "working_dir": str(tmp_path),
        "allow_network": False,
    }
    closure = {
        "layers": {
            "python_environment": {"selection_status": "selected"},
            "framework": {"framework_path": None, "import_probes": {}},
            "runtime_dependencies": {"import_probes": {}},
            "workspace_assets": {
                "entry_script": {"exists": True},
                "model_path": {"satisfied": False},
                "dataset_path": {"satisfied": False},
            },
            "remote_assets": {
                "assets": {
                    "model_path": {"repo_id": "Qwen/Qwen3-0.6B", "local_path": str(tmp_path / "model")},
                    "dataset_path": {"repo_id": "repo", "split": "train", "local_path": str(tmp_path / "dataset")},
                }
            },
        }
    }
    actions = build_fix_actions(target, closure, {"blockers_detailed": []}, allow_network=False)
    action_types = {item["action_type"] for item in actions}
    assert "download_model_asset" not in action_types
    assert "download_dataset_asset" not in action_types


def test_build_fix_actions_adds_example_scaffold_when_recipe_applies(tmp_path: Path):
    target = {
        "working_dir": str(tmp_path),
        "entry_script": str(tmp_path / "train.py"),
        "example_template_path": str((Path(__file__).resolve().parents[1] / "examples" / "qwen3_0_6b_training_example.py").resolve()),
    }
    closure = {
        "layers": {
            "python_environment": {"selection_status": "selected"},
            "framework": {"framework_path": None, "import_probes": {}},
            "runtime_dependencies": {"import_probes": {}},
            "workspace_assets": {
                "entry_script": {"exists": False},
                "model_path": {"satisfied": True},
                "dataset_path": {"satisfied": True},
            },
            "remote_assets": {"assets": {}},
        }
    }
    actions = build_fix_actions(target, closure, {"blockers_detailed": []}, allow_network=False)
    assert any(item["action_type"] == "scaffold_example_entry" for item in actions)


def test_probe_hf_endpoint_retries_and_falls_back_to_api_probe():
    _RetryProbeHandler.api_attempts = 0
    with socketserver.TCPServer(("127.0.0.1", 0), _RetryProbeHandler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        endpoint = f"http://127.0.0.1:{server.server_address[1]}"
        reachable, error = probe_hf_endpoint(endpoint)
        server.shutdown()
        thread.join(timeout=2)

    assert reachable is True
    assert error is None
    assert _RetryProbeHandler.api_attempts == 3
