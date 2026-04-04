# Readiness-Agent Product Audit

Date: 2026-03-25

## Audit Verdict

Current state:

- product-grade MVP remains credible for controlled local use
- recent P0 Python hardening materially improves real server robustness
- helper scripts now target Python 3.9 compatibility and the skill now has an
  explicit selected-workspace-Python contract
- a top-level orchestrator now closes the default no-env -> env-fix -> rerun
  path for workspace-local environments
- the top-level CLI now accepts `--check`, `--fix`, and `--auto` as
  compatibility aliases for `--mode`
- full confidence still requires a full pytest rerun in an environment with
  `pytest` installed and broader real-server regression coverage
- helper and report behavior now make it explicit that unresolved
  workspace-local environments must not fall back to system Python or system
  `pip`

Recommended readiness level:

- still appropriate for internal alpha / `ms-cli` integration experiments
- stronger than the earlier prompt-first MVP because Python control-plane
  behavior is now explicit instead of implicit
- not yet fully closed as a final GA skill

## What Is Already Product-Complete

- single-machine readiness certification is centered on one user-facing
  outcome: `READY` / `WARN` / `BLOCKED`
- execution target discovery is explicit and evidence-driven
- dependency closure is target-scoped, not machine-generic
- framework/runtime probing is selected-environment-aware
- selected workspace Python can now be expressed explicitly through
  `selected_python` and `selected_env_root`
- the deterministic helper pipeline now has a dedicated selected-Python
  resolution stage
- downstream probing and task smoke can now use the selected workspace Python
  instead of silently trusting the host interpreter
- helper scripts have been moved away from Python 3.10-only type syntax so the
  control plane is compatible with Python 3.9+
- `run_readiness_pipeline.py` now provides the missing top-level orchestration
  for `mode=check/fix/auto`, env-fix execution, and one-shot full-pipeline
  re-entry
- execution-target discovery can now infer `model_path` from local model
  marker files instead of requiring an explicit input in common inference
  workspaces
- explicit task smoke is supported through `task_smoke_cmd`
- blocker taxonomy is normalized and stable
- native `env_fix` planning and controlled execution exist
- revalidation is enforced from `needs_revalidation` coverage, not from a
  placeholder boolean
- final report preserves internal evidence fields and user-visible summary
- realistic workspace fixtures now cover training-style and inference-style
  workspaces

## Current Implemented Pipeline

1. `run_readiness_pipeline.py`
2. `resolve_selected_python.py`
3. `discover_execution_target.py`
4. `build_dependency_closure.py`
5. `run_task_smoke.py` when `task_smoke_cmd` exists
6. `collect_readiness_checks.py`
7. `normalize_blockers.py`
8. `plan_env_fix.py`
9. `execute_env_fix.py`
10. rerun required checks when remediation changes the environment
11. `build_readiness_report.py`

## Recent P0 Python Hardening

- new manifest inputs: `selected_python` and `selected_env_root`
- new helper: `resolve_selected_python.py`
- discovery now records selected-Python metadata into the execution target
- dependency closure now rebuilds Python facts from the selected workspace
  interpreter instead of defaulting to the current interpreter
- task smoke now skips with an explicit environment reason when selected Python
  is missing, rather than silently falling back to a host interpreter
- readiness checks now surface selected-Python resolution failures as
  environment blockers
- helper scripts now use Python 3.9-compatible typing forms instead of
  Python 3.10-only `X | None` syntax
- env-fix execution can now default environment creation to
  `<working_dir>/.venv` and rerun the full helper sequence after a successful
  mutation
- env-fix package planning now filters probe metadata out of package-name hints
  so framework/runtime repair commands stay syntactically valid
- PTA probing can now discover a local Ascend `set_env.sh` candidate and
  source it for framework/runtime import probes when the current environment is
  not already active
- PTA framework repair now prefers CPU `torch` wheels and installs `torch_npu`
  separately, avoiding unnecessary NVIDIA dependency pulls on Ascend
- Ascend-backed `pta`, `mindspore`, and `mixed` paths now carry a small known
  hidden runtime dependency profile for compiler-side Python modules such as
  `decorator`, `scipy`, and `attrs`, allowing env-fix to remediate them in one
  pass instead of discovering them one at a time during later runtime
- closure now models `accelerate` as a default companion dependency for
  `transformers` workspaces, and it also probes high-frequency explicit
  ecosystem imports such as `peft`, `trl`, `evaluate`, and `sentencepiece`

## Evidence Of Stability

- historical baseline before this P0 round: `31 passed, 1 warning`
- current local session after the P0 edits: all readiness helper scripts
  compile successfully with `python -m py_compile`
- selected-Python resolution was manually validated with an explicit Python
  interpreter
- the `discover_execution_target -> build_dependency_closure ->
  collect_readiness_checks` path was manually validated with the new
  selected-Python contract
- `run_readiness_pipeline.py` was manually validated with a fake `uv` tool to
  confirm the `.venv` creation path, two-pass pipeline re-entry, and final
  `python-selected-python=ok` outcome
- `discover_execution_target.py` was manually validated to infer `model_path`
  from workspace model markers
- full pytest rerun is still pending in this workspace because the local
  Python environment used for development does not currently have `pytest`
  installed

## Remaining Product Gaps

### 1. Real Framework Smoke Is Still Minimal

Current framework smoke verifies the minimum import/bootstrap prerequisite,
not a real task-level framework execution path such as:

- one tensor on device
- one forward pass
- one train-step primitive

This is acceptable for now, but stronger task evidence would reduce false
`READY` in edge cases where import succeeds but runtime usage still fails.

### 1.5 Ascend Hidden Dependency Coverage Is Now Better, But Still Profile-Based

The readiness closure now models a small known set of Ascend compiler-side
Python dependencies for `pta` and `mindspore` paths. This improves the common
server experience materially because `decorator`, `scipy`, and `attrs` can be
planned and installed in a single remediation round.

However, this is still a profile-based approach, not a full dynamic discovery
engine. Rare server-specific compiler-side imports may still surface later and
require one more remediation pass.

### 2. `env_fix` Capability Is Still Narrow By Design

The current native remediation scope is intentionally limited to safe
user-space actions. This is correct for product safety, but it means:

- no system-layer repair
- no CANN / driver remediation
- no config mutation
- no dataset repair
- no model code patching

This is not a flaw, but it should remain explicit in integration plans.

### 3. No Formal `ms-cli` Adapter Contract Yet

The skill now has a clearer internal Python contract and stable artifacts, but
the adapter layer that maps:

- user request
- `mode`
- `target`
- `selected_python` / `selected_env_root`
- artifact locations
- final surfaced summary

into `ms-cli` runtime behavior is not yet formalized in this repo.

## Recommended Final Steps

1. Rerun and expand the test matrix.
   Add:
   - a full pytest rerun in an environment that has `pytest`
   - a regression case for host Python 3.9 plus workspace `.venv` Python 3.10
   - a regression case for workspace Python 3.9 selected successfully
   - a regression case for missing `.venv` followed by `env_fix` recovery

2. Validate the orchestrator on real server matrices.
   Add:
   - one PTA/Ascend workspace with no initial `.venv` and `uv` available
   - one workspace with an existing `.venv` that should be selected directly
   - one workspace where `uv` is missing and the blocker path must remain clear
   - one workspace where Ascend runtime is only available after sourcing
     `set_env.sh`

3. Optionally strengthen runtime evidence.
   Add one more controlled smoke tier for frameworks or task kinds where
   minimal runtime execution is safe and deterministic.

## Bottom Line

`readiness-agent` is no longer just a prompt-first concept skill.
It now behaves like a real certification product with deterministic helpers,
native remediation, revalidation, stable artifacts, realistic fixtures, and a
much clearer Python control-plane story.

The biggest remaining work is no longer basic Python survivability or
top-level repair orchestration. It is strengthening real runtime evidence and
validating the current closure on broader real-server matrices.
