# Workstream B: ms-skills Update

## Goal

Add a new training-and-optimization skill family to `~/work/ms-skills`
that matches the `ms-cli` refactor direction:

- `ms-cli` remains the runtime and command entrypoint
- `ms-skills` provides prompt-oriented domain skills
- `ms-factory` provides structured knowledge assets

This workstream should follow the existing `ms-skills` repository
contract:

- each skill has `SKILL.md`
- each skill has `skill.yaml`
- `skill.yaml.entry.type` is valid against the repo contract
- contract tests continue to pass

For the first phase, these new skills are primarily **manual / prompt-oriented
skills**, not fully automated executable workflows.

## Scope

Add six new high-level skills:

- `setup-agent`
- `failure-agent`
- `accuracy-agent`
- `performance-agent`
- `op-agent`
- `algorithm-agent`

These are the high-level skills that map most directly to the current demo
stories in `ms-cli`.

## Out Of Scope For This Workstream

Do not treat this workstream as the place to finish all execution tooling.

The following are explicitly deferred unless their dependencies already exist:

- full remote execution framework
- fully standardized shared Python tool runtime
- direct Factory report submission
- automatic Factory card mutation

Skill runs may eventually consume helper scripts, but the initial version of
these skills should stand on the existing `SKILL.md` + `skill.yaml` contract.

## Design Principles

### 1. Prompt-first, not automation-first

These new skills should start as instruction packages that guide the agent.
They do not need to become fully automated tools on day one.

Recommended initial `skill.yaml.entry.type`:

- `manual`

### Quality bar: invoke ratio

Writing a SKILL.md that the LLM reliably follows is prompt engineering,
not documentation. Expect iteration: write → test against LLM → observe
deviations → rewrite → retest.

Common failure modes:
- LLM skips steps (jumps to guessing instead of querying factory first)
- LLM doesn't use tools (reasons from training data instead of calling tools)
- Instructions too vague (LLM doesn't know what concrete actions to take)
- Instructions too rigid (breaks when user's situation doesn't match template)

Techniques to improve invoke ratio:
- Few-shot examples in SKILL.md (show expected input → tool call → output)
- Explicit mandatory actions ("you MUST call factory_query before reasoning")
- Negative examples ("do NOT skip the evidence collection step")
- Checkpoint phrases ("before proceeding, confirm you have collected: ...")
- Short, directive instructions (LLMs follow short rules better than long docs)
- Role framing ("You are a failure diagnosis specialist. You always start by...")
- Restate key rules at end of SKILL.md (recency bias)

Each skill should have a small eval suite (10-20 test cases) to measure
step compliance. Track invoke ratio per skill and iterate until reliable.

### Fallback path: controlled execution

If a skill cannot reach acceptable invoke ratio through prompt techniques
alone, it can be escalated to `entry.type: "controlled"` in a future
refactor phase. This means a Go controller in ms-cli drives the steps
deterministically, calling the LLM per-step with focused instructions
(similar to the existing `workflow/train/controller.go` pattern).

This is not needed now — invest in prompt quality first. But the
architecture preserves this option:
- `skill.yaml` already has `entry.type` — add `"controlled"` value later
- The engine remains available for per-step LLM calls
- The controller pattern is proven in `workflow/train/`

Decision sequence:
1. Write skill as `manual` with good prompt engineering
2. Measure invoke ratio with eval suite
3. Iterate on SKILL.md until acceptable
4. Only if steps 1-3 fail: escalate to `controlled` in a future phase

### 2. Reuse existing specialized skills

`op-agent` should compose with the existing implementation-oriented skills
rather than replacing them.

Examples:

- `cpu-plugin-builder`
- `cpu-native-builder`
- `gpu-builder`
- `npu-builder`
- `mindspore-aclnn-operator-devflow`

The new agent-style skills should sit above these as diagnosis / routing /
strategy layers.

### 3. Keep Factory vocabulary aligned

Skill instructions and query examples should use the same core card types that
Factory uses:

- `known_failure`
- `operator`
- `model`
- `trick`

Do not introduce parallel terminology like `perf-features` or `algo-features`
unless the Factory side formally adopts those names.

### 4. Separate skill packaging from helper-tool rollout

If shared helper scripts are added, they should be staged after the basic skill
packages exist and after the corresponding `ms-cli` / Factory integration is
available.

## B1: Add Six New Manual Skills

Each skill should follow the existing package structure:

```text
skills/<skill-name>/
├── SKILL.md
├── skill.yaml
└── tests/
```

Optional `reference/` content may be added where useful.

### B1.1 setup-agent

**Purpose**

Validate and prepare execution environment for training or remote execution.

**Recommended use**

- environment readiness
- SSH preparation
- dependency checks
- device availability checks

**SKILL.md guidance**

The skill should instruct the agent to:

1. identify whether the target is local or remote
2. collect environment facts
3. check framework/device/dependency readiness
4. summarize pass/fail state
5. suggest fixes for missing prerequisites

### B1.2 failure-agent

**Purpose**

Diagnose crashes, runtime errors, hangs, and communication failures.

**Recommended use**

- training crash
- runtime failure
- HCCL / NCCL / device communication issues
- missing operator / unsupported path

**SKILL.md guidance**

The skill should instruct the agent to:

1. collect logs and error evidence
2. identify operator/component/platform facts
3. if Factory query tooling is available, search `known_failure` cards
4. if Factory query tooling is available, consult `operator` cards when no known failure matches
5. propose root cause and fix options

### B1.3 accuracy-agent

**Purpose**

Diagnose accuracy regression, numerical drift, and wrong-result issues.

**Recommended use**

- accuracy drop
- cross-platform numerical mismatch
- unexpected evaluation degradation

**SKILL.md guidance**

The skill should instruct the agent to:

1. compare baseline vs current results
2. gather model/config/runtime context
3. if Factory query tooling is available, inspect `model` cards for expected context
4. if Factory query tooling is available, inspect `known_failure` or future accuracy knowledge assets
5. inspect likely dtype / operator / preprocessing causes
6. propose fixes and validation steps

### B1.4 performance-agent

**Purpose**

Diagnose throughput, latency, and memory bottlenecks.

**Recommended use**

- low throughput
- high latency
- memory pressure
- poor platform utilization

**SKILL.md guidance**

The skill should instruct the agent to:

1. collect runtime metrics and bottleneck evidence
2. compare observed performance against expected behavior
3. if Factory query tooling is available, inspect `operator` cards for bottleneck operators
4. if Factory query tooling is available, inspect `trick` cards for applicable optimization techniques
5. recommend optimizations and expected gains

### B1.5 op-agent

**Purpose**

Drive missing-operator analysis and route to the right implementation workflow.

**Recommended use**

- missing operator
- unsupported backend kernel
- operator implementation gap

**SKILL.md guidance**

The skill should instruct the agent to:

1. identify the missing operator and platform gap
2. if Factory query tooling is available, inspect the `operator` card
3. choose the right implementation path
4. delegate to an existing builder skill where possible
5. summarize implementation and validation next steps

### B1.6 algorithm-agent

**Purpose**

Recommend and apply algorithm-level techniques for quality or convergence
improvement.

**Recommended use**

- improve quality
- improve convergence
- select a training trick

**SKILL.md guidance**

The skill should instruct the agent to:

1. understand the user goal
2. if Factory query tooling is available, inspect `trick` cards
3. filter by model / method / platform applicability
4. explain benefits, risks, and validation path
5. suggest the change before application

## B2: skill.yaml Requirements

Each new skill must include a valid `skill.yaml` following the existing
repository contract.

For the initial version:

- `entry.type: manual`
- `entry.path: SKILL.md`
- inputs and outputs should still be declared correctly
- dependencies should remain minimal and honest

Do not overstate automation in `skill.yaml` before helper tooling exists.

## B3: Shared Tooling, Staged Separately

Shared helper tools are useful, but should be introduced only when the
corresponding runtime support exists.

### Candidate shared tools

- `skills/_shared/tools/factory_query.py`
- `skills/_shared/tools/ssh_exec.py`
- `skills/_shared/tools/apply_patch.py`
- `skills/_shared/tools/stream_logs.py`

### Staging rule

Only add a shared tool when its upstream dependency is real:

- `factory_query.py`
  - depends on `ms-cli factory query ...` existing
- `ssh_exec.py`
  - depends on a stable remote execution contract
- `apply_patch.py`
  - depends on a clear policy for how skills apply changes
- `stream_logs.py`
  - depends on a stable log/streaming execution pattern

So shared tools should be treated as a later sub-phase, not as a prerequisite
for creating the six new skill packages.

## B4: Update AGENTS.md

Update `~/work/ms-skills/AGENTS.md` to include the new skill family.

Recommended section:

```markdown
### Training & Optimization

| Skill | Path | Description |
|-------|------|-------------|
| setup-agent | skills/setup-agent/ | validate environment readiness |
| failure-agent | skills/failure-agent/ | diagnose crashes and runtime failures |
| accuracy-agent | skills/accuracy-agent/ | diagnose accuracy drift and wrong results |
| performance-agent | skills/performance-agent/ | profile and optimize performance |
| op-agent | skills/op-agent/ | analyze missing operators and route implementation |
| algorithm-agent | skills/algorithm-agent/ | recommend and apply algorithm techniques |
```

Activation hints should also be added to the “Active Skills” section using the
same conventions as the rest of the repo.

## B5: Add Commands

Add command docs under `ms-skills/commands/` that match the user-facing
entrypoints expected by `ms-cli`.

These command docs must stay aligned with Workstream A, Phase A4:

- `ms-cli` owns runtime command handling and task construction
- command-scoped skill activation happens in `ms-cli`
- `ms-skills/commands/*.md` define the user-facing command vocabulary
- the same command names and routing intent must be reflected in both repos

Suggested commands:

- `commands/setup.md`
- `commands/diagnose.md`
- `commands/optimize.md`

Recommended routing:

- `/setup` -> `setup-agent`
- `/diagnose` -> router command, not direct multi-skill injection
- `/optimize` -> `performance-agent` or `algorithm-agent`

`/diagnose` should not load multiple full skills into one task. That would
conflict with the command-scoped design from A4.

Recommended implementations for `/diagnose`:

1. split into more specific commands later:
   - `/diagnose-crash`
   - `/diagnose-accuracy`
   - `/diagnose-perf`
2. or keep `/diagnose` as a thin router in `ms-cli`:
   - ask the user for clarification, or
   - use lightweight deterministic routing before loading one skill

Keep command wording aligned with `ms-cli` so the two repos describe the same
interaction model.

## B6: Validation

Run the existing repository validation after adding the skills:

```bash
python tools/check_consistency.py
pytest tests/contract
```

Add skill-specific tests where appropriate, but keep the first pass small and
contract-compliant.

## Recommended Build Order

1. Add `setup-agent`, `failure-agent`, `accuracy-agent`, `performance-agent`,
   `op-agent`, and `algorithm-agent` as manual skills.
2. Add matching `skill.yaml` files with honest dependency declarations.
3. Update `AGENTS.md`.
4. Add command docs.
5. Run consistency and contract tests.
6. Only then evaluate which shared helper tools are actually needed next.

## Summary

This workstream should first create a clean, prompt-oriented skill family that
matches the `ms-cli` demo-story direction. Shared helper tooling and deeper
Factory integration should follow only after the corresponding runtime support
is available.
