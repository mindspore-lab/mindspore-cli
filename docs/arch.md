# ms-cli Architecture

This document summarizes the current repository structure and runtime boundaries in this checkout. It is the short contributor-facing architecture reference.

## Top-Level Shape

```text
ms-cli/
  cmd/ms-cli/              process entrypoint
  internal/app/            composition root, startup, commands, UI bridging
  internal/project/        roadmap and weekly status helpers
  internal/train/          train request and target types
  agent/
    context/               token budget and compaction
    loop/                  concrete ReAct-style execution engine
    memory/                memory store, retrieval, and policy
    orchestrator/          planner-driven dispatch between agent and workflow
    planner/               plan generation and execution-mode selection
    session/               session state and persistence
  workflow/
    executor/              workflow executor implementations and stubs
    train/                 train lane controller, setup, run, demo backend
  integrations/
    domain/                external domain schema and client
    llm/                   provider registry and OpenAI-compatible client
    skills/                skill repository and invocation integration
  permission/              permission policy, types, store
  runtime/
    shell/                 stateful shell command runner
    probes/                local and target readiness probes
  tools/
    fs/                    read, grep, glob, edit, write tools
    shell/                 shell tool wrapper
  ui/                      Bubble Tea app, shared model, panels, slash commands
  trace/                   execution trace writing
  report/                  summary generation
  configs/                 config loading, state, shared config types
  demo/scenarios/          workflow demo data
  test/mocks/              test doubles
  docs/                    architecture, roadmap, and update docs
```

## Primary Runtime Flows

### Standard task execution

```text
cmd/ms-cli
  -> internal/app.Run(...)
  -> internal/app.Wire(...)
  -> ui.New(...)
  -> agent/orchestrator.Run(...)
  -> agent/planner.Plan(...)         if an LLM provider is configured
  -> agent executor or workflow executor

agent executor path:
  internal/app/adapter.go
    -> agent/loop.Engine
    -> tools.Registry
    -> tools/fs or tools/shell
    -> runtime/shell.Runner
```

Current behavior:

- `cmd/ms-cli/main.go` only delegates to `internal/app.Run(...)`.
- `internal/app` is the composition root and owns wiring, provider setup, tool setup, and UI event conversion.
- `agent/orchestrator` owns orchestration-level request and event types and chooses between agent mode and workflow mode.
- `agent/planner` is optional. When no provider is configured, the orchestrator falls back directly to agent mode.
- `workflow/executor` is split between a real stub (`NewStub`) and a demo executor path used by `--demo`.

### Train mode

```text
ui input
  -> internal/app /train command
  -> workflow/train.Controller
  -> workflow/train setup/run sequences
  -> runtime/probes/local and runtime/probes/target
  -> internal/app train-event conversion
  -> ui/model events
```

Current boundary:

- `workflow/train` owns train-specific sequencing and the demo backend.
- `runtime/probes/*` only perform checks and return probe results.
- `internal/app/train.go` is the bridge from workflow train events into UI-facing events and state.
- `internal/train` carries train request and target types shared by app and workflow layers.

## Package Responsibilities

- `internal/app/`
  Loads config, wires dependencies, starts the TUI, handles slash commands, and converts backend events into `ui/model.Event`.
- `agent/orchestrator/`
  Dispatches a request to either the ReAct engine or a workflow executor based on planner output.
- `agent/planner/`
  Builds and validates structured plans, including execution mode (`agent` vs `workflow`).
- `agent/loop/`
  Runs the concrete LLM/tool loop and owns execution details such as tool calling, permission checks, tracing, and context updates.
- `workflow/train/`
  Encapsulates the train lane setup/run/retry/analyze flows and their backend abstraction.
- `tools/`
  Exposes LLM-callable tool surfaces. It is the boundary the agent loop uses rather than reaching into lower-level runtime code directly.
- `runtime/shell/`
  Executes shell commands with workspace, timeout, and command safety checks.
- `permission/`
  Centralizes permission decisions and persistence for potentially sensitive actions.
- `ui/`
  Consumes events and renders the Bubble Tea interface. It should not be imported by lower layers.

## Dependency Boundaries

Keep dependencies flowing downward. The current code follows these rules:

```text
cmd/ms-cli -> internal/app
internal/app -> agent, workflow, ui, configs, integrations, tools, permission, trace
agent -> integrations, permission, configs, trace
workflow -> internal/train, runtime/probes, configs
tools -> runtime, integrations, configs
runtime -> configs
ui -> configs
report -> trace, configs
```

Important constraints:

- `cmd/ms-cli/` should stay thin.
- `internal/app/` is the wiring layer, not a reusable core package.
- `agent/` must not depend on `ui/` or `runtime/` directly.
- `workflow/train/` must not import `ui/model`; conversion belongs in `internal/app/train.go`.
- `tools/` may call `runtime/`, but `runtime/` must not call `tools/`.
- `configs/` is shared configuration, not a home for application logic.

## Related Docs

- `README.md` is the user-facing quick start and command overview.
- `docs/arch.md` is the concise contributor-facing architecture map.
- `docs/ms-cli-arch.md` and `docs/architecture.md` are older architecture references and should be kept aligned with the code before treating them as authoritative.

When docs and code disagree, follow the code and update the docs.
