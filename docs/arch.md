# ms-cli Architecture

This document describes the target architecture after the refactor
(see `docs/ms-cli-refactor.md`). It is the single contributor-facing
architecture reference.

## Three-Repo Model

```text
ms-cli (this repo)          runtime — TUI, agent loop, tool registry
ms-skills            instructions — SKILL.md + skill.yaml per skill
ms-factory (incubating/)    knowledge — operator/failure/trick/model cards
```

- `ms-cli` loads skills from `ms-skills` and cards from `ms-factory`.
- Skills are portable across CLIs (Claude Code, OpenCode, Gemini CLI, Codex).
- Factory cards grow from real experiment data and incident reports.

## Top-Level Shape

```text
ms-cli/
  cmd/ms-cli/              process entrypoint
  internal/
    app/                   composition root, startup, commands, UI bridging
    factory/               local factory card store and resolver
    project/               roadmap and weekly status helpers
    train/                 train request and target types
  agent/
    context/               token budget and compaction
    loop/                  ReAct-style execution engine (the core runtime)
    memory/                memory store, retrieval, and policy
    session/               session state and persistence
  workflow/
    train/                 train lane controller, setup, run, demo backend
  integrations/
    domain/                external domain schema and client
    factory/               remote factory fetch and sync
    llm/                   provider registry and OpenAI-compatible client
    skills/                skill listing, loading, and metadata
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
  incubating/factory/      factory schemas, cards, packs (future separate repo)
  test/mocks/              test doubles
  docs/                    architecture, plans, and backlog
```

## Primary Runtime Flow

```text
cmd/ms-cli
  -> internal/app.Run(...)
  -> internal/app.Wire(...)
  -> ui.New(...)

user input
  -> internal/app.processInput(...)
  -> slash command (/train, /diagnose, /factory status, ...)
     or free text -> runTask(...)

runTask:
  -> compose effective system prompt:
       EngineConfig.SystemPrompt
       + skill summaries (from integrations/skills)
       + active skill SKILL.md (if command-scoped)
  -> agent/loop.Engine.RunWithContext(task)
  -> tools.Registry
  -> tools/fs or tools/shell
  -> runtime/shell.Runner
  -> loop.Event stream -> model.Event -> ui
```

No orchestrator, no planner, no adapter layer. The app calls the engine
directly. The LLM plans inline within the agent loop.

### Skill activation

Skills are command-scoped. No sticky state.

```text
/diagnose my training crashed
  -> app loads failure-agent SKILL.md from integrations/skills
  -> sets task.SkillContext (one task only)
  -> engine composes effective system prompt: base + active skill
  -> engine runs
  -> next task has no skill context (base prompt only)
```

Free text uses the base system prompt which includes skill summaries
(name + one-line description). The LLM has awareness of available
capabilities but this is not deterministic routing — for reliable
skill activation, use explicit commands.

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

`/train` is preserved during transition. It uses its own controller
and demo backend, independent of the agent loop. Eventually `/train`
features migrate to agent-skills.

## Package Responsibilities

- **`internal/app/`**
  Loads config, wires dependencies, starts the TUI, handles slash commands,
  dispatches tasks to the engine, and converts `loop.Event` to `ui/model.Event`.

- **`agent/loop/`**
  The core runtime. Runs the LLM/tool loop: tool calling, permission checks,
  tracing, context updates. Composes effective system prompt per task
  (base + active skill).

- **`integrations/skills/`**
  Lists available skills from the ms-skills repo (summaries only).
  Loads one skill fully on demand (SKILL.md + skill.yaml metadata).

- **`internal/factory/`**
  Local card store. Search, get, list cards. Status from pack manifest.

- **`integrations/factory/`**
  Remote fetch and sync. Downloads packs to local store.

- **`workflow/train/`**
  Train lane controller, setup/run/retry/analyze flows, demo backend.
  Independent of agent loop — uses its own event system.

- **`tools/`**
  LLM-callable tool surfaces (filesystem, shell). Stateless tool definitions.

- **`runtime/shell/`**
  Stateful command runner with workspace, timeout, and safety checks.

- **`permission/`**
  Permission decisions and persistence for sensitive actions.

- **`ui/`**
  Bubble Tea interface. Consumes events, renders panels. Not imported by
  lower layers.

- **`configs/`**
  Shared configuration types and loaders.

## Dependency Boundaries

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

Constraints:

- `cmd/ms-cli/` stays thin.
- `internal/app/` is the wiring layer, not a reusable core package.
- `agent/` must not depend on `ui/` or `runtime/` directly.
- `workflow/train/` must not import `ui/model`; conversion belongs in
  `internal/app/train.go`.
- `tools/` may call `runtime/`, but `runtime/` must not call `tools/`.
- `configs/` is shared configuration, not a home for application logic.

## Removed Packages (refactor)

The following packages are removed by the refactor (see `docs/ms-cli-refactor.md`):

- `agent/orchestrator/` — was a dispatch layer between planner and executors.
  After removing workflow mode, it became a passthrough. The app now calls
  the engine directly.
- `agent/planner/` — was an LLM-based plan generator that decided agent vs
  workflow mode. Standard agent loops don't have a separate planner; the LLM
  plans inline. Skill selection uses explicit commands, not an LLM pre-call.
- `workflow/executor/` — was a workflow execution framework (demo JSON playback
  + stub). Never developed beyond demo. Train mode uses its own controller.
- `internal/app/adapter.go` — was a type converter between orchestrator and
  engine event types. No longer needed with direct engine calls.
- `demo/scenarios/` — JSON scenario files for workflow demo playback.

## Related Docs

- `docs/ms-cli-refactor.md` — refactor plan (workstream A)
- `docs/ms-skills-update-plan.md` — skills plan (workstream B)
- `docs/incubating-factory-plan.md` — factory plan (workstream C)
- `docs/features-backlog.md` — deferred features
- `docs/ai/contributor-guide.md` — contributor conventions

When docs and code disagree, follow the code and update the docs.
