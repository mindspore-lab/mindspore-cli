# mscode Architecture

This document describes the target architecture after the refactor
(see `docs/impl-guide/mindspore-code-refactor-3.md`). It is the single contributor-facing
architecture reference.

## Three-Repo Model

```text
mscode (this repo)          runtime — TUI, agent loop, tool registry
ms-skills            instructions — SKILL.md + skill.yaml per skill
ms-factory (incubating/)    knowledge — operator/failure/trick/model cards
```

- `mscode` loads skills from `ms-skills` and cards from `ms-factory`.
- Skills are portable across CLIs (Claude Code, OpenCode, Gemini CLI, Codex).
- Factory cards grow from real experiment data and incident reports.

## Top-Level Shape

```text
mindspore-code/
  cmd/mscode/              process entrypoint
  internal/
    app/                   composition root, startup, commands, UI bridging, prompt history persistence
    factory/               local factory card store and resolver
    project/               roadmap and weekly status helpers
    train/                 train request and target types
  agent/
    context/               context window management and compaction
    loop/                  ReAct-style execution engine (the core runtime)
    memory/                memory store, retrieval, and policy
    session/               session state and persistence
  workflow/
    train/                 train lane controller, setup, run, demo backend
  integrations/
    domain/                external domain schema and client
    factory/               remote factory fetch and sync
    llm/                   unified provider manager (openai-completion/openai-responses/anthropic)
    skills/                skill listing, loading, and metadata
  permission/              permission policy, types, store
  runtime/
    shell/                 stateful shell command runner
    probes/                local and target readiness probes
  tools/
    fs/                    read, grep, glob, edit, write tools
    shell/                 shell tool wrapper
  ui/                      Bubble Tea app, shared model, panels, slash commands
  report/                  summary generation
  configs/                 config loading, state, shared config types
  incubating/factory/      factory schemas, cards, packs (future separate repo)
  test/mocks/              test doubles
  docs/                    architecture, plans, and backlog
```

## Primary Runtime Flow

```text
cmd/mscode
  -> internal/app.Run(...)
  -> internal/app.Wire(...)
  -> ui.New(...)

user input
  -> internal/app.processInput(...)
  -> slash command (/train, /diagnose, /factory status, ...)
     or free text -> runTask(...)

runTask:
  -> compose effective conversation context:
       EngineConfig.SystemPrompt
       + skill summaries (from integrations/skills)
       + any skill content preloaded by /skill as a load_skill tool result
  -> agent/loop.Engine.RunWithContext(task)
  -> tools.Registry
  -> tools/fs or tools/shell
  -> runtime/shell.Runner
  -> loop.Event stream -> model.Event -> ui
```

No orchestrator, no planner, no adapter layer. The app calls the engine
directly. The LLM plans inline within the agent loop.

### Skill activation

`internal/app.Wire(...)` now boots from whatever skills are already available
locally, and the shared repo refresh for `~/.mscode/mindspore-skills` starts
as soon as the startup UI is visible. The refresh remains commit-aware: it stores
the local commit id in the repo directory, compares the local commit with the
remote branch head through a lightweight GitHub API check, uses the built-in
shared repo URL and branch, and asks for `y/n` confirmation inside the UI when
the commits differ. Sync progress and failures are emitted through the main chat
event stream instead of being printed only to the terminal. The synced
`~/.mscode/mindspore-skills/skills` directory remains the highest-priority
skill search path.

Current slash-skill activation is session-visible, not purely task-scoped.
`/skill <name>` and `/<name>` preload the skill by injecting a synthetic
`load_skill` tool call/result into conversation history. If a request is
provided, that request runs immediately; if omitted, the app submits a default
"start this skill now" task so the LLM begins following the skill steps
without waiting for another prompt.

```text
/skill failure-agent
  -> app loads failure-agent SKILL.md from integrations/skills
  -> app injects synthetic load_skill tool call/result into context
  -> app submits a default start request if the user did not provide one
  -> engine runs with base prompt + existing conversation context
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
  context updates. Composes effective system prompt per task
  (base + active skill).

- **`agent/session/`**
  Owns session state, trajectory persistence, and resume reconstruction.

- **`internal/app/` prompt history helpers**
  Own workdir-level prompt recall persistence for `↑/↓`. This is separate
  from session transcript persistence and only seeds the composer's local
  in-memory history on startup.

- **`integrations/skills/`**
  Refreshes the shared skills repo into `~/.mscode/mindspore-skills`,
  lists available skills across configured search paths, and loads one skill
  fully on demand (`SKILL.md` + metadata/frontmatter).

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
cmd/mscode -> internal/app
internal/app -> agent, workflow, ui, configs, integrations, tools, permission
agent -> integrations, permission, configs
workflow -> internal/train, runtime/probes, configs
tools -> runtime, integrations, configs
runtime -> configs
ui -> configs
report -> trace, configs
```

Constraints:

- `cmd/mscode/` stays thin.
- `internal/app/` is the wiring layer, not a reusable core package.
- `agent/` must not depend on `ui/` or `runtime/` directly.
- `workflow/train/` must not import `ui/model`; conversion belongs in
  `internal/app/train.go`.
- `tools/` may call `runtime/`, but `runtime/` must not call `tools/`.
- `configs/` is shared configuration, not a home for application logic.

## Removed Packages (refactor)

The following packages are removed by the refactor (see `docs/impl-guide/mindspore-code-refactor-3.md`):

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

- `docs/impl-guide/mindspore-code-refactor-3.md` — refactor plan (workstream A)
- `docs/impl-guide/ms-skills-whole-update-plan.md` — skills plan (workstream B)
- `docs/impl-guide/ms-factory-struct-v0.1.md` — factory structure and schemas (workstream C)
- `docs/features-backlog.md` — deferred features
- `docs/agent-contributor-guide.md` — contributor conventions

When docs and code disagree, follow the code and update the docs.
