# MindSpore CLI Architecture

## Two-Repo Model

```text
mindspore-cli (this repo)    runtime — TUI, agent loop, tool registry
mindspore-skills             instructions — SKILL.md + skill.yaml per skill
```

- `mindspore-cli` loads skills from `mindspore-skills` at build time (embedded in binary)
- Skills are portable across CLIs (Claude Code, OpenCode, Gemini CLI, Codex)
- `mindspore-cli` is the official end-to-end entrypoint
- `mindspore-skills` is the reusable capability layer

## Top-Level Shape

```text
mindspore-cli/
  cmd/mscli/              process entrypoint
  cmd/mscli-server/       bug/issue/project server
  internal/
    app/                   composition root, startup, commands, UI bridging
    server/                HTTP API for bugs, issues, projects
    project/               roadmap and weekly status helpers
    train/                 train request and target types
    update/                binary update checker
    workspacefile/         workspace path validation
  agent/
    context/               context window management and compaction
    loop/                  ReAct-style execution engine (the core runtime)
    memory/                memory store, retrieval, and policy
    session/               session state and persistence
  workflow/
    train/                 train lane controller, setup, run, demo backend
  integrations/
    llm/                   unified provider manager (openai-completion/openai-responses/anthropic)
    skills/                skill listing, loading, and metadata (embedded at build time)
  permission/              permission policy, types, store, safe command allowlist
  runtime/
    shell/                 stateful shell command runner
    probes/                local and target readiness probes
  tools/
    fs/                    read, grep, glob, edit, write tools
    shell/                 shell tool wrapper
    skills/                skill loading tool
  ui/                      Bubble Tea app, shared model, panels, slash commands
  configs/                 config loading, state, shared config types
  scripts/                 release, install, mirror deploy scripts
  docs/                    architecture and contributor guide
```

## Primary Runtime Flow

```text
cmd/mscli
  -> internal/app.Run(...)
  -> internal/app.Wire(...)
  -> ui.New(...)

user input
  -> internal/app.processInput(...)
  -> slash command (/diagnose, /fix, /model, ...)
     or free text -> runTask(...)

runTask:
  -> compose effective conversation context:
       EngineConfig.SystemPrompt
       + skill summaries (from integrations/skills)
       + any skill content preloaded by /skill
  -> agent/loop.Engine.RunWithContext(task)
  -> tools.Registry
  -> tools/fs or tools/shell
  -> runtime/shell.Runner
  -> loop.Event stream -> model.Event -> ui
```

No orchestrator, no planner, no adapter layer. The app calls the engine
directly. The LLM plans inline within the agent loop.

### Provider And Model Selection

`mscli` now converges provider connection and model selection into `/model`:

```text
/model
  -> merged provider catalog:
       builtin MindSpore CLI Free
       + models.dev cache/remote catalog
       + ~/.mscli/config.json extra_providers
  -> provider pane adds or refreshes connected providers in ~/.mscli/auth.json
  -> model pane loads usable providers:
       MindSpore CLI Free when logged in
       + providers connected in ~/.mscli/auth.json
  -> persist active/recent/favorite model refs in ~/.mscli/model.json
  -> translate logical provider selection into current configs.ModelConfig
  -> Application.SetProvider(...)
```

Local state files:

- `~/.mscli/credentials.json`: MindSpore CLI login only
- `~/.mscli/auth.json`: connected provider auth only
- `~/.mscli/model.json`: active/recent/favorite model refs
- `~/.mscli/cached/models-dev-api.json`: models.dev cache fallback
- `~/.mscli/config.json`: deprecated model compatibility plus `extra_providers`

### Skill activation

Skills are embedded in the binary at build time from the `mindspore-skills` repo.
The `scripts/update-skills.sh` script pulls the latest skills before each release build.

Skill activation is session-visible. `/skill <name>` and `/<name>` preload the
skill by injecting a synthetic `load_skill` tool call/result into conversation
history.

```text
/skill failure-agent
  -> app loads failure-agent SKILL.md from integrations/skills/builtin
  -> app injects synthetic load_skill tool call/result into context
  -> app submits a default start request if the user did not provide one
  -> engine runs with base prompt + existing conversation context
```

Free text uses the base system prompt which includes skill summaries
(name + one-line description). For reliable skill activation, use explicit commands.

## Package Responsibilities

- **`internal/app/`**
  Loads config, wires dependencies, starts the TUI, handles slash commands,
  dispatches tasks to the engine, and converts `loop.Event` to `ui/model.Event`.

- **`agent/loop/`**
  The core runtime. Runs the LLM/tool loop: tool calling, permission checks,
  context updates. Composes effective system prompt per task.

- **`agent/session/`**
  Owns session state, trajectory persistence, and resume reconstruction.

- **`integrations/skills/`**
  Lists available skills, loads one skill fully on demand (`SKILL.md` + metadata).
  Skills are embedded in `integrations/skills/builtin/` at build time.

- **`integrations/llm/`**
  Unified LLM provider interface. Supports OpenAI Chat Completions,
  OpenAI Responses, and Anthropic Messages protocols.

- **`tools/`**
  LLM-callable tool surfaces (filesystem, shell, skill loading). Stateless tool definitions.

- **`runtime/shell/`**
  Stateful command runner with workspace, timeout, and safety checks.

- **`permission/`**
  Permission decisions and persistence. Safe command allowlist for auto-approving
  read-only commands (ls, cat, grep, git, etc.).

- **`ui/`**
  Bubble Tea inline-mode interface. Consumes events, renders panels.
  Not imported by lower layers.

- **`configs/`**
  Shared configuration types and loaders. Reads `MSCLI_*` environment variables.

## Dependency Boundaries

```text
cmd/mscli -> internal/app
internal/app -> agent, workflow, ui, configs, integrations, tools, permission
agent -> integrations, permission, configs
workflow -> internal/train, runtime/probes, configs
tools -> runtime, integrations, configs
runtime -> configs
ui -> configs
```

Constraints:

- `cmd/mscli/` stays thin.
- `internal/app/` is the wiring layer, not a reusable core package.
- `agent/` must not depend on `ui/` or `runtime/` directly.
- `tools/` may call `runtime/`, but `runtime/` must not call `tools/`.
- `configs/` is shared configuration, not a home for application logic.

When docs and code disagree, follow the code and update the docs.
