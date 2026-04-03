# AI Contributor Guide

This guide contains repository rules that should apply to all AI coding agents working in `mscode`.

Agent-specific files at the repo root, such as `AGENTS.md` and `CLAUDE.md`, should stay thin and point here for shared project policy.

## Source Of Truth

- Treat the current code layout as authoritative.
- Use [`docs/arch.md`](./arch.md) for the broader architecture map.
- If this guide and the code disagree, follow the code and update the docs.

## Working Approach

- Read the relevant code before editing it.
- Prefer small, targeted changes over large refactors.
- Preserve package boundaries and ownership unless the maintainer explicitly asks for structural changes.
- Do not add new top-level packages without approval.
- Do not create placeholder packages for planned work.

## Build And Validation

Use the narrowest command that proves the change, then widen if needed:

```bash
go test ./...
go build ./...
go vet ./...
```

Useful entrypoint commands:

```bash
go build -o mscode ./cmd/mscode
go run ./cmd/mscode
```

## Code Style

- Use standard Go formatting with `gofmt`.
- Keep error messages lowercase and without trailing punctuation.
- Wrap errors with context using `fmt.Errorf("context: %w", err)`.
- Prefer extending existing types and flows over creating parallel abstractions.
- Keep `cmd/mscode` thin and push behavior into the owning package.
- Prefer returning `error` over `panic` except for truly unrecoverable states.

## Current Repository Shape

This summary matches the current tree in this checkout.

```text
mindspore-code/
  cmd/mscode/              process entrypoint
  internal/app/            bootstrap, wiring, commands, startup, train flow
  internal/inputhistory/   prompt recall persistence for the chat composer
  internal/project/        roadmap and weekly helpers
  internal/train/          training types and target abstraction
  agent/
    context/               context window management and compaction
    loop/                  ReAct execution loop
    memory/                memory store, retrieval, policy
    session/               session state and persistence
  workflow/
    train/                 train lane controller, setup, run, demo backend
  integrations/
    domain/                domain client and schema
    llm/                   provider registry and OpenAI/Anthropic clients
    skills/                skill repository and invocation integration
  permission/              permission service and types
  runtime/
    shell/                 stateful shell runner
    probes/                unified probe result model
      local/               local-side readiness probes (os, aiframework, algo)
      target/              remote target readiness probes (os, ai, algo, workdir, gpu, npu)
        ssh/               SSH connectivity probe
  tools/
    fs/                    filesystem tool implementations
    shell/                 shell tool wrapper
  ui/                      Bubble Tea app, panels, slash commands, model
  agent/session/           session state, trajectory persistence, resume
  report/                  summary generation
  configs/                 config loading and shared config types
  test/mocks/              test doubles
  docs/                    project docs
```

## Runtime Flow

The current primary runtime path is:

```text
cmd/mscode -> internal/app -> agent/loop -> tools -> runtime/shell
```

Important current details:

- `cmd/mscode/main.go` only delegates to `internal/app.Run(...)`.
- `internal/app` is the composition root and owns wiring plus event conversion.
- `internal/app` dispatches free-text tasks directly into `agent/loop.Engine`.
- `tools/` exposes LLM-callable tool surfaces; `runtime/shell/` owns stateful command execution.

## Dependency Boundaries

Keep dependencies flowing downward only. Avoid upward or circular imports.

```text
cmd/mscode -> internal/app -> agent, workflow, ui
agent -> permission, integrations, configs
workflow -> internal/train, runtime/probes, configs
workflow/train -> internal/train, runtime/probes (NOT ui/model)
runtime/probes -> internal/train
tools -> runtime, integrations, configs
runtime -> configs
permission -> configs
integrations -> configs
report -> trace, configs
ui -> configs
```

Package rules:

- `cmd/mscode/` should call `internal/app` only.
- `internal/app/` is the wiring layer and should not become a reusable dependency for the rest of the repo.
- `internal/inputhistory/` owns persisted prompt recall only. Do not reuse it for resume or transcript reconstruction.
- `internal/app/train.go` maps train lane events to UI state updates — it is the only place that bridges `workflow/train` and `ui/model`.
- `agent/` must not depend directly on `ui/` or `runtime/`.
- `agent/` should use tools or interfaces rather than reaching into execution infrastructure directly.
- `workflow/train/` must NOT import `ui/model` — event conversion happens in `internal/app/train.go`.
- `workflow/train/` owns train-specific sequencing (setup, run). Demo backend stays in `workflow/train/demo.go`.
- `runtime/probes/*` only perform checks — they do not own train sequencing or emit UI events.
- `tools/` may call `runtime/`, but `runtime/` must not call `tools/`.
- `ui/` must remain a consumer of events, not a dependency of lower layers.
- `configs/` should remain shared configuration/types, not an application logic layer.
- Do not add separate planner/orchestrator layers; keep runtime path app -> engine.

## Stable Contracts

Do not make breaking changes to foundational contracts without explicit approval. Extend them instead.

Examples:

- `permission.PermissionLevel` and `permission.PermissionDecision`
- LLM provider interfaces and tool schema types under `integrations/llm`
- loop task/event transport types under `agent/loop`
- session state and persistence types under `agent/session`

## Skills And External Integrations

- The repo contains skill integration code under `integrations/skills/`.
- Skill definitions themselves should not be treated as living in this repo unless the maintainer explicitly adds that responsibility.
- Keep direct LLM provider logic inside `integrations/llm/`.

## What To Avoid

- Do not merge unrelated layers just to save files.
- Do not bypass permission checks for tool or shell execution.
- Do not import UI packages from agent, tools, or runtime layers.
- Do not split session persistence into parallel trace-style subsystems.
- Do not make `ui/components` own disk persistence logic when app-layer wiring can keep the component stateful but storage-agnostic.
- Do not update docs by copying stale structure; verify against the current tree first.

## Root Agent Files

Recommended pattern:

- `AGENTS.md`: thin Codex/OpenAI wrapper that points here and adds only agent-specific notes.
- `CLAUDE.md`: thin Claude wrapper that points here and adds only Claude-specific notes.
- Future files such as `GEMINI.md` or `CURSOR.md`: same pattern.

Keep shared repository policy in this guide to avoid drift across agent-specific files.
