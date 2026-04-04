# refactor-arch-1 TODO

## Completed

- [x] Build `agent/planner/` (planner, step, parser, prompt, validator, 9 tests)
- [x] Resolve `app/` vs `internal/app/` — moved all logic to `internal/app/`, created `cmd/mscli/main.go`, deleted `app/`
- [x] Build `agent/orchestrator/` — mode dispatch (standard/plan, 4 tests)
- [x] Slim down `engine.go` — pure ReAct loop (662→295 lines), removed plan/review, deleted `loop/permission.go`
- [x] Delete `agent/plan/` (5 files, ~1,600 lines)
- [x] Delete `executor/` (dead code)
- [x] Delete empty `doc.go` stub packages (10 packages: runtime/*, workflow/*, ui/app, ui/views, agent/router)
- [x] Wire orchestrator into `internal/app/` — tasks route through orchestrator, `/mode` command added
- [x] Extract `runtime/shell/` from `tools/shell/` — stateful execution in runtime, thin LLM wrapper in tools
- [x] Decide `tools/` vs `runtime/` boundary — tools = stateless LLM schema, runtime = stateful execution
- [x] Update `CLAUDE.md` — reflects current packages, call chain, tools/runtime boundary, planned packages
- [x] `go build ./...` passes, `go test ./...` all green, demo verified

## Issues Fixed

| Issue | Resolution |
|---|---|
| Ghost code (old `app/`, `agent/plan/`, `executor/`) | Deleted all. Logic lives in `internal/app/`, `agent/planner/`, `agent/orchestrator/` |
| God Object `engine.go` (660 lines) | Slimmed to ~295 lines. Plan/review orchestration → `agent/orchestrator/` |
| `agent/loop` imports `agent/plan` (circular risk) | `agent/plan/` deleted. Orchestrator sits above loop, no circular deps |
| `tools/` vs `runtime/` unclear | `runtime/shell/` = execution, `tools/shell/` = LLM wrapper. Rule: needs workspace/env/timeout → runtime |
| Stub `doc.go` packages | All deleted. CLAUDE.md updated: "do NOT create packages for unbuilt features" |
| `executor/runner.go` dead code | Deleted |

## Deferred (future phases)

- [ ] Build `workflow/engine/` — execute `[]planner.Step` with DAG, retry. Not needed yet: orchestrator has `runPlanViaLoop` fallback
- [ ] Build `runtime/workspace/` — isolated working directories. When workflow needs workspace isolation
- [ ] Build `runtime/artifacts/` — user-deliverable output files. When workflow produces deliverables
- [ ] Build `agent/router/` — skill matching from `mindspore-skills`. When skill system is needed
