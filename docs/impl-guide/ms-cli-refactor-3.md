# Workstream A: ms-cli Refactor

## Goal
Transform ms-cli from a monolith with hardcoded workflows into a thin agent runtime that loads skills from ms-skills and knowledge from ms-factory.

## Status
- 2026-03-18: Phase A1 implementation landed in code (workflow mode removed; app dispatches tasks directly to `agent/loop.Engine`).

## Phase A1: Remove Workflow Mode

### Files to delete:
- `workflow/executor/demo.go` — DemoExecutor (JSON scenario playback)
- `workflow/executor/executor.go` — Stub executor
- `demo/scenarios/perf_opt.json` — pre-recorded scenario
- `demo/scenarios/train_qwen3_lora.json` — pre-recorded scenario

### Delete `agent/planner/` package entirely:
- `agent/planner/plan.go` — Plan struct, ExecutionMode, Step, ValidateSteps
- `agent/planner/planner.go` — Planner, Plan(), Refine()
- `agent/planner/prompt.go` (if exists) — buildPlanPrompt(), buildRefinePrompt()
- `agent/planner/*_test.go`

### Delete `agent/orchestrator/` package entirely:
- `agent/orchestrator/orchestrator.go` — Orchestrator, Run(), dispatch(), runWorkflow()
- `agent/orchestrator/modes.go` — PlanCallback, NoOpCallback
- `agent/orchestrator/types.go` (if exists) — RunRequest, RunEvent
- `agent/orchestrator/*_test.go`

### Delete `internal/app/adapter.go`:
- engineAdapter type converter between orchestrator and engine (no longer needed)

**Why:** Standard agent CLIs (Claude Code, Cursor, Codex, SWE-agent) have
no separate planner or orchestrator. The architecture is:
`app → engine (LLM + tools loop)` — that's it.

After removing workflow mode, the orchestrator is a passthrough
(`Run()` just calls `agentExec.Execute()`), and the adapter just copies
identical fields between `loop.Event` and `orchestrator.RunEvent`.
Two empty layers.

The LLM plans inline within the agent loop — the system prompt guides it.
Skill selection happens via explicit commands (`/diagnose`, `/optimize`)
or system prompt (lists available skills, LLM picks naturally).

If multi-agent coordination is needed later, add an `agent` tool
(like Claude Code's Agent tool) that spawns a child engine instance.
No orchestrator needed — the LLM coordinates via tool calls.

**`internal/app/wire.go`** — remove orchestrator + planner + workflow wiring
```
- Remove: orchestrator import
- Remove: wfexec import
- Remove: planner import
- Remove: wfexec.NewDemo(), wfexec.NewStub()
- Remove: planner.New() calls
- Remove: orchestrator.New() calls
- Remove: engineAdapter creation
- Remove: demoWf, wf variables
- App now holds *loop.Engine directly (no orchestrator wrapper)
- Add: skill loader initialization
- Add: factory client initialization
```

**`internal/app/run.go`** — single input path, direct engine calls
```
- Remove: runDemoTask() (free-text demo playback via DemoExecutor — dead code)
- Remove: demoInputLoop() (replaced by unified inputLoop)
- Remove: runDemo() (separate demo TUI entry point — no longer needed)
- Remove: orchestrator.RunRequest usage
- Remove: convertRunEvent() (orchestrator.RunEvent → model.Event)
- Add: convertLoopEvent() (loop.Event → model.Event, direct)
- runTask() calls engine.RunWithContext() directly instead of Orchestrator.Run()
- Unify run(): single code path for CLI runtime (no free-text demo mode flag)
- Keep: processInput() routes slash commands (/train) and free text → agent
- Keep: /train command and train demo flow (handled by A5, unchanged)
```

## Phase A2: Extend Skills Integration Layer

Extend the existing `integrations/skills/` package (currently placeholder
interfaces in `repo.go` and `invoke.go`). Not a heavyweight framework,
not ad-hoc file reads in internal/app. Skills are primarily prompt-time
instructions, but the ms-skills repo also contains structured metadata
(skill.yaml) and runtime contracts. This layer reads both.

Scope:
- List available skills from local repo/install
- Load one active skill on demand
- Expose concise summaries for prompt context
- Do NOT solve full skill execution yet

### Extend existing package: `integrations/skills/`

**`integrations/skills/types.go`** — minimal metadata (new file)
```go
type SkillSummary struct {
    Name        string   // directory name, e.g. "cpu-plugin-builder"
    DisplayName string   // from skill.yaml display_name
    Description string   // one-line, from skill.yaml description
    Tags        []string // hint keywords from skill.yaml tags (for display/filtering, not routing)
    EntryType   string   // from skill.yaml, e.g. "manual"
}

type LoadedSkill struct {
    SkillSummary
    Instructions string   // full SKILL.md content (only loaded for active skill)
    Tools        []string // from skill.yaml dependencies.tools
}
```

**`integrations/skills/index.go`** — list skills from repo (new file)
```go
func List(repoDir string) ([]SkillSummary, error)
// Scans skills/*/skill.yaml under repoDir (matches ms-skills repo layout)
// Returns summaries only — no SKILL.md content (avoids context bloat)
```

**`integrations/skills/load.go`** — read one skill fully (new file)
```go
func Load(repoDir, name string) (*LoadedSkill, error)
// Reads skills/<name>/SKILL.md + skills/<name>/skill.yaml
// Only called for the active skill
```

**`integrations/skills/repo.go`** — keep existing RepoSync interface
**`integrations/skills/invoke.go`** — keep existing Invoker interface (revisit later)

### System prompt strategy

Context budget matters. Do NOT inject all full SKILL.md files.

- **All skills**: summary only (name + one-line description)
- **Active skill**: full SKILL.md content
- Tags included only if needed for routing hints

This gives implicit routing context (LLM sees what skills exist) plus
explicit depth for the activated skill, at manageable token cost.

### Config:
Extend existing `SkillsConfig` in `configs/types.go` — do not replace it.
```go
type SkillsConfig struct {
    Repo      string   `yaml:"repo"`       // keep: remote repo URL
    Revision  string   `yaml:"revision"`   // keep: branch/tag
    CacheDir  string   `yaml:"cache_dir"`  // keep: cache for synced repo
    Workflows []string `yaml:"workflows"`  // keep for now, deprecate later
    Path      string   `yaml:"path"`       // NEW: local override path
}
```

```yaml
# configs/mscli.yaml
skills:
  repo: https://github.com/vigo/ms-skills.git
  revision: main
  cache_dir: .cache/skills
  path: ~/work/ms-skills   # optional local override
```

Resolution order: if `path` is set, use it directly. Otherwise fall back
to `cache_dir` (populated by repo sync).

### Wiring:
**`internal/app/wire.go`** — initialize at startup
```go
skillsDir := resolveSkillsDir(config.Skills)  // path > cache_dir
skillSummaries, _ := skills.List(skillsDir)
// summaries available for system prompt
// individual skill loaded on-demand when command activates it
```

## Phase A3: Add Factory Client

### New packages:

**`internal/factory/`** — local card store and resolver
```go
// internal/factory/store.go
type Store struct {
    cacheDir string  // ~/.ms-cli/factory/ or incubating/factory/
}

type Card struct {
    ID       string
    Type     string // "operator", "failure", "perf-feature", etc.
    Level    string // "L0", "L1", etc.
    Path     string
    Tags     []string
    Content  []byte // raw file content
}

func NewStore(cacheDir string) *Store
func (s *Store) Search(cardType string, tags []string) ([]Card, error)
func (s *Store) GetCard(id string) (*Card, error)
func (s *Store) ListCatalog(cardType string) ([]Card, error)
func (s *Store) Status() (*StoreStatus, error)
// Reads manifests/pack.yaml from cacheDir for version/channel/card counts

type StoreStatus struct {
    Version   string            // e.g. "v0.1"
    Channel   string            // e.g. "stable"
    CreatedAt string            // pack creation date
    CardCount map[string]int    // e.g. {"operators": 6, "failures": 3, ...}
}
```

**`integrations/factory/`** — remote fetch and sync
```go
// integrations/factory/updater.go
type Updater struct {
    store    *factory.Store
    remoteURL string
}

func NewUpdater(store *factory.Store, remoteURL string) *Updater
func (u *Updater) Update(channel string) error
// channel: "stable" or "nightly"
// Fetches latest pack from remote → unpacks to store's cacheDir
```

### New commands:
**`internal/app/commands.go`** — add `/factory` subcommands
```go
case "/factory":
    a.cmdFactory(parts[1:])  // routes to subcommand: update, report, status
```

**`internal/app/factory.go`** — factory subcommand implementations
```go
func (a *Application) cmdFactory(args []string)
// /factory status                  — show local version, channel, card counts
// /factory update [stable|nightly] — fetch latest pack
//
// Deferred to features-backlog.md:
//   /factory query  — resolver/query layer for agent tool consumption
//   /factory report — session data collection and report card submission
```

## Phase A4: Evolve Agent Loop for Skills

**`agent/loop/engine.go`** — add skill context support
```go
type Task struct {
    ID           string
    Description  string
    SkillContext string  // NEW: SKILL.md content, merged into system prompt per task
}
```

In `executor.run()` — compose effective system prompt per task:
```go
// Skill context is task-scoped. Compose the effective system prompt
// from base + active skill, then restore base after the run.
// This avoids mutating shared state or leaking old skills into future tasks.
effectivePrompt := ex.engine.ComposeSystemPrompt(ex.task.SkillContext)
ex.engine.ctxManager.SetSystemPrompt(effectivePrompt)
defer ex.engine.ctxManager.SetSystemPrompt(ex.engine.baseSystemPrompt)

ex.engine.ctxManager.AddMessage(llm.NewUserMessage(ex.task.Description))
```

**`agent/loop/engine.go`** — separate base prompt from effective prompt:
```go
type Engine struct {
    baseSystemPrompt string  // set once at init: EngineConfig.SystemPrompt + skill summaries
    // ...
}

// ComposeSystemPrompt builds the effective prompt for a task.
// Base prompt is always included. Active skill appended only if present.
func (e *Engine) ComposeSystemPrompt(skillContext string) string {
    if skillContext == "" {
        return e.baseSystemPrompt
    }
    return e.baseSystemPrompt + "\n\n## Active Skill\n" + skillContext
}
```

**Key design points:**
- `baseSystemPrompt` is immutable after init (from EngineConfig.SystemPrompt + skill summaries)
- `SetSystemPrompt()` on ctxManager changes what goes to the LLM API
- `defer` restores base prompt after each task — no skill leakage
- If next task has a skill, it composes fresh; if not, base is already restored

**Concurrency assumption:** Phase A4 assumes at most one interactive
free-text agent task runs at a time. The shared engine/context manager
is therefore safe for task-scoped skill prompt composition.

**Enforcement:** `runTask()` should check whether an agent task is already
running and reject or queue the new request (e.g. "agent is busy, please
wait"). A simple `sync.Mutex` or `atomic` busy flag is sufficient.
If concurrent agent tasks are added later, each task must get an isolated
engine/context.

**`internal/app/run.go`** — task dispatch (no orchestrator)

Skill context is **command-scoped**, not sticky. No persistent skill state
on Application. Explicit commands build a task with skill context; plain
free text builds a task with no skill context.

```go
// Free text — no skill context
func (a *Application) runTask(description string) {
    task := loop.Task{
        ID:          generateTaskID(),
        Description: description,
        // SkillContext empty — base prompt only
    }
    events, err := a.Engine.RunWithContext(ctx, task)
    // convert loop.Event → model.Event → EventCh
}

// Explicit command — e.g. /diagnose my training crashed
func (a *Application) cmdDiagnose(args []string) {
    skill, err := skills.Load(a.skillsDir, "failure-agent")
    if err != nil {
        a.EventCh <- model.Event{
            Type:    model.ToolError,
            Message: fmt.Sprintf("failed to load failure-agent skill: %v", err),
        }
        return
    }
    task := loop.Task{
        ID:           generateTaskID(),
        Description:  strings.Join(args, " "),
        SkillContext: skill.Instructions,  // full SKILL.md for this task only
    }
    events, err := a.Engine.RunWithContext(ctx, task)
    // convert loop.Event → model.Event → EventCh
}
```

**No sticky state:** skill context lives on the Task, not on Application.
Each task is self-contained. No `/skill clear`, no `/skills status`,
no leakage between tasks. User re-uses the command for repeated skill use.

If persistent "mode" behavior is needed later, add it as an explicit
mode toggle, not as implicit sticky skill state.

### Base system prompt composition

For free text, the base system prompt includes skill summaries so the LLM
has awareness of available capabilities. This is not deterministic skill
routing — the LLM may or may not leverage skill knowledge depending on
the query. For reliable skill activation, use explicit commands.

Composed at startup in `wire.go` / app init:

```go
basePrompt := EngineConfig.SystemPrompt
+ "\n\n## Available Skills\n"
+ formatSkillSummaries(skillSummaries)  // name + one-line description each
```

This base prompt is always present. When a command activates a skill,
the engine appends `## Active Skill` + full SKILL.md to the effective
prompt for that task only (see engine.go above).

## Phase A5: Keep /train During Transition

Don't remove `/train` and `workflow/train/` yet. They coexist:
- `/train` keeps working as-is — train demo flow (DemoBackend) is unaffected
- Free text goes through the new skill-aware agent path (requires LLM)
- Gradually, as agent-skills mature, `/train` features migrate to skills
- Eventually `/train` becomes a thin router that activates setup-agent

### Files preserved (no changes):
- `workflow/train/controller.go`
- `workflow/train/types.go`
- `workflow/train/demo.go`
- `workflow/train/setup.go`
- `workflow/train/run.go`
- `internal/app/train.go` (all of it)
- `runtime/probes/` (all)
- `ui/` (all)
