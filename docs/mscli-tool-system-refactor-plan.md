# ms-cli Tool System Refactor Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve tool hit rate, reduce invalid tool arguments, add the most valuable missing tools, and evolve the tool runtime toward a clearer Claude Code-inspired architecture without overcomplicating `ms-cli`.

**Architecture:** Keep the current lightweight `Tool` interface as the base, but add missing layers in order: better tool prompting and schema clarity first, then missing high-value tools, then execution/runtime refactors. The plan deliberately separates "better model behavior" improvements from "bigger architecture" work so `ms-cli` can gain quality quickly without taking on a full runtime rewrite.

**Tech Stack:** Go, current `tools.Registry`, `agent/loop.Engine`, `integrations/llm` codecs, `permission` service, Bubble Tea UI.

---

## Direct Answers

### 1. Which existing-vs-Claude gaps are worth filling?

Not every Claude Code tool is worth copying. For `ms-cli`, the most useful additions are:

- **`list_dir` / `ls`-style tool**
  Current gap: `glob` can find files by pattern, but it is not a strong "show me what is here" tool.
  Why it matters: improves exploration hit rate and reduces unnecessary `shell` calls.
  Priority: **high**

- **`multi_edit` or `apply_patch`-style editing tool**
  Current gap: `edit` only supports one exact replacement per call, which is brittle and increases retry loops.
  Why it matters: directly improves code-edit success rate and lowers tool-call count.
  Priority: **high**

- **`web_fetch` / `fetch_url` tool**
  Current gap: external docs, release notes, and troubleshooting references currently depend on `shell` usage.
  Why it matters: `ms-cli` targets ML infra and training workflows; external documentation lookup is common and should not require generic shell access.
  Priority: **high**

- **External tool adapter / MCP-compatible bridge**
  Current gap: permissions already understand `mcp__...`, but runtime does not.
  Why it matters: this is the natural long-term extension path.
  Priority: **medium-long term**

- **Subagent / task tool**
  Current gap: no decomposition mechanism inside the tool runtime.
  Why it matters: useful for larger workflows, but not necessary before the basics are fixed.
  Priority: **low for now**

### 2. Do current tool descriptions need optimization?

Yes. This is the easiest high-leverage change.

Current problems:

- descriptions are correct, but too generic
- they often describe what the tool does, but not when to prefer it over another tool
- they do not consistently encode decision boundaries
- parameter descriptions are sometimes weaker than runtime behavior
- some constraints are split across tool descriptions and the global system prompt

Highest-priority descriptions to improve:

- `write`: must strongly reinforce exact field names and overwrite semantics
- `edit`: must strongly reinforce exact-match behavior and when not to use it
- `shell`: must explain when shell is the fallback instead of the default
- `load_skill`: must better explain invocation boundary
- `grep` / `glob`: should clarify selection boundary between "search content" and "find files"

### 3. Does the current tool system architecture need optimization?

Yes, but not all at once.

Judgment:

- **Prompting/tool-description quality:** optimize immediately
- **Necessary missing tools:** add next
- **Architecture refactor:** start only after the tool contract and tool catalog are cleaner

The architecture is good enough for a compact agent, but it is already showing three pressure points:

- too much tool-specific logic lives in `agent/loop/engine.go`
- schema validation is too weak and too fragmented
- registry and model-visible tool set are still the same thing

So the right move is phased optimization, not a rewrite.

## Current Tool Catalog Review

### Existing tools

- `read`
- `write`
- `edit`
- `grep`
- `glob`
- `shell`
- `load_skill`

### Coverage assessment

What this set already covers well:

- basic code exploration
- single-file read and write
- exact string replacement edits
- shell fallback for everything else
- skill activation

What it covers poorly compared with Claude Code:

- directory structure inspection as a first-class action
- multi-change editing
- safe external information retrieval
- external tool extensibility
- richer tool-specific runtime behavior

## Tool Description Optimization Plan

This is the first and easiest phase because it improves tool hit rate without changing architecture.

### Description design rules

Each tool description should answer five things:

1. What the tool does
2. When to use it
3. When not to use it
4. Which sibling tool is preferable in nearby cases
5. Which arguments must be exact

### Recommended improvements by tool

#### `read`

Need to clarify:

- use `read` when you already know the file path
- use `glob` to find files first
- use `grep` when searching by content
- `path` is workspace-relative

#### `write`

Need to clarify:

- this overwrites the full file content
- prefer `edit` or `multi_edit` for targeted changes
- arguments must include exact keys `path` and `content`
- do not use aliases if aliases remain temporarily supported

#### `edit`

Need to clarify:

- best for one targeted exact replacement
- do not use when multiple distant changes are needed
- `old_string` must match exactly including whitespace and newlines
- if uniqueness is unclear, read the file first

#### `grep`

Need to clarify:

- use to search file contents, not filenames
- prefer `glob` for file discovery
- prefer `read` after locating the exact file
- `include` narrows file types

#### `glob`

Need to clarify:

- use to find candidate files by filename or path pattern
- not a replacement for content search
- use `path` to scope the search root
- use `offset`/`limit` when results are large

#### `shell`

Need to clarify:

- use shell only when dedicated tools are insufficient
- prefer `read`/`grep`/`glob`/`edit`/`write` for file operations
- command should be one focused operation
- destructive or high-risk commands may require confirmation

#### `load_skill`

Need to clarify:

- use when the user's task clearly matches a listed skill
- do not call it for ordinary coding actions that existing tools already support
- load once when needed rather than repeatedly

### Prompt cleanup rule

After tool descriptions improve, the global system prompt should shrink.

Keep in the system prompt:

- high-level behavior policy
- minimal workflow heuristics

Remove from the system prompt over time:

- duplicated per-tool contract details
- exact argument warnings that belong in schema/description

## Missing Tool Roadmap

### Phase A: Add easy, high-value tools

#### 1. `list_dir`

Why first:

- simple implementation
- directly reduces `shell ls/find/tree`
- gives the model a clearer exploration primitive than `glob`

Suggested shape:

- `path`
- `depth`
- `include_hidden`

Expected behavior:

- return files and directories distinctly
- preserve stable ordering
- optionally summarize counts

#### 2. `multi_edit`

Why next:

- biggest quality win for coding tasks
- directly addresses brittleness in current `edit`

Suggested shape:

- `path`
- `edits[]` with `old_string` and `new_string`

Expected behavior:

- validate all edits before writing
- fail safely if any edit is ambiguous
- apply in a deterministic order

### Phase B: Add medium-complexity capability tools

#### 3. `web_fetch`

Why:

- recurring need for docs, release notes, dependency changes, training/runtime troubleshooting
- safer and more structured than `shell` + `curl`

Suggested shape:

- `url`
- `prompt` or `extract`

Expected behavior:

- fetch and return cleaned text
- include source URL in the result
- integrate with permission policy if external access is controlled

### Phase C: Add extension-oriented tools

#### 4. External tool adapter / MCP bridge

Why:

- aligns with the existing permission-rule vocabulary
- creates a long-term extensibility path

Suggested scope:

- list external tools
- adapt external schema into `ms-cli` tool definitions
- run through shared permission and result handling

#### 5. Subagent / task tool

Why later:

- only valuable once the core runtime is stable
- high complexity and architecture surface area

## Architecture Optimization Plan

This section is ordered from easiest to hardest.

### Phase 1: Contract cleanup

#### 1. Make schema and runtime contract match exactly

Current issue:

- `write` accepts undocumented fallback keys
- engine still contains tool-name-specific argument extraction logic

Goal:

- one published contract
- one accepted contract

#### 2. Add shared validation hooks

Add optional interfaces:

```go
type ValidatingTool interface {
    ValidateInput(ctx context.Context, params json.RawMessage) error
}
```

Goal:

- move semantic validation out of arbitrary `Execute` bodies
- create a shared tool runtime frontier before side effects happen

#### 3. Add tool-owned permission context

Add optional interface:

```go
type PermissionAwareTool interface {
    PermissionContext(params json.RawMessage) (action string, path string, err error)
}
```

Goal:

- stop forcing the engine to infer tool semantics from tool names

### Phase 2: Runtime extraction

#### 4. Extract tool execution out of `agent/loop/engine.go`

Create a dedicated runtime path such as:

- lookup tool
- validate input
- derive permission context
- request permission
- execute
- normalize result
- append tool result message
- emit events

Goal:

- make the engine smaller
- make tool behavior more testable
- prepare for orchestration and hook points

#### 5. Separate registry from visible tool set

Current state:

- registry equals exposed tools

Target state:

- registry stores available tools
- a request-scoped selector decides visible tools

Why this matters:

- provider-specific filtering
- mode-specific filtering
- future deferred loading
- future external tools

### Phase 3: Capability scaling

#### 6. Upgrade schema representation

Current issue:

- `llm.ToolSchema` is too small for future tools

Target:

- richer JSON Schema support

Why:

- arrays for `multi_edit`
- nested objects for better contracts
- more accurate provider encoding

#### 7. Add orchestration metadata

Introduce optional tool metadata such as:

```go
type ConcurrencyAwareTool interface {
    ConcurrencyMode() string
}
```

Initial policy:

- `read`, `grep`, `glob`, `list_dir`, `web_fetch`: concurrency-safe
- `write`, `edit`, `multi_edit`, `shell`: serial

#### 8. Add extension/adaptation layer

Only after the previous steps:

- external tools
- MCP-style schema adaptation
- unified runtime wrapping

## Recommended Execution Order

This is the actual roadmap from easiest to hardest.

### Stage 1: Prompt and description quality

- Rewrite all tool descriptions with explicit decision boundaries
- Strengthen parameter descriptions
- Reduce duplicated tool guidance in `DefaultSystemPrompt()`
- Keep only high-level workflow rules in the system prompt

Expected outcome:

- better tool selection
- fewer malformed arguments
- fewer unnecessary shell calls

### Stage 2: Contract correctness

- Remove schema/runtime drift, especially in `write`
- Add validation hooks
- Add tool-owned permission context

Expected outcome:

- fewer invalid executions
- clearer runtime behavior
- less engine heuristic logic

### Stage 3: High-value tool additions

- Add `list_dir`
- Add `multi_edit`
- Add `web_fetch`

Expected outcome:

- better repository navigation
- better edit success rate
- less misuse of shell for external lookup

### Stage 4: Runtime refactor

- Extract tool execution into a runtime module
- Separate available tools from exposed tools

Expected outcome:

- cleaner architecture
- easier testing
- better future extensibility

### Stage 5: Long-term architecture

- Upgrade schema expressivity
- Add concurrency/orchestration metadata
- Add external tool adapter / MCP-like layer

Expected outcome:

- Claude Code-style evolution path without immediate overengineering

## Files Most Likely To Change

### Prompt and tool contract

- `agent/loop/engine.go`
- `tools/fs/read.go`
- `tools/fs/write.go`
- `tools/fs/edit.go`
- `tools/fs/grep.go`
- `tools/fs/glob.go`
- `tools/shell/shell.go`
- `tools/skills/load_skill.go`

### Runtime and architecture

- `tools/types.go`
- `tools/registry.go`
- `agent/loop/engine.go`
- `integrations/llm/provider.go`
- `integrations/llm/codec_anthropic.go`
- `integrations/llm/codec_openai_completion.go`
- `integrations/llm/codec_openai_responses.go`
- `permission/service.go`

### New tools

- `tools/fs/list_dir.go` or similar
- `tools/fs/multi_edit.go` or similar
- `tools/webfetch/...` or similar
- `runtime/webfetch/...` if runtime separation is desired

## Final Recommendation

The right order is:

1. **Fix tool prompting and descriptions first**
2. **Then fix contract correctness and validation**
3. **Then add `list_dir`, `multi_edit`, and `web_fetch`**
4. **Only then refactor the runtime architecture**

This order gives `ms-cli` visible quality gains quickly, while keeping the architecture changes justified by real tool pressure instead of abstract design purity.
