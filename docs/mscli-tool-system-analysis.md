# ms-cli Tool System Analysis

## Scope

This document analyzes the current `ms-cli` tool system from three angles:

1. Current-state design of the tool system
2. Key differences versus Claude Code
3. Concrete improvements `ms-cli` can borrow from Claude Code

The document is written in phases so the analysis can be refined incrementally.

## Phase 1: Current-State Analysis

### 1. Architectural Position

`ms-cli` currently places the tool system directly inside the main agent loop. The overall path is:

`Application wiring -> tool registry -> LLM tool schema export -> model tool call -> permission check -> tool execution -> tool result appended back into context`

Relevant files:

- `internal/app/wire.go`
- `tools/types.go`
- `tools/registry.go`
- `agent/loop/engine.go`
- `permission/service.go`

This is a relatively direct architecture. There is no separate tool runtime layer, no separate tool exposure layer, and no separate tool orchestration layer.

### 2. Tool Object Model

The core abstraction is intentionally thin:

```go
type Tool interface {
    Name() string
    Description() string
    Schema() llm.ToolSchema
    Execute(ctx context.Context, params json.RawMessage) (*Result, error)
}
```

Optional streaming support is added through a second interface:

```go
type StreamingTool interface {
    ExecuteStream(ctx context.Context, params json.RawMessage, emit func(StreamEvent)) (*Result, error)
}
```

Observations:

- The abstraction is easy to understand and easy to implement.
- Model-facing metadata and runtime behavior are partially unified, but only at a minimal level.
- The interface does not encode permission hints, read-only/destructive classification, concurrency safety, input validation stages, result mapping policy, or UI rendering behavior.
- As a result, responsibilities that Claude Code keeps inside a richer `Tool` contract are distributed across the engine, permission service, and individual tools.

### 3. Tool Registry and Tool Pool

`tools.Registry` is a simple ordered registry:

- `Register`
- `Get`
- `List`
- `Names`
- `ToLLMTools`

Current behavior:

- Tools are statically registered during application wiring.
- The exported tool set is effectively the full registry on every LLM request.
- Tool order is preserved by registration order.
- There is no request-scoped filtering, no delayed exposure, and no dynamic tool discovery step.

Current built-in set assembled in `internal/app/wire.go`:

- `read`
- `write`
- `edit`
- `grep`
- `glob`
- `shell`
- `load_skill`

This is simple and stable, but it also means the system does not yet distinguish:

- available tools vs exposed tools
- built-in tools vs external tools
- lightweight tools vs heavyweight tools
- always-load tools vs defer-load tools

### 4. Tool Schema Design

`ms-cli` uses a small custom schema model in `integrations/llm/provider.go`:

```go
type ToolSchema struct {
    Type       string
    Properties map[string]Property
    Required   []string
}

type Property struct {
    Type        string
    Description string
    Enum        []string
}
```

This is enough for simple object-shaped tools, but it is significantly less expressive than full JSON Schema.

Current strengths:

- easy to serialize
- easy to reason about
- provider-independent

Current limits:

- no nested object schema modeling
- no array item schema
- no unions / `oneOf`
- no numeric or string constraints
- no schema-level default/nullable/pattern controls
- no separate protocol fields like `strict` or `defer_loading` at the tool-definition level

There is also an important contract gap between schema and runtime behavior:

- `write` advertises only `path` and `content`
- runtime still accepts `file_path` and `filename` as fallback aliases

That means the true accepted input contract is broader than the model-visible schema.

### 5. Execution Flow

The execution path in `agent/loop/engine.go` is:

1. Build request with `ctxManager.GetMessages()` and `tools.ToLLMTools()`
2. Send request to provider
3. Persist assistant message and tool calls into context
4. For each returned tool call, execute serially
5. Append tool output as a tool message
6. Continue the loop

Important properties of the current design:

- tool calls are executed serially
- there is no tool batching or concurrency scheduler
- there is no separate pre-hook / post-hook pipeline
- there is no standardized `validateInput` stage before execution
- tool result mapping is lightweight and mostly string-based

The runtime is therefore straightforward, but thin:

- tool lookup happens in the engine
- permission checking happens in the engine
- input parsing is delegated to each tool
- semantic validation is delegated to each tool
- output formatting is delegated to each tool

### 6. Permission Design

`ms-cli` already has a meaningful permission subsystem, and this is one of the stronger parts of the design.

The decision chain in `permission/service.go` is broadly:

1. check tool-level permission
2. for `shell`, also check command-level permission
3. check path-level permission
4. if result is `ask`, optionally invoke interactive UI
5. remember session or persisted decisions when applicable

Notable strengths:

- explicit levels: `deny`, `ask`, `allow_once`, `allow_session`, `allow_always`
- shell command policy and path policy are both supported
- rules support Claude-like forms such as `Bash(...)`, `Read(...)`, `Edit(...)`
- dangerous shell commands are recognized separately
- edit/write can be granted together as a session-level convenience

Important limits versus Claude Code:

- permission logic is centralized, but tools do not provide their own permission metadata or tool-specific `checkPermissions`
- the permission pipeline is not integrated with hooks, classifiers, or sandbox-aware execution policy
- permission and tool schema are mostly separate worlds
- non-shell tools do not contribute rich action metadata beyond path extraction in the engine

### 7. Tool Result Model

The tool result shape is:

```go
type Result struct {
    Content string
    Summary string
    Meta    map[string]any
    Error   error
}
```

This is a pragmatic format, but it still collapses most tool results into free-form text.

Current implications:

- the agent loop can easily append results back into context
- the UI can render basic summaries
- some tools can attach metadata, such as edit/write diff meta

But compared with Claude Code:

- result typing is much weaker
- there is no dedicated result-to-transcript mapping layer
- large-output persistence/truncation is owned by tool/runtime code, not by a dedicated tool-result subsystem

### 8. Prompt Coupling

The current system prompt in `agent/loop/engine.go` hardcodes the tool list and some tool-usage rules.

This creates a duplication problem:

- registry is the actual source of tool availability
- system prompt separately lists tool names and rules

Consequences:

- adding a new tool requires both registry wiring and prompt maintenance
- prompt drift is possible
- the prompt cannot naturally reflect request-scoped tool visibility

### 9. Strong Parts of the Current Design

The current `ms-cli` design is not primitive; it is deliberately compact. Its strengths are:

- easy-to-follow control flow
- low conceptual overhead for contributors
- centralized permission service
- streaming support already exists for `shell`
- session/context integration is direct
- tools are small and easy to write

For a focused CLI agent, this gives good implementation velocity.

### 10. Main Structural Weaknesses

The main design pressure points are:

1. The `Tool` abstraction is too thin for future scale.
2. Schema expressivity is too weak for richer tools.
3. Runtime validation is fragmented inside individual tools.
4. Tool exposure is static and all-tools-by-default.
5. Prompt content duplicates tool registry knowledge.
6. There is no orchestration layer for concurrency or scheduling.
7. There is no native extension path equivalent to Claude Code's MCP adaptation layer.

## Phase 2: Differences Versus Claude Code

### 1. Tool Protocol Richness

Claude Code uses a much richer tool object model. In addition to name, prompt, and schema, its tool objects carry execution behavior, permission-related behavior, schema conversion policy, concurrency hints, and UI/result-rendering behavior.

`ms-cli` currently keeps only:

- name
- description
- schema
- execute

Difference summary:

- `ms-cli` optimizes for simplicity
- Claude Code optimizes for a unified runtime contract

### 2. Schema Pipeline

Claude Code has a clear pipeline:

`inputSchema -> input_schema -> safeParse -> validateInput -> permission -> call`

`ms-cli` currently has:

`Schema() -> provider export -> tool-specific ParseParams -> tool-specific semantic validation -> execute`

Main consequences:

- Claude Code has a shared validation frontier before tool logic runs
- `ms-cli` repeats parsing and semantic checks inside each tool
- `ms-cli` has a higher risk of schema/runtime drift

### 3. Tool Exposure Strategy

Claude Code separates:

- tool definition
- tool pool assembly
- model exposure of tools

It can also delay exposure of tools and use a `ToolSearch` step.

`ms-cli` does not currently distinguish these layers. The engine just exports the registry as the full visible tool list every turn.

### 4. Execution Runtime

Claude Code has:

- single-tool execution pipeline
- multi-tool orchestration
- streaming tool executor
- hook points
- standardized result mapping

`ms-cli` has:

- a single engine-owned execution path
- serial tool execution
- optional per-tool streaming
- no hook system
- direct string result injection into context

### 5. Permission Architecture

Claude Code integrates permissions as part of a broader policy system:

- tool-level metadata
- explicit permission rules
- mode/policy layering
- hook/classifier checks
- UI approval flow
- sandbox-aware decisions

`ms-cli` has a solid rule-based permission service, but it is not yet a first-class part of the `Tool` contract.

### 6. Extension Model

Claude Code adapts MCP tools into the same unified tool runtime as built-in tools.

`ms-cli` shows early conceptual awareness of external tools because permission rules already recognize `mcp__...`, but the runtime itself does not yet have an MCP-style external tool adaptation layer.

### 7. Prompt and Tool Catalog Relationship

Claude Code is closer to deriving model-visible tool data from tool objects and runtime policy.

`ms-cli` still hardcodes tool descriptions and usage guidance in the system prompt, which makes the prompt partially act as a second tool catalog.

## Phase 3: Improvements ms-cli Can Borrow from Claude Code

### 1. Evolve the Tool Contract

Recommended direction:

- keep the current lightweight developer experience
- add optional capabilities instead of one giant interface

A practical next step would be to extend the contract with optional interfaces such as:

- `ValidateInput`
- `PermissionContext`
- `ConcurrencyClass`
- `ResultRenderer`

This preserves simplicity while giving the runtime more structure.

### 2. Introduce a Shared Validation Stage

The highest-value improvement is to make validation a first-class runtime phase:

`decode -> schema validate -> semantic validate -> permission -> execute`

This would reduce duplicated parsing logic inside tools and make tool behavior more predictable.

### 3. Replace the Minimal Schema Type

`llm.ToolSchema` should eventually move toward a richer JSON Schema representation.

Recommended options:

- expand the existing struct to support nested schemas and constraints
- or adopt a generic JSON-schema-shaped map model plus helper builders

Without this, more advanced tools will be difficult to describe accurately across providers.

### 4. Remove Prompt Duplication

The hardcoded tool list in `DefaultSystemPrompt()` should be reduced or generated from registry metadata.

Best direction:

- system prompt provides stable behavioral guidance
- tool registry provides actual tool catalog
- per-tool descriptions stay with tool definitions

This avoids drift and makes future tool filtering possible.

### 5. Add a Tool Exposure Layer

Before copying Claude Code's full deferred-loading design, `ms-cli` should first add a smaller abstraction:

- available tools
- exposed tools for this request

Once that exists, later enhancements become possible:

- request-scoped filtering
- provider-specific filtering
- defer heavy tools
- future tool-search or skill-search flows

### 6. Separate Execution from Orchestration

The current engine execution logic is already understandable, so the refactor can be incremental:

- keep single-tool execution as one unit
- extract multi-tool scheduling into a separate orchestration unit

This would allow future rules such as:

- read-only tools may run concurrently
- write tools stay serial
- shell remains serial by default

### 7. Formalize External Tool Adaptation

If `ms-cli` wants to converge toward Claude Code-class extensibility, it should define an adapter path for external tools.

Even before full MCP support, the architecture can prepare for:

- externally supplied schema
- externally supplied execution backend
- shared runtime wrapping for permission, validation, and result mapping

### 8. Make Schema and Runtime Contract Match Exactly

This is a near-term cleanup item with low implementation risk and high conceptual value.

Example:

- if `write` accepts `file_path` or `filename`, either expose them formally
- or remove them and keep `path` as the only contract

Claude Code is strict about keeping model-visible schema and runtime validation on a clearer path. `ms-cli` should move in the same direction.

## Interim Conclusion

`ms-cli` already has a usable tool system, but it is still in an early-runtime form:

- one thin tool interface
- one simple registry
- one engine-owned execution path
- one strong centralized permission service

This is a good base for a compact CLI agent.

The main opportunity is not to copy Claude Code mechanically, but to selectively borrow the layers that reduce drift and increase extensibility:

1. richer schema and validation pipeline
2. clearer separation between tool definition, tool exposure, and execution
3. optional tool metadata for permissions and concurrency
4. less prompt duplication
5. an eventual external-tool adaptation layer

## Next Analysis Pass

The next pass should focus on:

1. A file-by-file mapping of the `ms-cli` tool runtime
2. A proposed target architecture for `ms-cli`
3. A staged refactor sequence with low-risk implementation order

## Phase 4: Proposed Target Architecture for ms-cli

The best path for `ms-cli` is not to clone Claude Code's full runtime. The better move is to introduce the missing layers one by one while preserving the current compact mental model.

### 1. Recommended Layering

Suggested target architecture:

```text
Tool Definition Layer
  -> tool metadata, schema, validation hooks, execute

Tool Registry Layer
  -> all installed/available tools

Tool Exposure Layer
  -> request-scoped visible tools

Permission & Policy Layer
  -> tool/action/path checks, interactive approval

Execution Runtime Layer
  -> parse, validate, permission, execute, result mapping

Orchestration Layer
  -> serial vs concurrent scheduling for tool batches

Transcript/UI Layer
  -> how tool execution is summarized back to the model and UI
```

This keeps the current engine-centered flow, but removes the need for the engine to personally own every tool-related concern.

### 2. Minimal Contract Evolution

Instead of replacing `tools.Tool` with a huge interface, add optional interfaces around it.

Base interface can remain:

```go
type Tool interface {
    Name() string
    Description() string
    Schema() llm.ToolSchema
    Execute(ctx context.Context, params json.RawMessage) (*Result, error)
}
```

Optional extensions can then be layered on top:

```go
type ValidatingTool interface {
    ValidateInput(ctx context.Context, params json.RawMessage) error
}

type PermissionAwareTool interface {
    PermissionContext(params json.RawMessage) (action string, path string, err error)
}

type ConcurrencyAwareTool interface {
    ConcurrencyMode() string
}

type ResultFormattingTool interface {
    FormatToolMessage(result *Result) string
}
```

This is a practical middle ground:

- contributor ergonomics stay good
- the runtime gains structured extension points
- not every tool needs to implement every concern

### 3. Recommended Execution Pipeline

The current engine logic should eventually become:

```text
tool lookup
  -> decode params
  -> schema validation
  -> semantic validation
  -> derive permission context
  -> permission request/check
  -> execute
  -> normalize result
  -> append transcript message
  -> emit UI event
```

Compared with today, the most important structural gain is that decode/validate/permission become explicit pipeline stages rather than ad hoc work split across engine and tool bodies.

### 4. Recommended Exposure Model

Before implementing deferred loading, `ms-cli` should first create a simple split between:

- installed tools
- exposed tools for this turn

That unlocks several future directions without overbuilding:

- hide tools unsupported by a provider
- hide tools disabled by config or mode
- later introduce skill-driven or request-driven tool subsets
- later introduce deferred or searchable tools if the catalog grows

### 5. Recommended Result Handling

`Result.Content` should remain for compatibility, but the runtime should start owning transcript formatting instead of treating every result as raw free text.

A good intermediate model would be:

```go
type Result struct {
    Content string
    Summary string
    Meta    map[string]any
    Error   error
    Kind    string
}
```

Then the execution runtime can decide:

- what goes to context
- what goes to UI
- what should be truncated or summarized
- what should be marked as error payload

That moves `ms-cli` closer to Claude Code's clearer result pipeline without requiring a large rewrite.

## Phase 5: Low-Risk Refactor Sequence

The safest sequence is to refactor in six small steps.

### Step 1. Remove Schema Drift

Clean up the places where runtime input handling is broader than the published schema.

Start with:

- `tools/fs/write.go`
- engine-side path extraction assumptions

Goal:

- model-visible schema and actual accepted arguments match exactly

### Step 2. Add an Explicit Validation Hook

Introduce `ValidatingTool` and teach the engine runtime to call it before execution.

Goal:

- move semantic validation out of the middle of `Execute`
- make tool errors more uniform

### Step 3. Add Permission Context Extraction to Tools

Move action/path derivation away from generic engine heuristics where possible.

Goal:

- tools define their own permission-relevant action and target path
- engine stops carrying tool-name-specific extraction logic

### Step 4. Extract a Tool Runtime Module

Move the execution logic out of `agent/loop/engine.go` into a dedicated tool runtime package.

Good boundary:

- `RunToolCall`
- `AppendToolResult`
- `EmitToolEvents`

Goal:

- reduce engine complexity
- make future orchestration possible

### Step 5. Add Tool Exposure as a Separate Decision

Introduce something like:

```go
type VisibleToolSet interface {
    ToLLMTools() []llm.Tool
}
```

Goal:

- separate registry from model-visible tool list
- prepare for provider-specific or policy-specific tool filtering

### Step 6. Add Basic Orchestration Metadata

Once the runtime is extracted, add a minimal concurrency model:

- read-like tools: concurrency-safe
- write/edit/shell: serial

Goal:

- preserve correctness
- gain performance where tool calls are independent

## File-by-File Mapping

This is the most useful current map of where tool-system responsibilities live today.

### Definition and registration

- `tools/types.go`: base tool protocol and result model
- `tools/registry.go`: ordered registry and LLM export
- `internal/app/wire.go`: concrete tool assembly and registration

### Schema and provider adaptation

- `integrations/llm/provider.go`: internal tool schema types
- `integrations/llm/codec_anthropic.go`: Anthropic tool encoding and `tool_result` decoding
- `integrations/llm/codec_openai_completion.go`: OpenAI chat-completions tool encoding
- `integrations/llm/codec_openai_responses.go`: OpenAI responses tool encoding, currently with unconditional `strict: true`

### Runtime execution

- `agent/loop/engine.go`: request construction, tool execution, context updates, event emission

### Permissions

- `permission/service.go`: core policy evaluation and interactive approval
- `permission/rules.go`: Claude-style rule parsing and matching
- `permission/dangerous.go`: dangerous shell command detection
- `internal/app/permission_ui.go`: TUI approval bridge

### Concrete tools

- `tools/fs/*.go`: read/write/edit/grep/glob tools
- `tools/shell/shell.go`: LLM-callable shell wrapper
- `runtime/shell/runner.go`: actual command execution runtime
- `tools/skills/load_skill.go`: deferred skill-content loading

## Updated Conclusion

After a second pass, the most important judgment is this:

`ms-cli` already has the right coarse-grained pieces, but they are still collapsed into too few layers.

The project does not need a wholesale redesign. It needs targeted separation of concerns:

1. richer validation and schema handling
2. tool-owned permission context instead of engine heuristics
3. a dedicated tool runtime module
4. a distinction between registry and model-visible tool set
5. gradual preparation for external-tool adaptation

That is the shortest path toward Claude Code-level robustness without losing the simplicity that currently makes `ms-cli` fast to evolve.
