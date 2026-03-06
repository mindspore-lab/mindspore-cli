## Detailed Code Review: PR #5 - Tool System Update

**PR:** #5 (`townwish4git/ms-cli:tool0306` -> `main`)
**Author:** townwish4git
**Scope:** 60 files, +6,818 / -3,070 lines

---

### Overall Assessment

This is a substantial architectural refactoring that modernizes the tool system from simple stubs into a production-ready, extensible framework. The direction is sound — interface-based registry, structured results, permission engine, event bus — but there are several issues across security, concurrency, and test coverage that should be addressed before merging.

---

### Architecture (Positive)

1. **Interface-based Registry** (`registry.Registry` replacing `*tools.Registry`) — Enables proper mocking and decouples consumers from implementation. The `DefaultRegistry` with RWMutex and `BuiltInRegistry` extension is well-structured.

2. **Structured ToolResult with Parts** (`tools/result.go`) — Multi-part result model (`Text`, `JSON`, `Binary`, `Artifact`, `Error`) with helper constructors (`NewTextResult`, `NewJSONResult`, `NewErrorResult`) is clean and future-proof.

3. **ToolDefinition with Rich Metadata** (`tools/definition.go`) — `ToolMeta` with `CostLevel`, `Permission`, `ReadOnly`, `Idempotent`, `Timeout` provides excellent introspection for both the permission system and LLM.

4. **Multi-protocol Resolver** (`tools/resolver/resolver.go`) — Supporting MCP, OpenAI, and Internal schemas in `convertToProviderSchema()` is forward-thinking and well-abstracted.

5. **Permission Engine with Caching** (`tools/permission/engine.go`) — Rule-based evaluation with wildcard matching and cache invalidation on ruleset update is a major improvement over the old `PermissionService`.

---

### Critical Issues

#### 1. Plan Executor Bypasses Permissions (Security)

**File:** `agent/plan/executor.go`

```go
type simpleToolExecutor struct{}
func (e *simpleToolExecutor) AskPermission(req tools.PermissionRequest) error { return nil }
```

The `executeTool()` method creates a `simpleToolExecutor` that auto-approves everything. Plans executing destructive tools (bash, write, edit) will **never** get permission-gated, even when the user has set them to "ask" or "deny".

The `SetPermissionEngine(pe permission.Engine)` method exists on `PlanExecutor` and stores `permEngine`, but `executeTool()` never uses it — it always creates a fresh `simpleToolExecutor`.

**Fix:** Create a proper executor that delegates to `e.permEngine`:
```go
func (e *PlanExecutor) executeTool(ctx context.Context, toolName string, params map[string]any) (string, error) {
    // ...
    toolExec := &planToolExecutor{permEngine: e.permEngine}
    // ...
}
```

#### 2. `generateID()` / `generateRequestID()` — Weak ID Generation

`invocation.go` uses `rand.Int63()` which is reasonable, but `permission/engine.go` uses:
```go
func generateRequestID() string {
    return fmt.Sprintf("perm_%d", time.Now().UnixNano())
}
```

This has **no random component** — two concurrent permission requests at the same nanosecond will collide, causing one response to be lost.

**Fix:** Add randomness or use `atomic.AddInt64` counter:
```go
func generateRequestID() string {
    return fmt.Sprintf("perm_%d_%x", time.Now().UnixNano(), rand.Int63())
}
```

#### 3. Permission Engine — Potential Deadlock

**File:** `tools/permission/engine.go`

The `DefaultEngine.Ask()` method:
1. Acquires write lock (`e.mu.Lock()`)
2. Stores pending request with response channel
3. Publishes event via event bus
4. Waits on the response channel

If the event handler that responds to `permission:requested` needs to call any engine method (e.g., `CanExecute`, `GetRuleset`), it will deadlock because the write lock is still held.

**Fix:** Release the lock before waiting on the channel:
```go
e.mu.Lock()
e.pending[req.ID] = ch
e.mu.Unlock()
// publish event...
// wait on ch (no lock held)
```

#### 4. AsyncEventBus — Silent Event Drop on Permission Events

When the async event bus buffer is full, `Publish()` returns an error but the caller in `Ask()` may not handle it. If a `permission:requested` event is dropped, the tool execution will hang forever waiting for a response that was never delivered.

**Fix:** For critical topics (`permission:*`), either block with timeout or return an error immediately to the caller. At minimum, log a warning.

---

### Major Issues

#### 5. Missing Tests (Biggest Gap)

60 files changed, but only import-path updates in existing tests (`engine_context_test.go`, `engine_trace_test.go`). **Zero unit tests** for:

- `ToolResult` / `NewTextResult` / `NewErrorResult` (result.go)
- `ToolContext.ContextOrBackground()` / `Done()` (context.go)
- `DefaultEventBus` / `AsyncEventBus` — subscribe, publish, unsubscribe, close (events/bus.go)
- `DefaultEngine` — rule evaluation, wildcard matching, caching, ask flow (permission/engine.go)
- `DefaultRegistry` / `BuiltInRegistry` — register, get, list, ToLLMTools (registry/registry.go)
- `DefaultResolver` — resolve, permission pre-filter, schema conversion (resolver/resolver.go)
- `PlanExecutor.executeTool()` — tool lookup, execution, error handling (plan/executor.go)
- `BashTool` — dangerous command detection, timeout, permission check (builtin/bash.go)
- `EditTool` — case-insensitive replace, multiline (builtin/edit.go)
- `ReadTool` — offset/limit handling (builtin/read.go)
- `WriteTool` — append mode, directory creation (builtin/write.go)

The permission system and event bus are **critical infrastructure** — they must have unit tests before merging.

#### 6. `DefaultResolver` — Not Thread-Safe

**File:** `tools/resolver/resolver.go`

```go
type DefaultResolver struct {
    tools map[string]tools.ExecutableTool  // No mutex!
    permEngine permission.Engine
}
```

The `tools` map is accessed by `Register()`, `Get()`, `Resolve()`, etc. without any synchronization. If multiple goroutines register/resolve concurrently, this will panic with a concurrent map access.

**Fix:** Add `sync.RWMutex` like `DefaultRegistry` does.

#### 7. `isDangerousCommand()` — Easily Bypassed

**File:** `tools/registry/builtin/bash.go`

```go
func isDangerousCommand(command string) bool {
    lower := strings.ToLower(command)
    dangerousPatterns := []string{
        "rm -rf /",
        "> /dev/sda",
        // ...
    }
    for _, pattern := range dangerousPatterns {
        if strings.Contains(lower, pattern) {
            return true
        }
    }
    return false
}
```

This blacklist is trivially bypassed:
- `rm -rf /` is caught, but `rm -rf /*` or `rm -r -f /` is not.
- `cat /dev/urandom > /dev/sda` bypasses `> /dev/sda` because of the space before `>`.
- Any obfuscation (variable expansion, base64 decode, eval) bypasses all patterns.

**Suggestion:** This provides a false sense of security. Either invest in proper sandboxing (seccomp, namespaces) or remove this check and rely entirely on the permission system (which is the correct architectural approach given this PR's design). The permission engine's `RequireConfirm: true` on bash is the right layer for this.

#### 8. `extractResultContent()` — Ignores Non-Text Parts

**File:** `agent/loop/engine.go`

Only `PartTypeText` content is extracted. Tool results with `PartTypeJSON` data (like `EditTool` which adds both JSON and Text parts) will lose the structured data in traces/events.

**Fix:**
```go
func extractResultContent(result *tools.ToolResult) string {
    var parts []string
    for _, p := range result.Parts {
        switch p.Type {
        case tools.PartTypeText:
            parts = append(parts, p.Content)
        case tools.PartTypeJSON:
            if data, err := json.Marshal(p.Data); err == nil {
                parts = append(parts, string(data))
            }
        }
    }
    return strings.Join(parts, "\n")
}
```

---

### Minor Issues

#### 9. `ToolContext.AbortSignal` — Non-Idiomatic Go Naming

```go
AbortSignal context.Context `json:"-"`
```

Go convention is to name context fields `ctx` or embed `context.Context`. `AbortSignal` comes from JavaScript/TypeScript conventions. Consider `Ctx` for Go idiom consistency.

#### 10. `buildPermissionConfig()` — Hardcoded Dangerous Tool List

**File:** `app/bootstrap.go`

```go
dangerousTools := []string{"write", "edit", "shell", "bash"}
```

This duplicates knowledge already encoded in `ToolMeta.Cost` (bash is `CostLevelHigh`) and `Permission.RequireConfirm: true`. If a new dangerous tool is added to `builtin/`, this list won't update automatically.

**Suggestion:** Iterate registered tools and use `tool.Info().Meta.Cost >= CostLevelHigh || hasRequireConfirm(tool)` instead.

#### 11. Chinese Comments on Exported Types

All comments are in Chinese (`// 工具成本等级`, `// 权限标识`, `// 创建新的Bash工具`). For `godoc` compatibility and wider readability, exported type/method comments should be in English. Internal comments can stay in Chinese per team convention.

#### 12. `EditTool.replaceAllCaseInsensitive()` — O(n*m) Performance

**File:** `tools/registry/builtin/edit.go`

```go
func replaceAllCaseInsensitive(content, oldText, newText string) (string, int) {
    for {
        idx := strings.Index(lowerContent, lowerOldText)
        // ...
        lowerContent = strings.ToLower(content)  // Re-lowercases entire content each iteration!
    }
}
```

Each iteration calls `strings.ToLower(content)` on the full content. For a file with many matches, this is O(n*m) where n = content length and m = match count. Use `strings.NewReplacer` or track offset instead.

#### 13. PR Size — Consider Splitting

60 files, ~10,000 lines changed in a single PR is very hard to review thoroughly. Consider splitting into:
1. **PR A:** New interfaces and types (`tools/definition.go`, `result.go`, `context.go`, `invocation.go`)
2. **PR B:** Infrastructure (`events/bus.go`, `permission/engine.go`, `registry/registry.go`, `resolver/resolver.go`) + tests
3. **PR C:** Built-in tools (`builtin/bash.go`, `read.go`, `write.go`, `edit.go`) + tests
4. **PR D:** Integration (`engine.go`, `bootstrap.go`, `commands.go`, remove old code)

#### 14. `cmdYolo()` — Uses Emoji in Code Output

**File:** `app/commands.go`

```go
Message: "⚡ YOLO mode enabled! All operations will be auto-approved. Use with caution!",
```

And `ReportToMarkdown()` in `plan/executor.go`:
```go
status := "⏳"  // ✅ ❌ ⏭️
```

Emoji in programmatic output can cause width calculation issues in terminals. Use text labels as default.

#### 15. `WriteTool` — Mode Parameter Takes Decimal

```go
"default": 420, // 0644 in decimal
```

The JSON schema says `"type": "integer"` and the default is `420` (0644 decimal). Users sending `644` intending octal permissions will get `os.FileMode(644)` = `01204` in octal, which is wrong. Either:
- Accept string input (`"0644"`) and parse with `strconv.ParseInt(s, 8, 32)`, or
- Document clearly that the value must be decimal.

---

### Summary

| Area | Rating | Notes |
|------|--------|-------|
| Architecture | Good | Clean interface-based design, multi-protocol support |
| Code Quality | Needs Work | ID collision, thread safety, performance |
| Security | Concern | Plan executor bypasses permissions, weak command blacklist |
| Concurrency | Concern | Resolver not thread-safe, permission engine deadlock risk |
| Test Coverage | Insufficient | Zero tests for new critical infrastructure |
| PR Size | Too Large | 60 files, recommend splitting into 3-4 PRs |

**Recommendation:** Request changes. Priority fixes before merge:
1. Wire permission engine into plan executor (security)
2. Add `sync.RWMutex` to `DefaultResolver` (crash prevention)
3. Fix permission engine lock scope (deadlock prevention)
4. Add unit tests for permission engine, event bus, and registry (correctness)
5. Fix `generateRequestID()` collision risk — no random component (correctness)
