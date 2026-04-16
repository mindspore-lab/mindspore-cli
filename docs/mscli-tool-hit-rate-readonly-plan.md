# Tool Hit Rate And Read-Only Exploration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve first-try tool selection and parameter correctness for filesystem tasks, add a first-class read-only directory exploration tool, and stop shell-based file-writing fallbacks from bypassing `write`/`edit` permissions.

**Architecture:** Keep the current lightweight `Tool` interface and centralized permission service, but make three focused improvements: move tool-choice guidance into tool descriptions, add a dedicated `list_dir` read-only primitive, and classify shell commands by intent so read-only exploration can be allowed without opening a write bypass. Avoid broad runtime rewrites in this cycle; only add seams that directly improve these two target workflows and the related shell-write guardrail.

**Tech Stack:** Go, `agent/loop`, `tools/fs`, `tools/shell`, `permission`, Bubble Tea UI tests, current `llm.Provider` capture tests.

---

## Scope

This plan is intentionally narrower than the broader tool-system refactor notes in `docs/mscli-tool-system-analysis.md` and `docs/mscli-tool-system-refactor-plan.md`.

In scope for this cycle:

- tool prompting and parameter-contract cleanup for existing filesystem tools
- add `list_dir` as the default directory-structure exploration primitive
- fix `glob("**/train.py")` so it also matches `./train.py`
- reduce or remove permission prompts for read-only shell exploration commands
- block shell-based file creation/editing fallbacks after `write`/`edit` denial
- add deterministic and agent-level E2E coverage for the three known bad cases

Explicitly out of scope for this cycle:

- `multi_edit`
- `web_fetch`
- MCP/external tool adaptation
- full schema-system rewrite
- large runtime extraction or orchestration refactor

## Success Criteria

- The request “当前目录是否有 `train.py` 文件” succeeds in one tool call and returns `./train.py` when present.
- The request “为我总结当前目录的代码结构” prefers `list_dir` over `shell(ls/find/tree)` in agent-level tests.
- Read-only shell exploration commands no longer interrupt the user with avoidable permission prompts in the directory-summary flow.
- After the user denies `write` or `edit`, the agent does not continue by attempting equivalent shell writes such as heredoc redirection, `tee`, or `echo >`.
- Tool descriptions and system prompt no longer conflict about tool boundaries.

## Design Decisions

### 1. `list_dir` is the new primary tool for directory structure questions

- Use `list_dir` for “what is in this directory”, “summarize the structure”, and first-pass tree inspection.
- Keep `glob` for file-name/path-pattern lookup, not general directory summarization.
- Keep `grep` for content search only.

### 2. `glob` semantics must treat `**/name` as “zero or more directories”

- `**/train.py` must match both `train.py` and `subdir/train.py`.
- This is a correctness bug, not just a prompting issue.

### 3. Shell permissions must distinguish read-only exploration from file authoring

- Read-only exploration: `ls`, `find`, `tree`, `pwd`, `stat`, `rg --files` and similar commands that do not modify files.
- File authoring via shell: output redirection, heredoc-to-file, `tee`, `sed -i`, `perl -pi`, and similar commands that create or modify files.
- Read-only shell exploration can be separately allowlisted without letting shell impersonate `write` or `edit`.

### 4. `write` and `edit` remain the exclusive file-authoring tools

- If file creation or modification is required, the agent must use `write` or `edit`.
- If those tools are denied, the agent must stop and report the restriction instead of routing through shell.

## File Map

### Existing files to modify

- `agent/loop/engine.go`
- `agent/loop/engine_prompt_test.go`
- `internal/app/wire.go`
- `tools/fs/glob.go`
- `tools/fs/grep.go`
- `tools/fs/read.go`
- `tools/fs/edit.go`
- `tools/fs/write.go`
- `tools/shell/shell.go`
- `permission/service.go`
- `permission/rules.go`
- `permission/service_test.go`
- `permission/rules_test.go`

### New files likely to add

- `tools/fs/list_dir.go`
- `tools/fs/list_dir_test.go`
- `agent/loop/engine_tool_selection_test.go`
- `internal/app/tool_system_e2e_test.go`

## Task 1: Tighten Tool Guidance And Contract Signals

**Files:**
- Modify: `agent/loop/engine.go`
- Modify: `agent/loop/engine_prompt_test.go`
- Modify: `tools/fs/read.go`
- Modify: `tools/fs/glob.go`
- Modify: `tools/fs/grep.go`
- Modify: `tools/fs/edit.go`
- Modify: `tools/fs/write.go`
- Modify: `tools/shell/shell.go`

- [ ] **Step 1: Rewrite tool descriptions with explicit decision boundaries**

Update each tool description so the model can answer:

- what the tool does
- when to use it
- when not to use it
- which sibling tool is preferred in nearby cases
- which arguments must be exact

Required message changes:

- `glob`: use for file-path/name pattern lookup, not for summarizing a directory tree
- `grep`: use for content search, not file discovery
- `read`: use after the exact file path is known
- `shell`: reserve for tests, build, git, process execution, or cases with no dedicated tool
- `write` / `edit`: file-authoring only; do not simulate them with shell redirection or heredoc

- [ ] **Step 2: Shrink the global system prompt to high-level policy**

Revise `DefaultSystemPrompt()` so it keeps only:

- gather information before editing
- read before editing
- prefer dedicated tools over shell
- `write` requires exact `path` and `content`
- shell must not be used to create or edit files when dedicated tools exist

Move detailed tool-by-tool routing guidance into tool descriptions.

- [ ] **Step 3: Clarify temporary compatibility for `write` arguments**

Keep runtime compatibility for legacy aliases during this cycle if needed, but make the model-visible contract strict:

- schema advertises only `path` and `content`
- error text says aliases are compatibility-only, not preferred
- new tests assert prompt and descriptions consistently reinforce the canonical fields

- [ ] **Step 4: Add prompt-contract regression tests**

Add or extend tests in `agent/loop/engine_prompt_test.go` to assert the prompt includes:

- prefer dedicated tools over shell
- do not use shell heredoc/redirection to create files
- `write` requires `path` and `content`

## Task 2: Add `list_dir` And Fix `glob` Root-Match Semantics

**Files:**
- Create: `tools/fs/list_dir.go`
- Create: `tools/fs/list_dir_test.go`
- Modify: `internal/app/wire.go`
- Modify: `agent/loop/engine.go`
- Modify: `tools/fs/glob.go`
- Modify: `tools/fs/ignore_test.go` or add a dedicated `glob` regression test file

- [ ] **Step 1: Implement `list_dir`**

Suggested tool contract:

```go
{
  "path": string,
  "depth": integer,
  "include_hidden": boolean,
  "offset": integer,
  "limit": integer
}
```

Behavior requirements:

- workspace-relative `path`
- deterministic ordering
- clear distinction between files and directories
- bounded output with paging
- concise summary including total visible entries

- [ ] **Step 2: Register and surface `list_dir` cleanly**

Wire the tool in `internal/app/wire.go`, and update tool call display logic in `agent/loop/engine.go` so transcripts show meaningful summaries for `list_dir`.

If UI needs a dedicated event type, add it only if necessary; otherwise reuse the current generic tool event path to keep the diff small.

- [ ] **Step 3: Fix `glob("**/train.py")` to match root files**

Update `tools/fs/glob.go` so `**/foo` means zero-or-more directories, not one-or-more. Add a regression test that proves:

- `train.py` is returned for `**/train.py`
- nested matches are also returned

- [ ] **Step 4: Add deterministic tool tests**

Add tests covering:

- `list_dir` basic listing
- `list_dir` depth limiting
- `list_dir` hidden-file filtering
- `glob` root-match regression for `**/train.py`

## Task 3: Classify Shell Commands By Intent And Block Shell Write Fallbacks

**Files:**
- Modify: `permission/service.go`
- Modify: `permission/rules.go`
- Modify: `tools/shell/shell.go`
- Modify: `permission/service_test.go`
- Modify: `permission/rules_test.go`

- [ ] **Step 1: Introduce shell intent classification helpers**

Add focused helpers that classify shell commands into at least:

- read-only exploration
- shell file authoring
- general execution

The first version only needs to cover commands that affect the known bad cases.

- [ ] **Step 2: Allow read-only exploration without broadening shell writes**

Update permission evaluation so read-only exploration commands can be auto-allowed or matched by read-only-style rules without turning on generic `shell(*)`.

Examples to cover:

- `ls`
- `find`
- `tree`
- `pwd`
- `stat`
- `rg --files`

- [ ] **Step 3: Treat shell authoring as equivalent to file write/edit**

Commands with these patterns must not bypass denied file-authoring permissions:

- `>`, `>>`
- `tee`
- `cat <<EOF > file`
- `echo ... > file`
- `sed -i`
- `perl -pi`

If `write`/`edit` is denied, the result should be deny-or-stop, not fallback shell execution.

- [ ] **Step 4: Add permission regression tests**

Add tests proving:

- read-only shell exploration is recognized and does not require the same approval path as a write
- shell commands with file-writing semantics are not treated as read-only
- heredoc and redirection patterns are classified as authoring

## Task 4: Lock The Behavior With Agent-Level E2E Tests

**Files:**
- Create: `agent/loop/engine_tool_selection_test.go`
- Create: `internal/app/tool_system_e2e_test.go`
- Modify: provider capture test helpers under `internal/app/wire_provider_test.go` only if needed

- [ ] **Step 1: Add E2E for bad case 1**

Fixture:

- workspace root contains `train.py`
- nested directory may also contain another `train.py`

Expected behavior:

- the agent can answer whether `train.py` exists from the current directory
- the first tool used is `glob` or `list_dir`
- `glob("**/train.py")` returns the root file on the first try

- [ ] **Step 2: Add E2E for bad case 2**

Prompt:

- “为我总结当前目录的代码结构”

Expected behavior:

- the first tool used is `list_dir`
- the flow does not rely on `shell(ls/find/tree)` as the primary path
- if shell is used secondarily, read-only exploration does not interrupt the user with avoidable approval prompts

- [ ] **Step 3: Add E2E for bad case 3**

Prompt:

- “帮我写一个 markdown 文档”

Harness behavior:

- deny `write` permission

Expected behavior:

- the agent does not attempt shell-based writes such as heredoc redirection
- the flow stops with a permission-aware explanation or explicit request for file-authoring permission

- [ ] **Step 4: Run narrow verification first**

Run the smallest relevant Go commands before any broader sweep:

```bash
go test ./tools/fs ./permission ./agent/loop ./internal/app
```

If those pass and the touched test set is narrow enough, stop there. Only widen if a broader package interaction needs proof.

## Implementation Order

1. Task 1: tighten tool guidance first
2. Task 2: add `list_dir` and fix `glob`
3. Task 3: add shell intent classification and write-bypass guardrails
4. Task 4: add the three E2E tests and run narrow verification

## Review Notes

This plan intentionally defers bigger architectural moves. The only architectural change justified in this cycle is introducing a small shell-intent seam inside the existing permission pipeline. That gives immediate benefit to the two target workflows and the write-bypass bad case without forcing a full runtime redesign.
