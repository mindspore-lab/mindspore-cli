# Remove Viewport Mode, Migrate Features to Inline

**Date:** 2026-04-01

## Problem

The TUI has two rendering paths:

1. **Viewport mode** — a bubbletea viewport component that re-renders all
   conversation messages on every frame via `View()`. Currently **disabled** —
   `updateViewport()` returns immediately on line 2850 of `ui/app.go`.
2. **Inline mode** — prints messages above the live area via `tea.Println()`.
   This is the **only active path** users see.

Features like tool output truncation (`collapsedToolDetails`,
`truncateToolContentForTool`), the Ctrl+O expand/collapse toggle
(`toolsExpanded`), and scroll handling only exist in the dead viewport path.
Users get no truncation and Ctrl+O does nothing.

## Goals

1. Remove all dead viewport code.
2. Migrate tool output truncation to the inline print path so output is
   collapsed by default.
3. Make Ctrl+O a toggle that switches between collapsed and expanded mode and
   re-prints the last tool result in the new mode.
4. Remove viewport scroll/mouse handlers — terminal native scrollback is
   sufficient.

## Non-goals

- Per-message expand/collapse (too complex for `tea.Println` — content is
  immutable once printed).
- A new viewport or scrollable panel component.

## Design

### 1. Code Removal

Delete from `ui/app.go`:

| Item | Reason |
|------|--------|
| `viewport` field on App struct | Dead |
| `updateViewport()` function | Dead (early return) |
| `viewportRenderState()` function | Only caller of renderToolMessageContent |
| `renderToolMessageContent()` function | Only called from dead viewport path |
| All `updateViewport()` call sites | No-op calls |
| Mouse viewport handler (`tea.MouseMsg` → `a.viewport.Update`) | Viewport never displayed |
| Page Up/Down viewport scroll (`handleKey` lines ~911-914) | Terminal native scroll suffices |
| `viewport` import from `components` | Unused after removal |

**Keep:**

| Item | Reason |
|------|--------|
| `toolsExpanded` bool | Reused for inline Ctrl+O toggle |
| `truncateToolContentForTool()` | Used by training replay + inline truncation |
| `collapsedToolDetails()` | Used by inline truncation |
| `collapsedPreviewLines()` | Used by inline truncation |
| `toolPreviewPolicy()` and constants | Used by truncateToolContentForTool |
| `RenderMessages()` in panels/chat.go | Used by `renderTranscriptMessage()` |

### 2. Inline Truncation

Apply truncation in the inline print path **before** calling `printMessage()`.

**Where to apply:** In `eventPrintCmd()` (inline.go), for tool result events
that call `printResolvedTool()` or `printShellFinished()`.

**Logic:**

```
if toolsExpanded:
    print full content (no truncation)
else if msg.Display == DisplayCollapsed:
    apply collapsedToolDetails(content, collapsedPreviewLines(toolName))
else:
    apply truncateToolContentForTool(toolName, content)
```

This reuses the existing truncation functions unchanged.

**Truncation policy (existing, unchanged):**

| Display mode | Tools | Behavior |
|---|---|---|
| `DisplayCollapsed` | Read, Grep, Glob, Skill | Max 3 detail lines |
| `DisplayExpanded` | Shell, Edit, Write | Head 5 lines, max 12000 runes |

When truncated, the last line reads:
`… +N lines (ctrl+o to expand)`

### 3. Ctrl+O Toggle

Keep the existing `toolsExpanded` bool on App.

**On Ctrl+O press:**

1. Flip `toolsExpanded = !toolsExpanded`.
2. Find the last `MsgTool` message in `state.Messages` that is not Pending.
3. Re-print it via `tea.Println` with the new truncation mode applied:
   - If now expanded: print full content.
   - If now collapsed: print truncated content.
4. Return the print command so it appears in scrollback.

This means:
- First press: expands the last tool result (re-prints full output).
- Second press: collapses again (re-prints truncated).
- All subsequent tool results respect the current toggle state.

### 4. Shell Streaming Output

Shell output streams line-by-line via `CmdOutput` events. Truncation cannot
be applied mid-stream because the final line count is unknown.

**Approach:** Shell streaming output prints untruncated during streaming. When
`CmdFinished` arrives, if the tool is not expanded and the output exceeds the
policy limit, `printShellFinished()` does nothing extra — the streamed output
is already in scrollback.

This matches the current behavior: shell output streams fully. The truncation
policy only affects the **final resolved message** when printed via
`printResolvedTool()` (non-streaming tools like Grep, Read, Edit, Write).

For shell specifically: `printShellFinished()` already returns `nil` when
output was streamed (summary is "completed"). No change needed.

### 5. Files Changed

| File | Changes |
|------|---------|
| `ui/app.go` | Remove viewport field, `updateViewport()`, `viewportRenderState()`, `renderToolMessageContent()`, mouse/scroll handlers, all `updateViewport()` calls. Keep `toolsExpanded`, truncation functions, Ctrl+O handler. |
| `ui/inline.go` | Add `truncateForPrint()` helper. Apply in `printResolvedTool()`, `printShellFinished()`. Add Ctrl+O re-print logic (or call from app.go key handler). |
| `ui/app.go` (handleKey) | Update Ctrl+O to re-print last tool message. |

### 6. Verification

1. `go build ./...` compiles.
2. `go test ./ui/...` passes — update/remove tests that depend on viewport.
3. Manual: run mscli, ask agent to grep something → output is truncated with
   `… +N lines` hint.
4. Manual: press Ctrl+O → last tool result re-prints expanded.
5. Manual: press Ctrl+O again → re-prints collapsed.
6. Manual: shell command → output streams fully (no truncation mid-stream).
