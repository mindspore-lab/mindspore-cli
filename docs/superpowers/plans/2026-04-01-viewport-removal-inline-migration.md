# Viewport Removal & Inline Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove dead viewport rendering code and migrate tool output truncation + Ctrl+O expand/collapse to the inline `tea.Println` path.

**Architecture:** The viewport component and its rendering pipeline are removed entirely. Tool output truncation is applied in the inline print path before calling `tea.Println`. Ctrl+O flips a toggle and re-prints the last tool message in the new mode.

**Tech Stack:** Go, Bubbletea (`tea.Println`, `tea.Sequence`), lipgloss

**Spec:** `docs/superpowers/specs/2026-04-01-viewport-removal-inline-migration-design.md`

---

### Task 1: Remove viewport field and dead rendering functions

**Files:**
- Modify: `ui/app.go`

- [ ] **Step 1: Remove viewport field from App struct**

In `ui/app.go`, remove the `viewport` field (line 169) from the App struct:

```go
// DELETE this line:
viewport             components.Viewport
```

- [ ] **Step 2: Remove `updateViewport()` function**

Delete the entire `updateViewport()` function (lines 2848-2874):

```go
// DELETE entire function:
func (a *App) updateViewport() {
	a.viewport = a.viewport.SetContent("")
	return
	// ... all unreachable code ...
}
```

- [ ] **Step 3: Remove `viewportRenderState()` function**

Delete the entire `viewportRenderState()` function (lines 2883-2908):

```go
// DELETE entire function:
func (a App) viewportRenderState() model.State {
	// ...
}
```

- [ ] **Step 4: Remove `renderToolMessageContent()` function**

Delete the entire `renderToolMessageContent()` function (lines 2911-2928):

```go
// DELETE entire function:
func (a App) renderToolMessageContent(msg model.Message) model.Message {
	// ...
}
```

- [ ] **Step 5: Remove all `updateViewport()` call sites**

Remove or replace each call:

```go
// Line ~300 (inside KeyMsg handler): delete the call
updated.updateViewport()

// Line ~331 (bootDoneMsg): delete the call
a.updateViewport()

// Line ~344 (default case): delete the call
a.updateViewport()

// Line ~429 (inside handleKey ctrl+o): will be replaced in Task 3
a.updateViewport()

// Line ~900 (inside handleKey): delete the call
a.updateViewport()

// Line ~1565 (inside handleEvent): delete the call
a.updateViewport()
```

- [ ] **Step 6: Remove viewport from `resizeActiveLayout()`**

In `resizeActiveLayout()` (line ~373), remove viewport sizing:

```go
func (a *App) resizeActiveLayout() {
	a.resizeInput()
	// DELETE these lines:
	// contentLines := a.viewport.TotalLines()
	// if contentLines < 1 {
	// 	contentLines = 1
	// }
	// a.viewport = a.viewport.SetSize(a.chatWidth()-4, a.desiredChatHeight(contentLines))
}
```

- [ ] **Step 7: Remove chatWidth viewport comment**

Update the `chatWidth()` comment (line ~360):

```go
// chatWidth returns the width available for the chat area.
func (a App) chatWidth() int {
	return a.width
}
```

- [ ] **Step 8: Verify build compiles**

Run: `go build ./...`
Expected: compilation errors from remaining viewport references (fixed in Task 2)

---

### Task 2: Remove viewport handlers from Update()

**Files:**
- Modify: `ui/app.go`

- [ ] **Step 1: Remove MouseMsg viewport handler**

Replace the `tea.MouseMsg` case (lines ~305-308):

```go
// REPLACE:
case tea.MouseMsg:
	var cmd tea.Cmd
	a.viewport, cmd = a.viewport.Update(msg)
	return a, cmd

// WITH:
case tea.MouseMsg:
	return a, nil
```

- [ ] **Step 2: Remove PageUp/PageDown viewport handler**

In `handleKey()`, remove the pgup/pgdown case (lines ~911-914):

```go
// DELETE entire case:
case "pgup", "pgdown":
	var cmd tea.Cmd
	a.viewport, cmd = a.viewport.Update(msg)
	return a, cmd
```

- [ ] **Step 3: Remove viewport import if unused**

Check if `components.Viewport` is still referenced anywhere. If not, the import will be cleaned up by the compiler. If there are other `components` usages (like `ThinkingSpinner`), the import stays.

- [ ] **Step 4: Verify build compiles**

Run: `go build ./...`
Expected: clean build, no errors

- [ ] **Step 5: Run tests**

Run: `go test ./ui/...`
Expected: `TestCtrlO_TogglesToolExpansion` fails (it calls `viewportRenderState` which no longer exists). Other tests pass. This is expected — fixed in Task 4.

- [ ] **Step 6: Commit**

```bash
git add ui/app.go
git commit -m "refactor: remove dead viewport rendering code

The viewport component was disabled (updateViewport returned immediately).
All UI output goes through tea.Println in inline mode. Remove the viewport
field, updateViewport, viewportRenderState, renderToolMessageContent, and
associated scroll/mouse handlers."
```

---

### Task 3: Add inline truncation to tool print path

**Files:**
- Modify: `ui/inline.go`
- Modify: `ui/app.go` (Ctrl+O handler)

- [ ] **Step 1: Add `truncateToolForPrint` helper to inline.go**

Add this function to `ui/inline.go`:

```go
// truncateToolForPrint applies collapse/truncation policy to a tool message
// before printing. When toolsExpanded is true, content is returned unchanged.
func (a App) truncateToolForPrint(msg model.Message) model.Message {
	if msg.Kind != model.MsgTool || msg.Pending || msg.Streaming {
		return msg
	}
	if *a.toolsExpanded {
		return msg
	}
	if strings.EqualFold(strings.TrimSpace(msg.ToolName), "Read") {
		msg.Content = ""
		return msg
	}
	if msg.Display == model.DisplayCollapsed {
		msg.Content = collapsedToolDetails(msg.Content, collapsedPreviewLines(msg.ToolName))
		return msg
	}
	msg.Content = truncateToolContentForTool(msg.ToolName, msg.Content)
	return msg
}
```

Note: `collapsedToolDetails`, `collapsedPreviewLines`, and `truncateToolContentForTool` already exist in `ui/app.go` and are package-level functions accessible from `inline.go`.

- [ ] **Step 2: Change `toolsExpanded` from bool to `*bool`**

In `ui/app.go`, change the field so inline.go can read it through the value-type App:

```go
// In App struct, REPLACE:
toolsExpanded    bool

// WITH:
toolsExpanded    *bool
```

In `New()` (and `NewReplay()` if it exists), initialize it:

```go
toolsExpanded: new(bool),
```

Update the Ctrl+O handler in `handleKey()`:

```go
if msg.String() == "ctrl+o" {
	*a.toolsExpanded = !*a.toolsExpanded
	return a, a.reprintLastTool()
}
```

- [ ] **Step 3: Apply truncation in `printResolvedTool`**

In `ui/inline.go`, modify `printResolvedTool`:

```go
func (a App) printResolvedTool(ev model.Event) tea.Cmd {
	msg, ok := a.resolvedToolMessage(ev)
	if !ok {
		return nil
	}
	msg = a.truncateToolForPrint(msg)
	return a.printMessage(msg)
}
```

- [ ] **Step 4: Apply truncation in `printShellFinished` fallback path**

In `ui/inline.go`, modify the fallback path of `printShellFinished` (the path where output was NOT streamed):

```go
func (a App) printShellFinished(ev model.Event, before []model.Message) tea.Cmd {
	prevMsg, hadPrev := findToolMessage(before, ev.ToolCallID)
	if hadPrev && strings.TrimSpace(prevMsg.Content) != "" {
		summary := strings.TrimSpace(ev.Summary)
		if summary == "" || summary == "completed" {
			return nil
		}
		return tea.Println(metaStyle.Render("shell " + summary))
	}

	msg, ok := a.resolvedToolMessage(ev)
	if !ok {
		return nil
	}
	if strings.TrimSpace(msg.Content) == "(No output)" && strings.TrimSpace(msg.Summary) == "completed" {
		return nil
	}
	msg = a.truncateToolForPrint(msg)
	return a.printMessage(msg)
}
```

Only the last line before `return a.printMessage(msg)` is new.

- [ ] **Step 5: Add `reprintLastTool` method**

Add to `ui/inline.go`:

```go
// reprintLastTool re-prints the most recent tool message with current
// expand/collapse state. Called when the user presses Ctrl+O.
func (a App) reprintLastTool() tea.Cmd {
	for i := len(a.state.Messages) - 1; i >= 0; i-- {
		msg := a.state.Messages[i]
		if msg.Kind == model.MsgTool && !msg.Pending && !msg.Streaming {
			msg = a.truncateToolForPrint(msg)
			return a.printMessage(msg)
		}
	}
	return nil
}
```

- [ ] **Step 6: Verify build compiles**

Run: `go build ./...`
Expected: clean build

---

### Task 4: Update tests

**Files:**
- Modify: `ui/app_tool_output_test.go`

- [ ] **Step 1: Rewrite `TestCtrlO_TogglesToolExpansion` for inline path**

Replace the existing test that used `viewportRenderState()`:

```go
func TestCtrlO_TogglesToolExpansion(t *testing.T) {
	app := New(make(chan model.Event), nil, "dev", ".", "", "model", 1024)
	app.bootActive = false
	app.state.Messages = []model.Message{{
		Kind:     model.MsgTool,
		ToolName: "Write",
		ToolArgs: "x.md",
		Display:  model.DisplayExpanded,
		Content: strings.Join([]string{
			"a", "b", "c", "d", "e", "f", "g",
		}, "\n"),
	}}

	// Default: toolsExpanded is false → truncation applied
	truncated := app.truncateToolForPrint(app.state.Messages[0])
	if !strings.Contains(truncated.Content, "ctrl+o to expand") {
		t.Fatalf("expected collapsed content with expansion hint, got:\n%s", truncated.Content)
	}

	// After Ctrl+O: toolsExpanded is true → full content
	next, _ := app.handleKey(tea.KeyMsg{Type: tea.KeyCtrlO})
	updated := next.(App)
	expanded := updated.truncateToolForPrint(updated.state.Messages[0])
	if strings.Contains(expanded.Content, "ctrl+o to expand") {
		t.Fatalf("expected expanded content after ctrl+o, got:\n%s", expanded.Content)
	}
	if !strings.Contains(expanded.Content, "\nf\ng") {
		t.Fatalf("expected full content after ctrl+o, got:\n%s", expanded.Content)
	}
}
```

- [ ] **Step 2: Run all tests**

Run: `go test ./ui/...`
Expected: all pass

- [ ] **Step 3: Run full test suite**

Run: `go test ./...`
Expected: only pre-existing `agent/context` failures

- [ ] **Step 4: Commit**

```bash
git add ui/inline.go ui/app.go ui/app_tool_output_test.go
git commit -m "feat: migrate tool output truncation to inline print path

Tool output is now collapsed by default when printed via tea.Println.
Ctrl+O toggles expand/collapse mode and re-prints the last tool result.

- DisplayCollapsed tools (Read, Grep, Glob, Skill): max 3 detail lines
- DisplayExpanded tools (Shell, Edit, Write): head 5 lines, max 12k runes
- Truncated output shows '… +N lines (ctrl+o to expand)' hint"
```

---

### Task 5: Manual verification

- [ ] **Step 1: Build binary**

Run: `go build -o mscli ./cmd/mscli`

- [ ] **Step 2: Test collapsed grep output**

Run mscli, ask agent to grep for something. Verify:
- Output shows summary + truncated details with `… +N lines (ctrl+o to expand)`
- Not all match lines are shown

- [ ] **Step 3: Test Ctrl+O expand**

Press Ctrl+O. Verify:
- Last tool result re-prints with full content below
- All match lines are now visible

- [ ] **Step 4: Test Ctrl+O collapse**

Press Ctrl+O again. Verify:
- Last tool result re-prints collapsed

- [ ] **Step 5: Test shell streaming**

Ask agent to run a shell command. Verify:
- Output streams line-by-line as before
- No truncation during streaming
- `⎿` connector appears on first output line

- [ ] **Step 6: Test permission prompt**

Ask agent to run a new shell command. Verify:
- Permission prompt appears (not "running command...")
- After approval, command executes normally
