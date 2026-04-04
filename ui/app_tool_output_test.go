package ui

import (
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/vigo999/mindspore-cli/ui/model"
)

var testANSIPattern = regexp.MustCompile(`\x1b\[[0-9;]*m`)

func TestTruncateToolContentWithPolicy_HeadTailAndOmittedLineCount(t *testing.T) {
	content := strings.Join([]string{
		"line-1",
		"line-2",
		"line-3",
		"line-4",
		"line-5",
		"line-6",
	}, "\n")

	got := truncateToolContentWithPolicy(content, 2, 1, 1000)

	if !strings.Contains(got, "line-1\nline-2\nline-6") {
		t.Fatalf("expected head/tail preview, got:\n%s", got)
	}
	if !strings.Contains(got, "… +3 lines (ctrl+o to expand)") {
		t.Fatalf("expected omitted line hint, got:\n%s", got)
	}
}

func TestTruncateToolContentForTool_WriteUses3To5LinePreview(t *testing.T) {
	lines := make([]string, 9)
	for i := range lines {
		lines[i] = "line"
	}
	content := strings.Join(lines, "\n")

	got := truncateToolContentForTool("Write", content)

	visibleLines := strings.Count(got, "\n") + 1
	if visibleLines > 6 { // 5 visible + 1 omitted hint
		t.Fatalf("expected compact preview, got %d lines:\n%s", visibleLines, got)
	}
	if !strings.Contains(got, "… +4 lines (ctrl+o to expand)") {
		t.Fatalf("expected omitted hint, got:\n%s", got)
	}
}

func TestReadToolFinalization_HidesContent(t *testing.T) {
	pending := model.Message{
		Kind:     model.MsgTool,
		ToolName: "Read",
		ToolArgs: "configs/skills.yaml",
		Display:  model.DisplayCollapsed,
		Pending:  true,
	}

	resolved := finalizeToolMessage(pending, model.Event{
		Type:    model.ToolRead,
		Message: "skills:\n  repo: x",
		Summary: "6 lines",
	})

	if strings.TrimSpace(resolved.Content) != "" {
		t.Fatalf("expected read tool content hidden, got: %q", resolved.Content)
	}
	if resolved.Summary != "6 lines" {
		t.Fatalf("expected summary preserved, got: %q", resolved.Summary)
	}
}

func TestCtrlO_OpensToolOutputViewer(t *testing.T) {
	app := New(make(chan model.Event), nil, "dev", ".", "", "model", 1024)
	app.bootActive = false
	app.width = 80
	app.height = 24
	app.state.Messages = []model.Message{{
		Kind:     model.MsgTool,
		ToolName: "Write",
		ToolArgs: "x.md",
		Display:  model.DisplayExpanded,
		Content: strings.Join([]string{
			"a", "b", "c", "d", "e", "f", "g",
		}, "\n"),
	}}

	// Ctrl+O opens the tool output viewer
	next, _ := app.handleKey(tea.KeyMsg{Type: tea.KeyCtrlO})
	updated := next.(App)
	if updated.toolOutputView == nil {
		t.Fatal("expected tool output view to be open after ctrl+o")
	}
	if updated.toolOutputView.msg.ToolName != "Write" {
		t.Fatalf("expected Write tool in viewer, got %q", updated.toolOutputView.msg.ToolName)
	}

	// View should show full content
	view := updated.View()
	if !strings.Contains(view, "f") || !strings.Contains(view, "g") {
		t.Fatalf("expected full content in viewer, got:\n%s", view)
	}

	// Ctrl+O again closes it (goes through handleToolOutputViewKey)
	next, _ = updated.handleKey(tea.KeyMsg{Type: tea.KeyCtrlO})
	closed := next.(App)
	if closed.toolOutputView != nil {
		t.Fatal("expected tool output view to be closed after second ctrl+o")
	}
}

func TestCtrlO_OpensStreamingToolOutputViewerAndTracksLiveUpdates(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 20})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:       model.ToolCallStart,
		ToolName:   "shell",
		ToolCallID: "call-shell-live",
		Message:    "go test ./ui",
	})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:       model.CmdStarted,
		ToolName:   "shell",
		ToolCallID: "call-shell-live",
	})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:       model.CmdOutput,
		ToolName:   "shell",
		ToolCallID: "call-shell-live",
		Message:    "line-1",
	})
	app = next.(App)

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyCtrlO})
	app = next.(App)

	if app.toolOutputView == nil {
		t.Fatal("expected ctrl+o to open viewer for streaming tool output")
	}
	if got, want := app.toolOutputView.toolCallID, "call-shell-live"; got != want {
		t.Fatalf("toolCallID = %q, want %q", got, want)
	}

	view := testANSIPattern.ReplaceAllString(app.View(), "")
	if !strings.Contains(view, "line-1") {
		t.Fatalf("expected initial streamed output in viewer, got:\n%s", view)
	}

	next, _ = app.handleEvent(model.Event{
		Type:       model.CmdOutput,
		ToolName:   "shell",
		ToolCallID: "call-shell-live",
		Message:    "line-2",
	})
	app = next.(App)

	view = testANSIPattern.ReplaceAllString(app.View(), "")
	if !strings.Contains(view, "line-2") {
		t.Fatalf("expected viewer to refresh with new streamed output, got:\n%s", view)
	}
}

func TestShellCmdOutputRendersLiveInUI(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 20})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:       model.ToolCallStart,
		ToolName:   "shell",
		ToolCallID: "call-shell-1",
		Message:    "go test ./ui",
	})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:       model.CmdStarted,
		ToolName:   "shell",
		ToolCallID: "call-shell-1",
	})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:       model.CmdOutput,
		ToolName:   "shell",
		ToolCallID: "call-shell-1",
		Message:    "=== RUN   TestShell",
	})
	app = next.(App)

	if app.state.Messages[0].Pending {
		t.Fatal("expected shell message to stop being pending after streamed output")
	}
	if !app.state.Messages[0].Streaming {
		t.Fatal("expected shell message to stay streaming before command finishes")
	}
	if got, want := app.state.Messages[0].Content, "=== RUN   TestShell"; got != want {
		t.Fatalf("content = %q, want %q", got, want)
	}
	if got, want := app.state.WaitKind, model.WaitTool; got != want {
		t.Fatalf("wait kind = %v, want %v", got, want)
	}

	view := app.View()
	if !strings.Contains(view, "=== RUN   TestShell") {
		t.Fatalf("expected streamed shell output in live view, got:\n%s", view)
	}

	next, _ = app.handleEvent(model.Event{
		Type:       model.CmdFinished,
		ToolName:   "shell",
		ToolCallID: "call-shell-1",
		Message:    "=== RUN   TestShell\nPASS",
		Summary:    "completed",
	})
	app = next.(App)

	if app.state.Messages[0].Streaming {
		t.Fatal("expected shell message streaming flag cleared after finish")
	}
	if got, want := app.state.WaitKind, model.WaitNone; got != want {
		t.Fatalf("wait kind = %v, want %v", got, want)
	}
	if got, want := app.state.Messages[0].Summary, "completed"; got != want {
		t.Fatalf("summary = %q, want %q", got, want)
	}
	if got, want := app.state.Messages[0].Content, "=== RUN   TestShell\nPASS"; got != want {
		t.Fatalf("content = %q, want %q", got, want)
	}
}

func TestAgentReplyDeltaRendersLiveInUI(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 20})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{Type: model.AgentThinking})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:    model.AgentReplyDelta,
		Message: "partial reply",
	})
	app = next.(App)

	if got, want := len(app.state.Messages), 1; got != want {
		t.Fatalf("message count = %d, want %d", got, want)
	}
	if !app.state.Messages[0].Streaming {
		t.Fatal("expected agent message to stay streaming before reply completes")
	}
	if got, want := app.state.Messages[0].Content, "partial reply"; got != want {
		t.Fatalf("content = %q, want %q", got, want)
	}

	view := testANSIPattern.ReplaceAllString(app.View(), "")
	if !strings.Contains(view, "partial reply") {
		t.Fatalf("expected streamed agent reply in live view, got:\n%s", view)
	}
}

func TestAgentReplyDeltaUsesAvailableViewportHeightInsteadOfFixedEightLines(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 20})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{Type: model.AgentThinking})
	app = next.(App)

	lines := make([]string, 10)
	for i := range lines {
		lines[i] = "agent-line-" + strconv.Itoa(i+1)
	}

	next, _ = app.handleEvent(model.Event{
		Type:    model.AgentReplyDelta,
		Message: strings.Join(lines, "\n"),
	})
	app = next.(App)

	view := testANSIPattern.ReplaceAllString(app.View(), "")
	if !strings.Contains(view, "agent-line-1") || !strings.Contains(view, "agent-line-10") {
		t.Fatalf("expected full streamed agent preview to use available viewport height, got:\n%s", view)
	}
}

func TestShellStreamingPreviewUsesFixedEightLineTailAndStatusOnCommandLine(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 20})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:       model.ToolCallStart,
		ToolName:   "shell",
		ToolCallID: "call-shell-many-lines",
		Message:    "go test ./ui",
	})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:       model.CmdStarted,
		ToolName:   "shell",
		ToolCallID: "call-shell-many-lines",
	})
	app = next.(App)

	lines := make([]string, 10)
	for i := range lines {
		lines[i] = "shell-line-" + strconv.Itoa(i+1)
	}

	next, _ = app.handleEvent(model.Event{
		Type:       model.CmdOutput,
		ToolName:   "shell",
		ToolCallID: "call-shell-many-lines",
		Message:    strings.Join(lines, "\n"),
	})
	app = next.(App)
	app.state.WaitStartedAt = time.Now().Add(-2 * time.Second)

	view := testANSIPattern.ReplaceAllString(app.View(), "")
	if strings.Contains(view, "shell-line-1\n") || strings.Contains(view, "shell-line-2\n") {
		t.Fatalf("expected shell preview to keep only the last 8 output lines, got:\n%s", view)
	}
	if !strings.Contains(view, "shell-line-3") || !strings.Contains(view, "shell-line-10") {
		t.Fatalf("expected shell preview tail lines in live view, got:\n%s", view)
	}
	if !strings.Contains(view, "Bash($ go test ./ui) running, ctrl+o to expand... 2s") {
		t.Fatalf("expected shell preview status on command line, got:\n%s", view)
	}
}

func TestRenderShellActivePreviewDoesNotPadToEightLinesBeforeOutputArrives(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false
	app.state.WaitKind = model.WaitTool
	app.state.WaitStartedAt = time.Now().Add(-2 * time.Second)

	preview := testANSIPattern.ReplaceAllString(app.renderShellActivePreview(model.Message{
		Kind:     model.MsgTool,
		ToolName: "Bash",
		ToolArgs: "$ go test ./ui",
		Content:  "line-1",
	}, 9), "")

	lines := strings.Split(preview, "\n")
	if got, want := len(lines), 2; got != want {
		t.Fatalf("preview line count = %d, want %d:\n%s", got, want, preview)
	}
	if !strings.Contains(lines[0], "Bash($ go test ./ui) running, ctrl+o to expand... 2s") {
		t.Fatalf("expected status on command line, got: %q", lines[0])
	}
	if !strings.Contains(lines[1], "line-1") {
		t.Fatalf("expected single output line without padding, got: %q", lines[1])
	}
}
