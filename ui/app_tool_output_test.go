package ui

import (
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
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

func TestCtrlO_OpensTranscriptViewer(t *testing.T) {
	app := New(make(chan model.Event), nil, "dev", ".", "", "model", 1024)
	app.bootActive = false
	app.width = 80
	app.height = 24
	app.state.Messages = []model.Message{{
		Kind:    model.MsgUser,
		Content: "show the file",
	}, {
		Kind:     model.MsgTool,
		ToolName: "Write",
		ToolArgs: "x.md",
		Display:  model.DisplayExpanded,
		Content: strings.Join([]string{
			"a", "b", "c", "d", "e", "f", "g",
		}, "\n"),
	}}

	// Ctrl+O opens the transcript viewer.
	next, _ := app.handleKey(tea.KeyMsg{Type: tea.KeyCtrlO})
	updated := next.(App)
	if updated.transcriptView == nil {
		t.Fatal("expected transcript view to be open after ctrl+o")
	}

	// View should show the full transcript, not only the latest tool body.
	view := updated.View()
	if !strings.Contains(view, "show the file") || !strings.Contains(view, "f") || !strings.Contains(view, "g") {
		t.Fatalf("expected full history in viewer, got:\n%s", view)
	}

	// Ctrl+O again closes it (goes through handleTranscriptViewKey).
	next, _ = updated.handleKey(tea.KeyMsg{Type: tea.KeyCtrlO})
	closed := next.(App)
	if closed.transcriptView != nil {
		t.Fatal("expected transcript view to be closed after second ctrl+o")
	}
}

func TestCtrlO_TranscriptViewerIncludesAllToolHistory(t *testing.T) {
	app := New(make(chan model.Event), nil, "dev", ".", "", "model", 1024)
	app.bootActive = false
	app.width = 100
	app.height = 30
	app.state.Messages = []model.Message{
		{Kind: model.MsgUser, Content: "first command"},
		{
			Kind:     model.MsgTool,
			ToolName: "Bash",
			ToolArgs: "$ echo first",
			Display:  model.DisplayCollapsed,
			Content:  "first-output",
			Summary:  "completed",
		},
		{Kind: model.MsgAgent, Content: "now second"},
		{
			Kind:     model.MsgTool,
			ToolName: "Bash",
			ToolArgs: "$ echo second",
			Display:  model.DisplayCollapsed,
			Content:  "second-output",
			Summary:  "completed",
		},
	}

	next, _ := app.handleKey(tea.KeyMsg{Type: tea.KeyCtrlO})
	app = next.(App)

	view := testANSIPattern.ReplaceAllString(app.View(), "")
	if !strings.Contains(view, "first-output") || !strings.Contains(view, "second-output") {
		t.Fatalf("expected transcript viewer to include all tool history, got:\n%s", view)
	}
}

func TestCtrlO_OpensTranscriptViewerAndTracksStreamingToolUpdates(t *testing.T) {
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

	if app.transcriptView == nil {
		t.Fatal("expected ctrl+o to open transcript viewer")
	}

	view := testANSIPattern.ReplaceAllString(app.View(), "")
	if !strings.Contains(view, "line-1") {
		t.Fatalf("expected initial streamed output in transcript viewer, got:\n%s", view)
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
		t.Fatalf("expected transcript viewer to refresh with new streamed output, got:\n%s", view)
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

func TestToolEditPreservesMetaInResolvedMessage(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.handleEvent(model.Event{
		Type:       model.ToolCallStart,
		ToolName:   "edit",
		ToolCallID: "call-edit-1",
		Message:    "sample.txt",
	})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:       model.ToolEdit,
		ToolName:   "edit",
		ToolCallID: "call-edit-1",
		Message:    "Edited: sample.txt\n- old\n+ new",
		Meta: map[string]any{
			"edit_diff": map[string]any{
				"path":   "sample.txt",
				"header": "@@ -1,1 +1,1 @@",
				"lines":  []string{"-old", "+new"},
			},
		},
	})
	app = next.(App)

	if len(app.state.Messages) != 1 {
		t.Fatalf("message count = %d, want 1", len(app.state.Messages))
	}
	diff, ok := app.state.Messages[0].Meta["edit_diff"].(map[string]any)
	if !ok {
		t.Fatalf("expected edit diff meta in resolved message, got %#v", app.state.Messages[0].Meta)
	}
	if got, _ := diff["path"].(string); got != "sample.txt" {
		t.Fatalf("diff path = %q, want sample.txt", got)
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

func TestBackgroundModelWorkShowsWorkingBetweenAgentTextAndToolCall(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 20})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{Type: model.AgentThinking})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:    model.AgentReplyDelta,
		Message: "好的，我来处理。",
	})
	app = next.(App)

	view := testANSIPattern.ReplaceAllString(app.View(), "")
	if !strings.Contains(view, "好的，我来处理。") {
		t.Fatalf("expected streamed text in live view before background work, got:\n%s", view)
	}

	next, _ = app.handleEvent(model.Event{Type: model.AgentBackgroundWork})
	app = next.(App)

	view = testANSIPattern.ReplaceAllString(app.View(), "")
	if !strings.Contains(view, "Working...") {
		t.Fatalf("expected Working... during background model work, got:\n%s", view)
	}
	if strings.Contains(view, "好的，我来处理。") {
		t.Fatalf("expected live preview to switch away from streaming text during background work, got:\n%s", view)
	}

	next, _ = app.handleEvent(model.Event{
		Type:       model.ToolCallStart,
		ToolName:   "write",
		ToolCallID: "call-write-1",
		Message:    `{"path":"README.md"}`,
	})
	app = next.(App)

	view = testANSIPattern.ReplaceAllString(app.View(), "")
	if strings.Contains(view, "Working...") {
		t.Fatalf("expected Working... cleared once tool call starts, got:\n%s", view)
	}
	if !strings.Contains(view, "Write") {
		t.Fatalf("expected pending tool preview after tool call start, got:\n%s", view)
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

func TestShellInterruptedResolvesStreamingToolAndStopsRunningState(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 20})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:       model.ToolCallStart,
		ToolName:   "shell",
		ToolCallID: "call-shell-interrupt",
		Message:    "go test ./ui",
	})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:       model.CmdStarted,
		ToolName:   "shell",
		ToolCallID: "call-shell-interrupt",
	})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:       model.CmdOutput,
		ToolName:   "shell",
		ToolCallID: "call-shell-interrupt",
		Message:    "partial output",
	})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:       model.ToolInterrupted,
		ToolName:   "shell",
		ToolCallID: "call-shell-interrupt",
		Message:    "partial output",
		Summary:    "interrupted",
	})
	app = next.(App)

	if got, want := app.state.WaitKind, model.WaitNone; got != want {
		t.Fatalf("wait kind = %v, want %v", got, want)
	}
	if app.state.Messages[0].Pending {
		t.Fatal("expected interrupted shell message to clear pending flag")
	}
	if app.state.Messages[0].Streaming {
		t.Fatal("expected interrupted shell message to clear streaming flag")
	}
	if got, want := app.state.Messages[0].Display, model.DisplayWarning; got != want {
		t.Fatalf("display = %v, want %v", got, want)
	}
	if got, want := app.state.Messages[0].Summary, "interrupted"; got != want {
		t.Fatalf("summary = %q, want %q", got, want)
	}
	if got, want := app.state.Messages[0].Content, "partial output"; got != want {
		t.Fatalf("content = %q, want %q", got, want)
	}

	view := testANSIPattern.ReplaceAllString(app.View(), "")
	if strings.Contains(view, "running command...") {
		t.Fatalf("expected interrupted view to stop showing running status, got:\n%s", view)
	}
}

func TestShellInterruptedAfterLargeOutputKeepsRecentWindow(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 20})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:       model.ToolCallStart,
		ToolName:   "shell",
		ToolCallID: "call-shell-large-interrupt",
		Message:    "go test ./ui",
	})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:       model.CmdStarted,
		ToolName:   "shell",
		ToolCallID: "call-shell-large-interrupt",
	})
	app = next.(App)

	lines := make([]string, 12000)
	for i := range lines {
		lines[i] = "line-" + strconv.Itoa(i+1)
	}
	largeOutput := strings.Join(lines, "\n")

	next, _ = app.handleEvent(model.Event{
		Type:       model.CmdOutput,
		ToolName:   "shell",
		ToolCallID: "call-shell-large-interrupt",
		Message:    largeOutput,
	})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:       model.ToolInterrupted,
		ToolName:   "shell",
		ToolCallID: "call-shell-large-interrupt",
		Message:    "line-11999\nline-12000",
		Summary:    "interrupted",
	})
	app = next.(App)

	got := app.state.Messages[0].Content
	if !strings.Contains(got, "line-12000") {
		t.Fatalf("expected interrupted content to keep latest output, got suffix:\n%s", got)
	}
	if strings.Contains(got, "line-1\n") || strings.HasPrefix(got, "line-1") {
		t.Fatalf("expected interrupted content to drop oldest output once truncated, got prefix:\n%s", got)
	}
	if !strings.Contains(got, "[output truncated]") {
		t.Fatalf("expected interrupted content to indicate truncation, got:\n%s", got)
	}
	if app.state.Messages[0].Streaming {
		t.Fatal("expected interrupted shell message to stop streaming after large output")
	}
	if app.state.Messages[0].Pending {
		t.Fatal("expected interrupted shell message to stop pending after large output")
	}
}

func TestShellLateCmdStartedWithoutCallIDDoesNotCreatePhantomActiveTool(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 20})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:       model.ToolCallStart,
		ToolName:   "shell",
		ToolCallID: "call-shell-interrupt-no-id",
		Message:    "uv run train.py",
	})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:       model.CmdStarted,
		ToolName:   "shell",
		ToolCallID: "call-shell-interrupt-no-id",
	})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:       model.CmdOutput,
		ToolName:   "shell",
		ToolCallID: "call-shell-interrupt-no-id",
		Message:    "partial output",
	})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:       model.ToolInterrupted,
		ToolName:   "shell",
		ToolCallID: "call-shell-interrupt-no-id",
		Message:    "partial output",
		Summary:    "interrupted",
	})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:     model.CmdStarted,
		ToolName: "shell",
	})
	app = next.(App)

	if got, want := len(app.state.Messages), 1; got != want {
		t.Fatalf("message count = %d, want %d", got, want)
	}
	if _, ok := app.lastActiveTool(); ok {
		t.Fatal("expected no active tool after stray shell start without call id")
	}

	view := testANSIPattern.ReplaceAllString(app.View(), "")
	if strings.Contains(view, "Bash() running, ctrl+o to expand...") {
		t.Fatalf("expected stray shell start to be ignored, got:\n%s", view)
	}
}

func TestShellLateCmdOutputAfterInterruptDoesNotResurrectInterruptedTool(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 20})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:       model.ToolCallStart,
		ToolName:   "shell",
		ToolCallID: "call-shell-late-output",
		Message:    "uv run train.py",
	})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:       model.CmdStarted,
		ToolName:   "shell",
		ToolCallID: "call-shell-late-output",
	})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:       model.CmdOutput,
		ToolName:   "shell",
		ToolCallID: "call-shell-late-output",
		Message:    "partial output",
	})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:       model.ToolInterrupted,
		ToolName:   "shell",
		ToolCallID: "call-shell-late-output",
		Message:    "partial output",
		Summary:    "interrupted",
	})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type:       model.CmdOutput,
		ToolName:   "shell",
		ToolCallID: "call-shell-late-output",
		Message:    "late output should be ignored",
	})
	app = next.(App)

	if got, want := len(app.state.Messages), 1; got != want {
		t.Fatalf("message count = %d, want %d", got, want)
	}
	if app.state.Messages[0].Streaming {
		t.Fatal("expected interrupted shell message to remain non-streaming after late output")
	}
	if app.state.Messages[0].Pending {
		t.Fatal("expected interrupted shell message to remain non-pending after late output")
	}
	if got, want := app.state.Messages[0].Content, "partial output"; got != want {
		t.Fatalf("content = %q, want %q", got, want)
	}
	if _, ok := app.lastActiveTool(); ok {
		t.Fatal("expected no active tool after late output for interrupted shell")
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
