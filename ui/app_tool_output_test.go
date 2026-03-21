package ui

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/vigo999/ms-cli/ui/model"
)

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

	collapsed := app.viewportRenderState().Messages[0].Content
	if !strings.Contains(collapsed, "ctrl+o to expand") {
		t.Fatalf("expected collapsed content with expansion hint, got:\n%s", collapsed)
	}

	next, _ := app.handleKey(tea.KeyMsg{Type: tea.KeyCtrlO})
	updated := next.(App)
	expanded := updated.viewportRenderState().Messages[0].Content
	if strings.Contains(expanded, "ctrl+o to expand") {
		t.Fatalf("expected expanded content after ctrl+o, got:\n%s", expanded)
	}
	if !strings.Contains(expanded, "\nf\ng") {
		t.Fatalf("expected full content after ctrl+o, got:\n%s", expanded)
	}
}
