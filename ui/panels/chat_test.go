package panels

import (
	"strings"
	"testing"

	"github.com/vigo999/ms-cli/ui/model"
)

func TestRenderMessages_ToolPendingShowsOneCallLine(t *testing.T) {
	state := model.State{
		Messages: []model.Message{
			{
				Kind:     model.MsgTool,
				ToolName: "Write",
				ToolArgs: "none-1.md",
				Display:  model.DisplayCollapsed,
				Pending:  true,
			},
		},
	}

	view := RenderMessages(state, "", 80, true)
	if !strings.Contains(view, "⏺ Write(none-1.md)") {
		t.Fatalf("expected pending write call line, got:\n%s", view)
	}
	if strings.Contains(view, "⎿") {
		t.Fatalf("expected pending tool to not render result summary, got:\n%s", view)
	}
}

func TestRenderMessages_ToolSuccessShowsSummaryAndDetails(t *testing.T) {
	state := model.State{
		Messages: []model.Message{
			{
				Kind:     model.MsgTool,
				ToolName: "Write",
				ToolArgs: "none.md",
				Display:  model.DisplayExpanded,
				Content:  "Wrote 1 lines to none.md\n1 (No content)",
			},
		},
	}

	view := RenderMessages(state, "", 80, true)
	if !strings.Contains(view, "⏺ Write(none.md)") {
		t.Fatalf("expected success call line, got:\n%s", view)
	}
	if !strings.Contains(view, "⎿  Wrote 1 lines to none.md") {
		t.Fatalf("expected success summary line, got:\n%s", view)
	}
	if !strings.Contains(view, "1 (No content)") {
		t.Fatalf("expected success detail line, got:\n%s", view)
	}
}

func TestRenderMessages_ToolFailureShowsErrorSummaryAndDetails(t *testing.T) {
	state := model.State{
		Messages: []model.Message{
			{
				Kind:     model.MsgTool,
				ToolName: "Write",
				ToolArgs: "none.md",
				Display:  model.DisplayError,
				Content:  "User rejected write to none.md\n1 (No content)",
			},
		},
	}

	view := RenderMessages(state, "", 80, true)
	if !strings.Contains(view, "⏺ Write(none.md)") {
		t.Fatalf("expected failure call line, got:\n%s", view)
	}
	if !strings.Contains(view, "⎿  User rejected write to none.md") {
		t.Fatalf("expected failure summary line, got:\n%s", view)
	}
	if !strings.Contains(view, "1 (No content)") {
		t.Fatalf("expected failure detail line, got:\n%s", view)
	}
}
