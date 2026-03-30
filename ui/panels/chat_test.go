package panels

import (
	"regexp"
	"strings"
	"testing"

	"github.com/vigo999/mindspore-code/ui/model"
)

var testANSIPattern = regexp.MustCompile(`\x1b\[[0-9;]*m`)

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

func TestRenderMessagesRendersAgentMarkdown(t *testing.T) {
	state := model.NewState("test", ".", "", "demo-model", 4096)
	state = state.WithMessage(model.Message{
		Kind: model.MsgAgent,
		Content: "# Title\n\n- item one\n1. item two\n\n`inline`\n\n```go\nfmt.Println(\"hi\")\n```" +
			"\n\n[docs](https://example.com)",
	})

	rendered := RenderMessages(state, "", 80)
	plain := testANSIPattern.ReplaceAllString(rendered, "")

	if strings.Contains(plain, "# Title") {
		t.Fatalf("expected heading markers to be removed, got:\n%s", plain)
	}
	if strings.Contains(plain, "- item one") {
		t.Fatalf("expected bullet markers to be rendered, got:\n%s", plain)
	}
	if strings.Contains(plain, "```") {
		t.Fatalf("expected code fences to be removed, got:\n%s", plain)
	}
	for _, want := range []string{"Title", "• item one", "1. item two", "inline", "fmt.Println(\"hi\")", "docs (https://example.com)"} {
		if !strings.Contains(plain, want) {
			t.Fatalf("expected %q in rendered output, got:\n%s", want, plain)
		}
	}
}

func TestRenderMessagesRendersMarkdownTable(t *testing.T) {
	state := model.NewState("test", ".", "", "demo-model", 4096)
	state = state.WithMessage(model.Message{
		Kind: model.MsgAgent,
		Content: "| 类别 | 内容 |\n" +
			"|------|------|\n" +
			"| 核心入口 | cmd/ - 命令行命令定义 |\n" +
			"| 业务模块 | agent/ - AI Agent 相关（含8个skill）、runtime/ - 运行时、workflow/ - 工作流 |",
	})

	rendered := RenderMessages(state, "", 120)
	plain := testANSIPattern.ReplaceAllString(rendered, "")

	for _, want := range []string{"┌", "┐", "类别", "内容", "核心入口", "业务模块", "cmd/ - 命令行命令定义"} {
		if !strings.Contains(plain, want) {
			t.Fatalf("expected %q in rendered output, got:\n%s", want, plain)
		}
	}
	if strings.Contains(plain, "|------|") {
		t.Fatalf("expected markdown separator row to be hidden, got:\n%s", plain)
	}
}

func TestRenderMessagesRendersTaskAndNestedLists(t *testing.T) {
	state := model.NewState("test", ".", "", "demo-model", 4096)
	state = state.WithMessage(model.Message{
		Kind: model.MsgAgent,
		Content: "- [ ] todo\n" +
			"- [x] done\n" +
			"  - child item\n" +
			"    1. ordered child",
	})

	rendered := RenderMessages(state, "", 100)
	plain := testANSIPattern.ReplaceAllString(rendered, "")

	for _, want := range []string{"[ ] todo", "[x] done", "  • child item", "    1. ordered child"} {
		if !strings.Contains(plain, want) {
			t.Fatalf("expected %q in rendered output, got:\n%s", want, plain)
		}
	}
}

func TestRenderMessagesRendersCodeFenceLangAndStrikethrough(t *testing.T) {
	state := model.NewState("test", ".", "", "demo-model", 4096)
	state = state.WithMessage(model.Message{
		Kind:    model.MsgAgent,
		Content: "~~deprecated~~ and __bold__ and _italic_\n\n```bash\necho hi\n```",
	})

	rendered := RenderMessages(state, "", 100)
	plain := testANSIPattern.ReplaceAllString(rendered, "")

	for _, want := range []string{"deprecated", "bold", "italic", "bash", "echo hi"} {
		if !strings.Contains(plain, want) {
			t.Fatalf("expected %q in rendered output, got:\n%s", want, plain)
		}
	}
	if strings.Contains(plain, "```bash") {
		t.Fatalf("expected fenced code marker to be hidden, got:\n%s", plain)
	}
}

func TestRenderMessagesInlineCodeAndFenceMarkers(t *testing.T) {
	state := model.NewState("test", ".", "", "demo-model", 4096)
	state = state.WithMessage(model.Message{
		Kind:    model.MsgAgent,
		Content: "Use `<words>` and `inline` here.\n\n```txt\nnot a fence marker\n```",
	})

	rendered := RenderMessages(state, "", 80)
	plain := testANSIPattern.ReplaceAllString(rendered, "")

	for _, want := range []string{"<words>", "inline", "txt", "not a fence marker"} {
		if !strings.Contains(plain, want) {
			t.Fatalf("expected %q in rendered output, got:\n%s", want, plain)
		}
	}
	if strings.Contains(plain, "`") {
		t.Fatalf("expected inline and fenced code markers to be hidden, got:\n%s", plain)
	}
}

func TestRenderMessagesKeepsWideTableBordersStable(t *testing.T) {
	state := model.NewState("test", ".", "", "demo-model", 4096)
	state = state.WithMessage(model.Message{
		Kind: model.MsgAgent,
		Content: "| Name | Description | Notes |\n" +
			"| ---- | ----------- | ----- |\n" +
			"| alpha | this cell is intentionally very wide to exercise truncation | keep border stable |\n" +
			"| beta | another wide value that used to trigger outer wrapping | second row |",
	})

	rendered := RenderMessages(state, "", 42)
	plain := testANSIPattern.ReplaceAllString(rendered, "")
	lines := strings.Split(plain, "\n")

	var tableLines []string
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "┌") || strings.HasPrefix(trimmed, "├") || strings.HasPrefix(trimmed, "└") || strings.HasPrefix(trimmed, "│") {
			tableLines = append(tableLines, trimmed)
		}
	}
	if len(tableLines) < 5 {
		t.Fatalf("expected full rendered table, got:\n%s", plain)
	}
	for _, fragment := range []string{"┐", "┤", "┘"} {
		if !strings.Contains(plain, fragment) {
			t.Fatalf("expected %q in rendered output, got:\n%s", fragment, plain)
		}
	}
	counts := map[string]int{"┌": 0, "├": 0, "└": 0}
	for _, line := range tableLines {
		switch {
		case strings.HasPrefix(line, "┌"):
			counts["┌"]++
		case strings.HasPrefix(line, "├"):
			counts["├"]++
		case strings.HasPrefix(line, "└"):
			counts["└"]++
		}
	}
	for border, count := range counts {
		if count != 1 {
			t.Fatalf("expected exactly one %s border line, got %d in:\n%s", border, count, plain)
		}
	}
	for _, want := range []string{"this cell is", "intentionall", "another wide", "wrapping"} {
		if !strings.Contains(plain, want) {
			t.Fatalf("expected wrapped table content %q, got:\n%s", want, plain)
		}
	}
}

func TestRenderMessagesRendersCodeBlockAsDistinctBlock(t *testing.T) {
	state := model.NewState("test", ".", "", "demo-model", 4096)
	state = state.WithMessage(model.Message{
		Kind:    model.MsgAgent,
		Content: "before\n\n```py\nprint(\"hi\")\n```\n\nafter",
	})

	rendered := RenderMessages(state, "", 60)
	plain := testANSIPattern.ReplaceAllString(rendered, "")

	for _, want := range []string{"before", "py", "┃ print(\"hi\")", "after"} {
		if !strings.Contains(plain, want) {
			t.Fatalf("expected %q in rendered output, got:\n%s", want, plain)
		}
	}
	if strings.Contains(plain, "```py") {
		t.Fatalf("expected fenced code marker to be hidden, got:\n%s", plain)
	}
}

func TestRenderMessagesRendersTableInlineCodeWithoutBreakingCodeSpan(t *testing.T) {
	state := model.NewState("test", ".", "", "demo-model", 4096)
	state = state.WithMessage(model.Message{
		Kind: model.MsgAgent,
		Content: "| File | Description |\n" +
			"| ---- | ----------- |\n" +
			"| `manager_test.go` | Tests for context manager |",
	})

	rendered := RenderMessages(state, "", 38)
	plain := testANSIPattern.ReplaceAllString(rendered, "")

	if strings.Contains(plain, "manager_test.g\no") {
		t.Fatalf("expected inline code token to stay on one table line, got:\n%s", plain)
	}
	if !strings.Contains(plain, "manager") {
		t.Fatalf("expected file name content to remain visible, got:\n%s", plain)
	}
}
