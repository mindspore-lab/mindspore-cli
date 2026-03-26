package panels

import (
	"regexp"
	"strings"
	"testing"

	"github.com/vigo999/ms-cli/ui/model"
)

var testANSIPattern = regexp.MustCompile(`\x1b\[[0-9;]*m`)

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

func TestRenderMessagesRendersTableAlignmentSyntax(t *testing.T) {
	state := model.NewState("test", ".", "", "demo-model", 4096)
	state = state.WithMessage(model.Message{
		Kind: model.MsgAgent,
		Content: "| left | center | right |\n" +
			"| :--- | :----: | ----: |\n" +
			"| a | bb | ccc |",
	})

	rendered := RenderMessages(state, "", 100)
	plain := testANSIPattern.ReplaceAllString(rendered, "")

	for _, want := range []string{"left", "center", "right", "a", "bb", "ccc"} {
		if !strings.Contains(plain, want) {
			t.Fatalf("expected %q in rendered output, got:\n%s", want, plain)
		}
	}
	if strings.Contains(plain, ":----:") {
		t.Fatalf("expected alignment separator row to be hidden, got:\n%s", plain)
	}
}
