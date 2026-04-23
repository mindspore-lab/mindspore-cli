package ui

import (
	"reflect"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

func TestPrintMessage_ToolMessagesGetLeadingBlankLine(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	cmd := app.printMessage(model.Message{
		Kind:     model.MsgTool,
		ToolName: "Bash",
		ToolArgs: "$ which uv",
		Display:  model.DisplayCollapsed,
		Content:  "completed\n/Users/townwish/.local/bin/uv",
	})
	if cmd == nil {
		t.Fatal("expected non-nil print command")
	}

	msg := cmd()
	value := reflect.ValueOf(msg)
	if value.Kind() != reflect.Slice || value.Len() != 2 {
		t.Fatalf("expected sequence with blank line + tool block, got %#v", msg)
	}

	first, ok := value.Index(0).Interface().(tea.Cmd)
	if !ok {
		t.Fatalf("expected first sequence element to be tea.Cmd, got %#v", value.Index(0))
	}
	second, ok := value.Index(1).Interface().(tea.Cmd)
	if !ok {
		t.Fatalf("expected second sequence element to be tea.Cmd, got %#v", value.Index(1))
	}

	firstMsg := first()
	firstValue := reflect.ValueOf(firstMsg)
	if got := firstValue.FieldByName("messageBody").String(); got != "" {
		t.Fatalf("expected first print line to be blank, got %q", got)
	}

	secondMsg := second()
	secondValue := reflect.ValueOf(secondMsg)
	secondBody := secondValue.FieldByName("messageBody").String()
	if !strings.Contains(secondBody, "✓ Bash($ which uv)") {
		t.Fatalf("expected second print line to contain tool render, got %q", secondBody)
	}
}

func TestWindowResizeClearsAndReprintsHistory(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false
	app.state.Messages = []model.Message{
		{Kind: model.MsgUser, Content: "hello"},
		{Kind: model.MsgAgent, Content: "world"},
		{
			Kind:     model.MsgTool,
			ToolName: "Bash",
			ToolArgs: "$ echo one",
			Display:  model.DisplayCollapsed,
			Content:  "one",
			Summary:  "completed",
		},
	}

	next, cmd := app.Update(tea.WindowSizeMsg{Width: 100, Height: 24})
	_ = next.(App)
	if cmd == nil {
		t.Fatal("expected resize to return history replay command")
	}

	msg := cmd()
	value := reflect.ValueOf(msg)
	if value.Kind() != reflect.Slice {
		t.Fatalf("expected sequence command, got %#v", msg)
	}

	var printed strings.Builder
	for i := 0; i < value.Len(); i++ {
		cmd, ok := value.Index(i).Interface().(tea.Cmd)
		if !ok || cmd == nil {
			continue
		}
		out := cmd()
		outValue := reflect.ValueOf(out)
		if outValue.Kind() != reflect.Struct {
			continue
		}
		body := outValue.FieldByName("messageBody")
		if body.IsValid() && body.Kind() == reflect.String {
			printed.WriteString(body.String())
			printed.WriteByte('\n')
		}
	}
	got := testANSIPattern.ReplaceAllString(printed.String(), "")
	for _, want := range []string{"MindSpore CLI", "hello", "world", "Bash($ echo one)", "one"} {
		if !strings.Contains(got, want) {
			t.Fatalf("expected replayed history to contain %q, got:\n%s", want, got)
		}
	}
}
