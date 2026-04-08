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
