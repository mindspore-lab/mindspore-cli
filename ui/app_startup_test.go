package ui

import (
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/vigo999/mindspore-cli/ui/model"
)

func TestStartupToolMessageRecordedInState(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 24})
	app = next.(App)

	next, cmd := app.handleEvent(model.Event{
		Type:     model.ToolSkill,
		ToolName: "mindspore-skills",
		Summary:  "shared skills repo update available: 25002f2 -> 1bef901. enter y to update or n to skip.",
	})
	app = next.(App)

	found := false
	for _, msg := range app.state.Messages {
		if msg.ToolName == "mindspore-skills" {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("expected startup tool message to be recorded in state")
	}
	if cmd == nil {
		t.Fatal("expected inline event command for tool message")
	}
}
