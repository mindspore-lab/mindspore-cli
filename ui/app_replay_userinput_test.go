package ui

import (
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

func TestEnterIncrementsUserEventPrintSuppressionForFreeText(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 20})
	app = next.(App)
	app.input.Model.SetValue("hello replay")

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)

	if got, want := app.suppressUserEventPrints, 1; got != want {
		t.Fatalf("suppressUserEventPrints after free-text enter = %d, want %d", got, want)
	}
}

func TestEnterDoesNotSuppressUserEventPrintForSlashCommand(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 20})
	app = next.(App)
	app.input.Model.SetValue("/replay sess_123")

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)

	if got := app.suppressUserEventPrints; got != 0 {
		t.Fatalf("suppressUserEventPrints after slash enter = %d, want 0", got)
	}
}

func TestReplayUserInputGeneratesPrintWhenNotLocallySuppressed(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	cmd := app.eventPrintCmd(model.Event{
		Type:    model.UserInput,
		Message: "historical prompt",
	}, nil, false)
	if cmd == nil {
		t.Fatal("expected replayed user input to produce a print command")
	}

	cmd = app.eventPrintCmd(model.Event{
		Type:    model.UserInput,
		Message: "local prompt",
	}, nil, true)
	if cmd != nil {
		t.Fatal("expected locally echoed user input to skip duplicate print command")
	}
}
