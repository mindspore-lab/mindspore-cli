package ui

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
)

func TestViewShowsTopBarTitle(t *testing.T) {
	app := New(nil, nil, "MindSpore CLI. test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 24})
	app = next.(App)

	view := app.View()
	if !strings.Contains(view, "MindSpore CLI. test") {
		t.Fatalf("expected top bar title in view, got:\n%s", view)
	}
}
