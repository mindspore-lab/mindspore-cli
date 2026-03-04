package ui

import (
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/vigo999/ms-cli/ui/model"
)

func TestCtrlC_FirstInterruptSecondQuit(t *testing.T) {
	eventCh := make(chan model.Event)
	userCh := make(chan string, 1)
	a := New(eventCh, userCh, "v0.0.0", ".", "", "openrouter", "deepseek", 24000)
	a.width = 120
	a.height = 40
	a.input = a.input.SetValue("pending text")
	a.resizeViewport()

	m, cmd := a.handleKey(tea.KeyMsg{Type: tea.KeyCtrlC})
	if cmd != nil {
		t.Fatalf("first ctrl+c should not quit")
	}

	updated, ok := m.(App)
	if !ok {
		t.Fatalf("expected App model")
	}
	if got := updated.input.Value(); got != "" {
		t.Fatalf("first ctrl+c should clear input, got %q", got)
	}
	if !updated.ctrlCArmed {
		t.Fatalf("first ctrl+c should arm exit")
	}

	select {
	case sig := <-userCh:
		if sig != InterruptSignal {
			t.Fatalf("unexpected signal %q", sig)
		}
	default:
		t.Fatalf("expected interrupt signal to backend")
	}

	_, cmd = updated.handleKey(tea.KeyMsg{Type: tea.KeyCtrlC})
	if cmd == nil {
		t.Fatalf("second ctrl+c should quit")
	}
	if _, ok := cmd().(tea.QuitMsg); !ok {
		t.Fatalf("expected tea.QuitMsg on second ctrl+c")
	}
}
