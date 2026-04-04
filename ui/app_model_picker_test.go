package ui

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

func TestSetupPopupOpenAndNavigate(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false
	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	app = next.(App)

	// Open setup popup via event
	next, _ = app.handleEvent(model.Event{
		Type: model.ModelSetupOpen,
		SetupPopup: &model.SetupPopup{
			Screen: model.SetupScreenModeSelect,
			PresetOptions: []model.SelectionOption{
				{ID: "kimi-k2.5-free", Label: "kimi-k2.5 [free]"},
				{ID: "deepseek-v3", Label: "deepseek-v3"},
				{ID: "glm-4.7", Label: "glm-4.7 (coming soon)", Disabled: true},
			},
			CanEscape: true,
		},
	})
	app = next.(App)

	if app.setupPopup == nil {
		t.Fatal("expected setup popup to be open")
	}

	view := app.View()
	if !strings.Contains(view, "mscli-provided") {
		t.Fatalf("expected mode select screen in view, got:\n%s", view)
	}

	// Press enter to go to preset picker (mode 0 = mscli-provided)
	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)
	if app.setupPopup.Screen != model.SetupScreenPresetPicker {
		t.Fatalf("expected preset picker screen, got %d", app.setupPopup.Screen)
	}

	// Press esc to go back to mode select
	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEscape})
	app = next.(App)
	if app.setupPopup.Screen != model.SetupScreenModeSelect {
		t.Fatalf("expected mode select screen, got %d", app.setupPopup.Screen)
	}

	// Navigate to "your own model" and press enter
	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyDown})
	app = next.(App)
	if app.setupPopup.ModeSelected != 1 {
		t.Fatalf("expected mode 1, got %d", app.setupPopup.ModeSelected)
	}
	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)
	if app.setupPopup.Screen != model.SetupScreenEnvInfo {
		t.Fatalf("expected env info screen, got %d", app.setupPopup.Screen)
	}

	// Press esc to go back
	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEscape})
	app = next.(App)
	if app.setupPopup.Screen != model.SetupScreenModeSelect {
		t.Fatalf("expected mode select screen after esc from env info, got %d", app.setupPopup.Screen)
	}

	// Press esc again to close (CanEscape=true)
	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEscape})
	app = next.(App)
	if app.setupPopup != nil {
		t.Fatal("expected setup popup to close on esc from mode select")
	}
}

func TestSetupPopupNoEscapeOnFirstBoot(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "", 4096)
	app.bootActive = false

	next, _ := app.handleEvent(model.Event{
		Type: model.ModelSetupOpen,
		SetupPopup: &model.SetupPopup{
			Screen:    model.SetupScreenModeSelect,
			CanEscape: false,
		},
	})
	app = next.(App)

	// Esc should NOT close the popup
	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEscape})
	app = next.(App)
	if app.setupPopup == nil {
		t.Fatal("expected setup popup to stay open when CanEscape=false")
	}
}

func TestInlineModeSetupPopupUsesTemporaryFullscreenView(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false
	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	app = next.(App)

	next, cmd := app.handleEvent(model.Event{
		Type: model.ModelSetupOpen,
		SetupPopup: &model.SetupPopup{
			Screen:    model.SetupScreenModeSelect,
			CanEscape: true,
		},
	})
	app = next.(App)

	if cmd == nil {
		t.Fatal("expected inline mode setup popup to request temporary alt-screen")
	}
	if !app.modalAltScreen {
		t.Fatal("expected inline mode setup popup to mark temporary alt-screen active")
	}
	if view := app.View(); !strings.Contains(view, "mscli-provided") {
		t.Fatalf("expected inline setup popup to be visible, got:\n%s", view)
	}

	next, cmd = app.handleKey(tea.KeyMsg{Type: tea.KeyEscape})
	app = next.(App)

	if cmd == nil {
		t.Fatal("expected closing inline setup popup to request alt-screen exit")
	}
	if app.modalAltScreen {
		t.Fatal("expected temporary alt-screen flag to clear after popup close")
	}
}

func TestModelSetupPopupSuppressesThinkingIndicatorWithoutClearingState(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{Type: model.AgentThinking})
	app = next.(App)
	if view := app.View(); !strings.Contains(view, "Working...") {
		t.Fatalf("expected thinking indicator before popup, got:\n%s", view)
	}

	next, _ = app.handleEvent(model.Event{
		Type: model.ModelSetupOpen,
		SetupPopup: &model.SetupPopup{
			Screen:    model.SetupScreenModeSelect,
			CanEscape: true,
		},
	})
	app = next.(App)

	if !app.state.IsThinking {
		t.Fatal("expected popup open to preserve underlying thinking state")
	}
	view := app.View()
	if !strings.Contains(view, "mscli-provided") {
		t.Fatalf("expected model setup popup in view, got:\n%s", view)
	}
	if strings.Contains(view, "Working...") {
		t.Fatalf("expected popup view to suppress background thinking indicator, got:\n%s", view)
	}

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEscape})
	app = next.(App)
	if view := app.View(); !strings.Contains(view, "Working...") {
		t.Fatalf("expected thinking indicator to return after popup close, got:\n%s", view)
	}
}
