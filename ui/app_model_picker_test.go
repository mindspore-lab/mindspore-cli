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

func TestSetupPopupPresetPickerSupportsSearch(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false
	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type: model.ModelSetupOpen,
		SetupPopup: &model.SetupPopup{
			Title:  "Connect Provider",
			Screen: model.SetupScreenPresetPicker,
			PresetOptions: []model.SelectionOption{
				{ID: "__header__popular", Label: "Popular", Header: true, Disabled: true},
				{ID: "anthropic", Label: "Anthropic"},
				{ID: "openai", Label: "OpenAI"},
			},
			PresetSelected: 1,
			CanEscape:      true,
		},
	})
	app = next.(App)

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("open")})
	app = next.(App)

	if got, want := app.setupPopup.SearchQuery, "open"; got != want {
		t.Fatalf("SearchQuery = %q, want %q", got, want)
	}
	view := app.View()
	if !strings.Contains(view, "OpenAI") {
		t.Fatalf("expected filtered match in view, got:\n%s", view)
	}
	if strings.Contains(view, "Anthropic") {
		t.Fatalf("expected non-matching option filtered out, got:\n%s", view)
	}
}

func TestSetupPopupSearchClearRestoresPreviousSelection(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false
	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type: model.ModelSetupOpen,
		SetupPopup: &model.SetupPopup{
			Title:  "Connect Provider",
			Screen: model.SetupScreenPresetPicker,
			PresetOptions: []model.SelectionOption{
				{ID: "__header__popular", Label: "Popular", Header: true, Disabled: true},
				{ID: "anthropic", Label: "Anthropic"},
				{ID: "openai", Label: "OpenAI"},
			},
			PresetSelected: 1,
			CanEscape:      true,
		},
	})
	app = next.(App)

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("open")})
	app = next.(App)
	if got, want := app.setupPopup.PresetSelected, 2; got != want {
		t.Fatalf("PresetSelected after search = %d, want %d", got, want)
	}

	for range 4 {
		next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyBackspace})
		app = next.(App)
	}

	if got, want := app.setupPopup.SearchQuery, ""; got != want {
		t.Fatalf("SearchQuery after clear = %q, want %q", got, want)
	}
	if got, want := app.setupPopup.PresetSelected, 1; got != want {
		t.Fatalf("PresetSelected after clear = %d, want %d", got, want)
	}
}

func TestConnectPopupEscapeClosesInsteadOfShowingModeSelect(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false
	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type: model.ModelSetupOpen,
		SetupPopup: &model.SetupPopup{
			Title:  "Connect Provider",
			Screen: model.SetupScreenPresetPicker,
			PresetOptions: []model.SelectionOption{
				{ID: "__header__popular", Label: "Popular", Header: true, Disabled: true},
				{ID: "anthropic", Label: "Anthropic"},
			},
			PresetSelected: 1,
			BackCloses:     true,
			CanEscape:      true,
		},
	})
	app = next.(App)

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEscape})
	app = next.(App)

	if app.setupPopup != nil {
		t.Fatal("expected connect popup to close on esc")
	}
	if view := app.View(); strings.Contains(view, "mscli-provided model") {
		t.Fatalf("expected old mode-select screen not to appear, got:\n%s", view)
	}
}

func TestModelPickerSupportsSearch(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false
	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type: model.ModelPickerOpen,
		Popup: &model.SelectionPopup{
			Title: "Select model",
			Options: []model.SelectionOption{
				{ID: "__header__provider:free", Label: "MindSpore CLI Free", Header: true, Disabled: true},
				{ID: "mindspore-cli-free:kimi-k2.5", Label: "Kimi K2.5"},
				{ID: "mindspore-cli-free:deepseek-chat", Label: "DeepSeek V3"},
			},
			Selected: 1,
		},
	})
	app = next.(App)

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("deep")})
	app = next.(App)

	if got, want := app.modelPicker.SearchQuery, "deep"; got != want {
		t.Fatalf("SearchQuery = %q, want %q", got, want)
	}
	view := app.View()
	if !strings.Contains(view, "DeepSeek V3") {
		t.Fatalf("expected filtered match in view, got:\n%s", view)
	}
	if strings.Contains(view, "Kimi K2.5") {
		t.Fatalf("expected non-matching option filtered out, got:\n%s", view)
	}
}

func TestModelPickerSearchClearRestoresPreviousSelection(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false
	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type: model.ModelPickerOpen,
		Popup: &model.SelectionPopup{
			Title: "Select model",
			Options: []model.SelectionOption{
				{ID: "__header__provider:free", Label: "MindSpore CLI Free", Header: true, Disabled: true},
				{ID: "mindspore-cli-free:kimi-k2.5", Label: "Kimi K2.5"},
				{ID: "mindspore-cli-free:deepseek-chat", Label: "DeepSeek V3"},
			},
			Selected: 1,
		},
	})
	app = next.(App)

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("deep")})
	app = next.(App)
	if got, want := app.modelPicker.Selected, 2; got != want {
		t.Fatalf("Selected after search = %d, want %d", got, want)
	}

	for range 4 {
		next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyBackspace})
		app = next.(App)
	}

	if got, want := app.modelPicker.SearchQuery, ""; got != want {
		t.Fatalf("SearchQuery after clear = %q, want %q", got, want)
	}
	if got, want := app.modelPicker.Selected, 1; got != want {
		t.Fatalf("Selected after clear = %d, want %d", got, want)
	}
}

func TestModelPickerCtrlAOpensConnect(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false
	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type: model.ModelPickerOpen,
		Popup: &model.SelectionPopup{
			Title:    "Select model",
			ActionID: "model_picker",
			Options: []model.SelectionOption{
				{ID: "__header__Recent", Label: "Recent", Header: true, Disabled: true},
				{ID: "mindspore-cli-free:kimi-k2.5", Label: "Kimi K2.5"},
			},
			Selected: 1,
		},
	})
	app = next.(App)

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyCtrlA})
	app = next.(App)

	if app.modelPicker != nil {
		t.Fatal("expected model picker to close on ctrl+a")
	}
	select {
	case got := <-userCh:
		if got != "/connect" {
			t.Fatalf("user input = %q, want %q", got, "/connect")
		}
	default:
		t.Fatal("expected /connect shortcut to be sent")
	}
}

func TestProviderScopedModelPickerEnterSelectsModel(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false
	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type: model.ModelPickerOpen,
		Popup: &model.SelectionPopup{
			Title:    "Select model",
			ActionID: "connect_provider_model_picker:openrouter",
			Options: []model.SelectionOption{
				{ID: "openrouter:openai/gpt-4o-mini", Label: "GPT-4o mini"},
			},
			Selected: 0,
		},
	})
	app = next.(App)

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)

	if app.modelPicker != nil {
		t.Fatal("expected provider-scoped model picker to close on enter")
	}
	select {
	case got := <-userCh:
		if got != "/model openrouter:openai/gpt-4o-mini" {
			t.Fatalf("user input = %q, want %q", got, "/model openrouter:openai/gpt-4o-mini")
		}
	default:
		t.Fatal("expected selected model command to be sent")
	}
}
