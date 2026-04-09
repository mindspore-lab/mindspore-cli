package ui

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

func TestModelBrowserDefaultsToModelPaneAndExpandsProvidersOnLeft(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false
	next, _ := app.Update(tea.WindowSizeMsg{Width: 120, Height: 30})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type: model.ModelBrowserOpen,
		ModelBrowser: &model.ModelBrowserPopup{
			Providers: model.SelectionPopup{
				Title: "Providers",
				Options: []model.SelectionOption{
					{ID: "openrouter", Label: "OpenRouter", RequiresInput: true},
				},
				Selected: 0,
			},
			Models: model.SelectionPopup{
				Title: "Models",
				Options: []model.SelectionOption{
					{ID: "openrouter:openai/gpt-4o-mini", Label: "GPT-4o mini"},
				},
				Selected: 0,
			},
			Focus:            model.ModelBrowserFocusModel,
			ProvidersVisible: false,
		},
	})
	app = next.(App)

	if app.modelBrowser == nil {
		t.Fatal("expected model browser to open")
	}
	if view := app.View(); !strings.Contains(view, "Providers") {
		t.Fatalf("expected provider hint in view, got:\n%s", view)
	}

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyLeft})
	app = next.(App)

	if !app.modelBrowser.ProvidersVisible {
		t.Fatal("expected left key to expand providers pane")
	}
	if got, want := app.modelBrowser.Focus, model.ModelBrowserFocusProvider; got != want {
		t.Fatalf("focus = %v, want %v", got, want)
	}
}

func TestModelBrowserProviderInputAndModelSelectionSendInternalTokens(t *testing.T) {
	userCh := make(chan string, 2)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false
	next, _ := app.Update(tea.WindowSizeMsg{Width: 120, Height: 30})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type: model.ModelBrowserOpen,
		ModelBrowser: &model.ModelBrowserPopup{
			Providers: model.SelectionPopup{
				Title: "Providers",
				Options: []model.SelectionOption{
					{ID: "openrouter", Label: "OpenRouter", RequiresInput: true},
				},
				Selected: 0,
			},
			Models: model.SelectionPopup{
				Title: "Models",
				Options: []model.SelectionOption{
					{ID: "openrouter:openai/gpt-4o-mini", Label: "GPT-4o mini"},
				},
				Selected: 0,
			},
			Focus:            model.ModelBrowserFocusProvider,
			ProvidersVisible: true,
		},
	})
	app = next.(App)

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)
	if app.modelBrowser.ProviderInput == nil {
		t.Fatal("expected provider input to open")
	}

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("sk-test")})
	app = next.(App)
	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)

	select {
	case got := <-userCh:
		if got != connectProviderInputToken+" openrouter sk-test" {
			t.Fatalf("provider input token = %q", got)
		}
	default:
		t.Fatal("expected provider connect token")
	}

	app.modelBrowser.ProviderInput = nil
	app.modelBrowser.Focus = model.ModelBrowserFocusModel
	app.modelBrowser.ProvidersVisible = false
	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)

	select {
	case got := <-userCh:
		if got != selectModelInputToken+" openrouter:openai/gpt-4o-mini" {
			t.Fatalf("model select token = %q", got)
		}
	default:
		t.Fatal("expected model select token")
	}
}

func TestModelBrowserProviderSelectionWithoutInputSendsConnectToken(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false
	next, _ := app.Update(tea.WindowSizeMsg{Width: 120, Height: 30})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type: model.ModelBrowserOpen,
		ModelBrowser: &model.ModelBrowserPopup{
			Providers: model.SelectionPopup{
				Title: "Providers",
				Options: []model.SelectionOption{
					{ID: "__header__detected", Label: "Import", Header: true, Disabled: true},
					{ID: "__import_provider__:kimi-for-coding", Label: "Kimi For Coding"},
					{ID: "__detail__kimi-for-coding__source", Label: "from Claude Code environment detected:", Disabled: true, DetailRow: true},
				},
				Selected: 1,
			},
			Models: model.SelectionPopup{
				Title:    "Models",
				Options:  []model.SelectionOption{},
				Selected: 0,
			},
			Focus:            model.ModelBrowserFocusProvider,
			ProvidersVisible: true,
		},
	})
	app = next.(App)

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEnter})
	app = next.(App)

	select {
	case got := <-userCh:
		if got != connectProviderInputToken+" __import_provider__:kimi-for-coding" {
			t.Fatalf("provider connect token = %q", got)
		}
	default:
		t.Fatal("expected provider connect token")
	}
}

func TestModelBrowserFocusSwitchStartsAndSettlesAnimation(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false
	next, _ := app.Update(tea.WindowSizeMsg{Width: 120, Height: 30})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type: model.ModelBrowserOpen,
		ModelBrowser: &model.ModelBrowserPopup{
			Providers: model.SelectionPopup{
				Options: []model.SelectionOption{
					{ID: "openrouter", Label: "OpenRouter", RequiresInput: true},
				},
				Selected: 0,
			},
			Models: model.SelectionPopup{
				Options: []model.SelectionOption{
					{ID: "openrouter:openai/gpt-4o-mini", Label: "GPT-4o mini"},
				},
				Selected: 0,
			},
			Focus:            model.ModelBrowserFocusModel,
			ProvidersVisible: false,
		},
	})
	app = next.(App)

	next, cmd := app.handleKey(tea.KeyMsg{Type: tea.KeyLeft})
	app = next.(App)
	if cmd == nil {
		t.Fatal("expected animation tick command on focus switch")
	}
	if got := app.modelBrowser.AnimationOffset; got == 0 {
		t.Fatal("expected non-zero animation offset after focus switch")
	}

	next, _ = app.Update(modelBrowserAnimTickMsg{})
	app = next.(App)
	next, _ = app.Update(modelBrowserAnimTickMsg{})
	app = next.(App)

	if got, want := app.modelBrowser.AnimationOffset, 0; got != want {
		t.Fatalf("animation offset = %d, want %d", got, want)
	}
}

func TestModelBrowserDoublePressDDeletesSelectedProviderRow(t *testing.T) {
	userCh := make(chan string, 1)
	app := New(nil, userCh, "test", ".", "", "demo-model", 4096)
	app.bootActive = false
	next, _ := app.Update(tea.WindowSizeMsg{Width: 120, Height: 30})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type: model.ModelBrowserOpen,
		ModelBrowser: &model.ModelBrowserPopup{
			Models: model.SelectionPopup{
				Title: "Models",
				Options: []model.SelectionOption{
					{ID: "__provider__openrouter", Label: "OpenRouter", ProviderRow: true, DeleteProviderID: "openrouter"},
					{ID: "openrouter:openai/gpt-4o-mini", Label: "GPT-4o mini"},
				},
				Selected: 0,
			},
			Focus:            model.ModelBrowserFocusModel,
			ProvidersVisible: false,
		},
	})
	app = next.(App)

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("d")})
	app = next.(App)
	if got, want := app.modelBrowser.PendingDeleteProviderID, "openrouter"; got != want {
		t.Fatalf("pending delete provider = %q, want %q", got, want)
	}

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("d")})
	app = next.(App)

	select {
	case got := <-userCh:
		if got != deleteProviderInputToken+" openrouter" {
			t.Fatalf("delete provider token = %q", got)
		}
	default:
		t.Fatal("expected delete provider token")
	}
}

func TestModelBrowserMovingSelectionClearsPendingDelete(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false
	next, _ := app.Update(tea.WindowSizeMsg{Width: 120, Height: 30})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type: model.ModelBrowserOpen,
		ModelBrowser: &model.ModelBrowserPopup{
			Models: model.SelectionPopup{
				Title: "Models",
				Options: []model.SelectionOption{
					{ID: "__provider__openrouter", Label: "OpenRouter", ProviderRow: true, DeleteProviderID: "openrouter"},
					{ID: "openrouter:openai/gpt-4o-mini", Label: "GPT-4o mini"},
				},
				Selected: 0,
			},
			Focus:            model.ModelBrowserFocusModel,
			ProvidersVisible: false,
		},
	})
	app = next.(App)

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("d")})
	app = next.(App)
	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyDown})
	app = next.(App)

	if got, want := app.modelBrowser.PendingDeleteProviderID, ""; got != want {
		t.Fatalf("pending delete provider = %q, want empty", got)
	}
}

func TestModelBrowserEscapeClearsWorkingIndicator(t *testing.T) {
	app := New(nil, nil, "test", ".", "", "demo-model", 4096)
	app.bootActive = false
	next, _ := app.Update(tea.WindowSizeMsg{Width: 120, Height: 30})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{Type: model.AgentThinking})
	app = next.(App)

	next, _ = app.handleEvent(model.Event{
		Type: model.ModelBrowserOpen,
		ModelBrowser: &model.ModelBrowserPopup{
			Models: model.SelectionPopup{
				Title: "Models",
				Options: []model.SelectionOption{
					{ID: "openrouter:openai/gpt-4o-mini", Label: "GPT-4o mini"},
				},
				Selected: 0,
			},
			Focus:            model.ModelBrowserFocusModel,
			ProvidersVisible: false,
		},
	})
	app = next.(App)

	next, _ = app.handleKey(tea.KeyMsg{Type: tea.KeyEscape})
	app = next.(App)

	if app.modelBrowser != nil {
		t.Fatal("expected model browser to close on esc")
	}
	if app.state.IsThinking {
		t.Fatal("expected esc close to clear thinking state")
	}
	if got, want := app.state.WaitKind, model.WaitNone; got != want {
		t.Fatalf("wait kind = %v, want %v", got, want)
	}
	if view := app.View(); strings.Contains(view, "Working...") {
		t.Fatalf("expected working indicator to stay cleared after esc, got:\n%s", view)
	}
}
