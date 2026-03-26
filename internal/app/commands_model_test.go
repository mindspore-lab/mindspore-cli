package app

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vigo999/ms-cli/configs"
	"github.com/vigo999/ms-cli/integrations/llm"
	"github.com/vigo999/ms-cli/ui/model"
)

func TestCmdModel_UnprefixedKeepsProvider(t *testing.T) {
	app := newModelCommandTestApp()
	app.Config.Model.Provider = "anthropic"
	app.Config.Model.Model = "claude-3-5-sonnet"

	app.cmdModel([]string{"claude-3-5-haiku"})

	drainUntilEventType(t, app, model.AgentThinking)
	drainUntilEventType(t, app, model.ModelUpdate)
	drainUntilEventType(t, app, model.AgentReply)

	if got, want := app.Config.Model.Provider, "anthropic"; got != want {
		t.Fatalf("provider = %q, want %q", got, want)
	}
	if got, want := app.Config.Model.Model, "claude-3-5-haiku"; got != want {
		t.Fatalf("model = %q, want %q", got, want)
	}
}

func TestCmdModel_PrefixedUpdatesProviderAndModel(t *testing.T) {
	app := newModelCommandTestApp()
	app.Config.Model.Provider = "openai-completion"
	app.Config.Model.Model = "gpt-4o-mini"

	app.cmdModel([]string{"anthropic:claude-3-5-sonnet"})

	drainUntilEventType(t, app, model.AgentThinking)
	drainUntilEventType(t, app, model.ModelUpdate)
	drainUntilEventType(t, app, model.AgentReply)

	if got, want := app.Config.Model.Provider, "anthropic"; got != want {
		t.Fatalf("provider = %q, want %q", got, want)
	}
	if got, want := app.Config.Model.Model, "claude-3-5-sonnet"; got != want {
		t.Fatalf("model = %q, want %q", got, want)
	}
}

func TestCmdModel_ModelUpdateCarriesContextWindow(t *testing.T) {
	app := newModelCommandTestApp()
	app.Config.Context.Window = 200000

	app.cmdModel([]string{"gpt-4o"})

	drainUntilEventType(t, app, model.AgentThinking)
	ev := drainUntilEventType(t, app, model.ModelUpdate)

	if got, want := ev.CtxMax, 200000; got != want {
		t.Fatalf("model update ctx max = %d, want %d", got, want)
	}
}

func TestCmdModel_InvalidPrefixNoMutation(t *testing.T) {
	app := newModelCommandTestApp()
	app.Config.Model.Provider = "openai-completion"
	app.Config.Model.Model = "gpt-4o-mini"

	app.cmdModel([]string{"invalid:gpt-4o"})

	ev := drainUntilEventType(t, app, model.AgentReply)
	if !strings.Contains(ev.Message, "Unsupported provider prefix") {
		t.Fatalf("unexpected message: %q", ev.Message)
	}

	if got, want := app.Config.Model.Provider, "openai-completion"; got != want {
		t.Fatalf("provider = %q, want %q", got, want)
	}
	if got, want := app.Config.Model.Model, "gpt-4o-mini"; got != want {
		t.Fatalf("model = %q, want %q", got, want)
	}
}

func TestCmdModel_NoArgsShowsBuiltinPresetCandidate(t *testing.T) {
	app := newModelCommandTestApp()

	app.cmdModel(nil)

	ev := drainUntilEventType(t, app, model.ModelPickerOpen)
	if ev.Popup == nil {
		t.Fatal("ModelPickerOpen popup = nil, want popup")
	}
	if !strings.Contains(ev.Popup.Title, "Model Selection") {
		t.Fatalf("popup title = %q, want Model Selection header", ev.Popup.Title)
	}
	if len(ev.Popup.Options) == 0 || ev.Popup.Options[0].ID != "kimi-k2.5-free" {
		t.Fatalf("popup options = %#v, want kimi preset option", ev.Popup.Options)
	}
}

func TestCmdModel_BuiltinPresetRequiresLogin(t *testing.T) {
	app := newModelCommandTestApp()
	t.Setenv("HOME", t.TempDir())

	app.cmdModel([]string{"kimi-k2.5-free"})

	drainUntilEventType(t, app, model.AgentThinking)
	ev := drainUntilEventType(t, app, model.ToolError)
	if !strings.Contains(ev.Message, "not logged in") {
		t.Fatalf("tool error = %q, want login requirement", ev.Message)
	}
}

func TestCmdModel_BuiltinPresetUsesServerCredentialAndRestoresOnSwitchBack(t *testing.T) {
	app := newModelCommandTestApp()
	app.Config.Model.Provider = "openai-completion"
	app.Config.Model.URL = "https://api.openai.com/v1"
	app.Config.Model.Model = "gpt-4o-mini"
	app.Config.Model.Key = "env-key"

	var capturedAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedAuth = r.Header.Get("Authorization")
		if r.URL.Path != "/model-presets/kimi-k2.5-free/credential" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]string{"api_key": "server-kimi-key"})
	}))
	t.Cleanup(srv.Close)

	t.Setenv("HOME", t.TempDir())
	cred := credentials{
		ServerURL: srv.URL,
		Token:     "user-token",
		User:      "alice",
		Role:      "user",
	}
	if err := saveCredentials(&cred); err != nil {
		t.Fatalf("saveCredentials() error = %v", err)
	}

	var resolved llm.ResolvedConfig
	origBuildProvider := buildProvider
	buildProvider = func(cfg llm.ResolvedConfig) (llm.Provider, error) {
		resolved = cfg
		return &blockingStreamProvider{started: make(chan struct{})}, nil
	}
	defer func() { buildProvider = origBuildProvider }()

	app.cmdModel([]string{"kimi-k2.5-free"})
	drainUntilEventType(t, app, model.AgentThinking)
	drainUntilEventType(t, app, model.ModelUpdate)
	drainUntilEventType(t, app, model.AgentReply)

	if got, want := capturedAuth, "Bearer user-token"; got != want {
		t.Fatalf("credential request auth = %q, want %q", got, want)
	}
	if got, want := string(resolved.Kind), "anthropic"; got != want {
		t.Fatalf("resolved provider = %q, want %q", got, want)
	}
	if got, want := resolved.BaseURL, "https://api.kimi.com/coding/"; got != want {
		t.Fatalf("resolved base url = %q, want %q", got, want)
	}
	if got, want := resolved.Model, "kimi-k2.5"; got != want {
		t.Fatalf("resolved model = %q, want %q", got, want)
	}
	if got, want := resolved.APIKey, "server-kimi-key"; got != want {
		t.Fatalf("resolved key = %q, want %q", got, want)
	}

	app.cmdModel([]string{"gpt-4o"})
	drainUntilEventType(t, app, model.AgentThinking)
	drainUntilEventType(t, app, model.ModelUpdate)
	drainUntilEventType(t, app, model.AgentReply)

	if got, want := app.Config.Model.Provider, "openai-completion"; got != want {
		t.Fatalf("provider after restore = %q, want %q", got, want)
	}
	if got, want := app.Config.Model.URL, "https://api.openai.com/v1"; got != want {
		t.Fatalf("url after restore = %q, want %q", got, want)
	}
	if got, want := app.Config.Model.Key, "env-key"; got != want {
		t.Fatalf("key after restore = %q, want %q", got, want)
	}
	if got, want := app.Config.Model.Model, "gpt-4o"; got != want {
		t.Fatalf("model after switch = %q, want %q", got, want)
	}
}

func newModelCommandTestApp() *Application {
	cfg := configs.DefaultConfig()
	cfg.Model.Key = "test-key"
	cfg.Issues.ServerURL = "https://issues.example"
	return &Application{
		EventCh: make(chan model.Event, 16),
		Config:  cfg,
	}
}

func drainUntilEventType(t *testing.T, app *Application, target model.EventType) model.Event {
	t.Helper()
	timer := time.NewTimer(2 * time.Second)
	defer timer.Stop()

	for {
		select {
		case ev := <-app.EventCh:
			if ev.Type == target {
				return ev
			}
		case <-timer.C:
			t.Fatalf("timed out waiting for event type %s", target)
		}
	}
}
