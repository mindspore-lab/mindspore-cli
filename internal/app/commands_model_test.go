package app

import (
	"strings"
	"testing"
	"time"

	"github.com/vigo999/ms-cli/configs"
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
	app.Config.Model.Provider = "openai-compatible"
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

func TestCmdModel_InvalidPrefixNoMutation(t *testing.T) {
	app := newModelCommandTestApp()
	app.Config.Model.Provider = "openai-compatible"
	app.Config.Model.Model = "gpt-4o-mini"

	app.cmdModel([]string{"invalid:gpt-4o"})

	ev := drainUntilEventType(t, app, model.AgentReply)
	if !strings.Contains(ev.Message, "Unsupported provider prefix") {
		t.Fatalf("unexpected message: %q", ev.Message)
	}

	if got, want := app.Config.Model.Provider, "openai-compatible"; got != want {
		t.Fatalf("provider = %q, want %q", got, want)
	}
	if got, want := app.Config.Model.Model, "gpt-4o-mini"; got != want {
		t.Fatalf("model = %q, want %q", got, want)
	}
}

func newModelCommandTestApp() *Application {
	cfg := configs.DefaultConfig()
	cfg.Model.Key = "test-key"
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
