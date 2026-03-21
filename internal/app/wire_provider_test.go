package app

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/vigo999/ms-cli/configs"
	"github.com/vigo999/ms-cli/integrations/llm"
	providerpkg "github.com/vigo999/ms-cli/integrations/llm/provider"
	"github.com/vigo999/ms-cli/permission"
	"github.com/vigo999/ms-cli/ui/model"
)

func TestInitProviderAnthropic(t *testing.T) {
	provider, err := initProvider(configs.ModelConfig{
		Provider: "anthropic",
		Model:    "claude-3-5-sonnet",
		Key:      "anthropic-token",
	}, providerResolveNoOverrides())
	if err != nil {
		t.Fatalf("initProvider() error = %v", err)
	}
	if provider == nil {
		t.Fatal("initProvider() provider = nil, want provider")
	}
	if got, want := provider.Name(), "anthropic"; got != want {
		t.Fatalf("provider.Name() = %q, want %q", got, want)
	}
}

func TestInitProviderOpenAICompatibleDefault(t *testing.T) {
	provider, err := initProvider(configs.ModelConfig{Model: "gpt-4o-mini", Key: "mscli-token"}, providerResolveNoOverrides())
	if err != nil {
		t.Fatalf("initProvider() error = %v", err)
	}
	if provider == nil {
		t.Fatal("initProvider() provider = nil, want provider")
	}
	if got, want := provider.Name(), "openai-compatible"; got != want {
		t.Fatalf("provider.Name() = %q, want %q", got, want)
	}
}

func TestInitProviderMapsMissingKeyToAppSentinel(t *testing.T) {
	_, err := initProvider(configs.ModelConfig{Model: "gpt-4o-mini"}, providerResolveNoOverrides())
	if err == nil {
		t.Fatal("initProvider() error = nil, want missing api key error")
	}
	if !errors.Is(err, errAPIKeyNotFound) {
		t.Fatalf("initProvider() error = %v, want errAPIKeyNotFound", err)
	}
}

func TestWireBootstrapKeyAndURLOverrideEnvDuringProviderInit(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	t.Setenv("MSCLI_PROVIDER", "openai-compatible")
	t.Setenv("MSCLI_API_KEY", "env-key")
	t.Setenv("MSCLI_BASE_URL", "http://127.0.0.1:1")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OPENAI_BASE_URL", "")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "")
	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("ANTHROPIC_BASE_URL", "")

	tempDir := t.TempDir()
	t.Chdir(tempDir)

	defaultCfg := configs.DefaultConfig()
	defaultCfg.Model.Provider = "openai-compatible"
	defaultCfg.Model.Model = "gpt-4o-mini"
	defaultCfg.Model.Key = ""
	defaultCfg.Model.URL = "https://api.openai.com/v1"
	configPath := filepath.Join(tempDir, ".ms-cli", "config.yaml")
	if err := os.MkdirAll(filepath.Dir(configPath), 0755); err != nil {
		t.Fatalf("mkdir config dir: %v", err)
	}
	if err := configs.SaveToFile(defaultCfg, configPath); err != nil {
		t.Fatalf("SaveToFile() error = %v", err)
	}

	var gotAuth string
	var gotPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "cmpl-test",
			"choices": []map[string]any{
				{
					"message": map[string]any{
						"role":    "assistant",
						"content": "ok",
					},
					"finish_reason": "stop",
				},
			},
		})
	}))
	defer server.Close()

	app, err := Wire(BootstrapConfig{
		URL:   server.URL + "/v1",
		Key:   "cli-key",
		Model: "gpt-4o-mini",
	})
	if err != nil {
		t.Fatalf("Wire() error = %v", err)
	}

	_, err = app.provider.Complete(context.Background(), &llm.CompletionRequest{
		Messages: []llm.Message{
			{Role: "user", Content: "ping"},
		},
	})
	if err != nil {
		t.Fatalf("provider.Complete() error = %v", err)
	}

	if got, want := gotPath, "/v1/chat/completions"; got != want {
		t.Fatalf("request path = %q, want %q", got, want)
	}
	if got, want := gotAuth, "Bearer cli-key"; got != want {
		t.Fatalf("Authorization header = %q, want %q", got, want)
	}
}

func providerResolveNoOverrides() providerpkg.ResolveOptions {
	return providerpkg.ResolveOptions{}
}

func TestWire_ConfiguresPermissionPromptUI(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("MSCLI_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "")
	t.Setenv("ANTHROPIC_API_KEY", "")

	tempDir := t.TempDir()
	t.Chdir(tempDir)

	app, err := Wire(BootstrapConfig{})
	if err != nil {
		t.Fatalf("Wire() error = %v", err)
	}

	if app.permissionUI == nil {
		t.Fatal("permissionUI = nil, want initialized prompt UI")
	}

	permSvc, ok := app.permService.(*permission.DefaultPermissionService)
	if !ok {
		t.Fatalf("permService type = %T, want *permission.DefaultPermissionService", app.permService)
	}

	done := make(chan struct {
		granted bool
		err     error
	}, 1)
	go func() {
		granted, err := permSvc.Request(context.Background(), "write", "", "tmp.txt")
		done <- struct {
			granted bool
			err     error
		}{granted: granted, err: err}
	}()

	select {
	case ev := <-app.EventCh:
		if ev.Type != model.PermissionPrompt {
			t.Fatalf("event type = %s, want %s", ev.Type, model.PermissionPrompt)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for permission prompt event")
	}

	app.processInput("yes")

	select {
	case out := <-done:
		if out.err != nil {
			t.Fatalf("Request() err = %v", out.err)
		}
		if !out.granted {
			t.Fatal("Request() granted = false, want true")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for permission request completion")
	}
}
