package app

import (
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mindspore-lab/mindspore-cli/configs"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

func TestCmdModel_NoArgsShowsModelBrowser(t *testing.T) {
	app := newModelCommandTestApp()
	home := t.TempDir()
	t.Setenv("HOME", home)
	origCache := modelsDevCachePathOverride
	modelsDevCachePathOverride = filepath.Join(home, ".mscli", "cached", "models-dev-api.json")
	t.Cleanup(func() { modelsDevCachePathOverride = origCache })
	origURL := modelsDevAPIURL
	server := newModelsDevTestServer(`{
		"openrouter": {
			"id": "openrouter",
			"name": "OpenRouter",
			"api": "https://openrouter.ai/api/v1",
			"npm": "@openrouter/ai-sdk-provider",
			"models": {"openai/gpt-4o-mini": {"id": "openai/gpt-4o-mini", "name": "GPT-4o mini"}}
		}
	}`)
	defer server.Close()
	modelsDevAPIURL = server.URL
	t.Cleanup(func() { modelsDevAPIURL = origURL })
	resetModelsDevProviderCacheForTest()
	t.Cleanup(resetModelsDevProviderCacheForTest)
	if err := writeModelsDevCache([]byte(`{
		"openrouter": {
			"id": "openrouter",
			"name": "OpenRouter",
			"api": "https://openrouter.ai/api/v1",
			"npm": "@openrouter/ai-sdk-provider",
			"models": {"openai/gpt-4o-mini": {"id": "openai/gpt-4o-mini", "name": "GPT-4o mini"}}
		}
	}`)); err != nil {
		t.Fatalf("writeModelsDevCache() error = %v", err)
	}
	if err := saveProviderAuthState(&providerAuthState{
		Providers: map[string]providerAuthEntry{
			"openrouter": {ProviderID: "openrouter", APIKey: "sk-openrouter"},
		},
	}); err != nil {
		t.Fatalf("saveProviderAuthState() error = %v", err)
	}

	app.cmdModel(nil)

	ev := drainUntilEventType(t, app, model.ModelBrowserOpen)
	if ev.ModelBrowser == nil {
		t.Fatal("ModelBrowserOpen popup = nil, want ModelBrowserPopup")
	}
	if len(ev.ModelBrowser.Models.Options) == 0 {
		t.Fatal("expected model browser model options")
	}
	if len(ev.ModelBrowser.Providers.Options) == 0 {
		t.Fatal("expected model browser provider options")
	}
	if got, want := ev.ModelBrowser.Focus, model.ModelBrowserFocusModel; got != want {
		t.Fatalf("focus = %v, want %v", got, want)
	}
}

func TestCmdModel_NoArgsDoesNotAutoImportClaudeCodeProvider(t *testing.T) {
	app := newModelCommandTestApp()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("ANTHROPIC_BASE_URL", "https://api.kimi.com/coding/")
	t.Setenv("ANTHROPIC_API_KEY", "sk-kimi-test")

	origCache := modelsDevCachePathOverride
	modelsDevCachePathOverride = filepath.Join(home, ".mscli", "cached", "models-dev-api.json")
	t.Cleanup(func() { modelsDevCachePathOverride = origCache })
	origURL := modelsDevAPIURL
	server := newModelsDevTestServer(`{
		"kimi-for-coding": {
			"id": "kimi-for-coding",
			"name": "Kimi For Coding",
			"api": "https://api.kimi.com/coding/v1",
			"npm": "@ai-sdk/anthropic",
			"models": {
				"k2p5": {"id": "k2p5", "name": "Kimi K2.5"}
			}
		}
	}`)
	defer server.Close()
	modelsDevAPIURL = server.URL
	t.Cleanup(func() { modelsDevAPIURL = origURL })
	resetModelsDevProviderCacheForTest()
	t.Cleanup(resetModelsDevProviderCacheForTest)

	app.cmdModel(nil)

	ev := drainUntilEventType(t, app, model.ModelBrowserOpen)
	if ev.ModelBrowser == nil {
		t.Fatal("ModelBrowserOpen popup = nil, want ModelBrowserPopup")
	}
	if got := len(ev.ModelBrowser.ImportSuggestions); got != 0 {
		t.Fatalf("len(ev.ModelBrowser.ImportSuggestions) = %d, want 0", got)
	}
	for _, opt := range ev.ModelBrowser.Models.Options {
		if opt.ID == "__provider__kimi-for-coding" || opt.ID == "kimi-for-coding:k2p5" {
			t.Fatalf("unexpected auto-imported option %q", opt.ID)
		}
	}
}

func TestCmdModel_NoArgsUsesCachedCatalogWithoutBlocking(t *testing.T) {
	app := newModelCommandTestApp()
	home := t.TempDir()
	t.Setenv("HOME", home)
	origCache := modelsDevCachePathOverride
	modelsDevCachePathOverride = filepath.Join(home, ".mscli", "cached", "models-dev-api.json")
	t.Cleanup(func() { modelsDevCachePathOverride = origCache })
	origURL := modelsDevAPIURL
	server := newModelsDevTestServer(`{
		"openrouter": {
			"id": "openrouter",
			"name": "OpenRouter",
			"api": "https://openrouter.ai/api/v1",
			"npm": "@openrouter/ai-sdk-provider",
			"models": {}
		}
	}`)
	defer server.Close()
	modelsDevAPIURL = server.URL
	t.Cleanup(func() { modelsDevAPIURL = origURL })
	resetModelsDevProviderCacheForTest()
	t.Cleanup(resetModelsDevProviderCacheForTest)
	if err := writeModelsDevCache([]byte(`{
		"openrouter": {
			"id": "openrouter",
			"name": "OpenRouter",
			"api": "https://openrouter.ai/api/v1",
			"npm": "@openrouter/ai-sdk-provider",
			"models": {}
		}
	}`)); err != nil {
		t.Fatalf("writeModelsDevCache() error = %v", err)
	}

	start := time.Now()
	app.cmdModel(nil)
	if elapsed := time.Since(start); elapsed >= 500*time.Millisecond {
		t.Fatalf("cmdModel(nil) took %v, want under 500ms", elapsed)
	}
	drainUntilEventType(t, app, model.ModelBrowserOpen)
}

func TestCmdModel_NoArgsHidesFreeProviderWhenLoggedOut(t *testing.T) {
	app := newModelCommandTestApp()
	home := t.TempDir()
	t.Setenv("HOME", home)
	origCache := modelsDevCachePathOverride
	modelsDevCachePathOverride = filepath.Join(home, ".mscli", "cached", "models-dev-api.json")
	t.Cleanup(func() { modelsDevCachePathOverride = origCache })
	origURL := modelsDevAPIURL
	server := newModelsDevTestServer(`{
		"anthropic": {"id": "anthropic", "name": "Anthropic", "npm": "@ai-sdk/anthropic", "models": {}}
	}`)
	defer server.Close()
	modelsDevAPIURL = server.URL
	t.Cleanup(func() { modelsDevAPIURL = origURL })
	resetModelsDevProviderCacheForTest()
	t.Cleanup(resetModelsDevProviderCacheForTest)
	if err := writeModelsDevCache([]byte(`{
		"anthropic": {"id": "anthropic", "name": "Anthropic", "npm": "@ai-sdk/anthropic", "models": {}}
	}`)); err != nil {
		t.Fatalf("writeModelsDevCache() error = %v", err)
	}

	app.cmdModel(nil)

	ev := drainUntilEventType(t, app, model.ModelBrowserOpen)
	if ev.ModelBrowser == nil {
		t.Fatal("ModelBrowserOpen popup = nil, want ModelBrowserPopup")
	}
	for _, opt := range ev.ModelBrowser.Providers.Options {
		if opt.ID == mindsporeCLIFreeProviderID {
			t.Fatal("expected free provider to be hidden when logged out")
		}
	}
}

func TestCmdModel_WithArgsRejected(t *testing.T) {
	app := newModelCommandTestApp()

	app.cmdModel([]string{"openrouter:openai/gpt-4o-mini"})

	ev := drainUntilEventType(t, app, model.AgentReply)
	if !strings.Contains(ev.Message, "no longer accepts arguments") {
		t.Fatalf("message = %q, want argument rejection", ev.Message)
	}
}

func TestCmdConnect_WithAPIKeyPersistsAuthAndRefreshesModelBrowser(t *testing.T) {
	app := newModelCommandTestApp()
	home := t.TempDir()
	t.Setenv("HOME", home)
	origAuthPath := authStatePathOverride
	authStatePathOverride = filepath.Join(home, ".mscli", "auth.json")
	t.Cleanup(func() { authStatePathOverride = origAuthPath })
	origCache := modelsDevCachePathOverride
	modelsDevCachePathOverride = filepath.Join(home, ".mscli", "cached", "models-dev-api.json")
	t.Cleanup(func() { modelsDevCachePathOverride = origCache })
	origURL := modelsDevAPIURL
	server := newModelsDevTestServer(`{
		"openrouter": {
			"id": "openrouter",
			"name": "OpenRouter",
			"api": "https://openrouter.ai/api/v1",
			"npm": "@openrouter/ai-sdk-provider",
			"models": {"openai/gpt-4o-mini": {"id": "openai/gpt-4o-mini", "name": "GPT-4o mini"}}
		}
	}`)
	defer server.Close()
	modelsDevAPIURL = server.URL
	t.Cleanup(func() { modelsDevAPIURL = origURL })
	resetModelsDevProviderCacheForTest()
	t.Cleanup(resetModelsDevProviderCacheForTest)

	app.cmdConnect([]string{"openrouter", "sk-openrouter"})

	drainUntilEventType(t, app, model.AgentThinking)
	ev := drainUntilEventType(t, app, model.ModelBrowserOpen)
	if ev.ModelBrowser == nil {
		t.Fatal("expected refreshed model browser")
	}
	if got, want := ev.ModelBrowser.Focus, model.ModelBrowserFocusModel; got != want {
		t.Fatalf("focus = %v, want %v", got, want)
	}
	if got, want := ev.ModelBrowser.Models.Options[1].ID, "openrouter:openai/gpt-4o-mini"; got != want {
		t.Fatalf("first model option = %q, want %q", got, want)
	}
	authState, err := loadProviderAuthState()
	if err != nil {
		t.Fatalf("loadProviderAuthState() error = %v", err)
	}
	entry, ok := authState.Providers["openrouter"]
	if !ok {
		t.Fatalf("authState.Providers = %#v, want openrouter entry", authState.Providers)
	}
	if got, want := entry.APIKey, "sk-openrouter"; got != want {
		t.Fatalf("entry.APIKey = %q, want %q", got, want)
	}
}

func TestCmdSelectModel_LogicalProviderSelectionWorksWithoutCache(t *testing.T) {
	app := newModelCommandTestApp()
	home := t.TempDir()
	t.Setenv("HOME", home)

	origAuthPath := authStatePathOverride
	authStatePathOverride = filepath.Join(home, ".mscli", "auth.json")
	t.Cleanup(func() { authStatePathOverride = origAuthPath })
	origModelPath := modelStatePathOverride
	modelStatePathOverride = filepath.Join(home, ".mscli", "model.json")
	t.Cleanup(func() { modelStatePathOverride = origModelPath })
	origCache := modelsDevCachePathOverride
	modelsDevCachePathOverride = filepath.Join(home, ".mscli", "cached", "models-dev-api.json")
	t.Cleanup(func() { modelsDevCachePathOverride = origCache })
	origURL := modelsDevAPIURL
	server := newModelsDevTestServer(`{
		"openrouter": {
			"id": "openrouter",
			"name": "OpenRouter",
			"api": "https://openrouter.ai/api/v1",
			"npm": "@openrouter/ai-sdk-provider",
			"models": {"openai/gpt-4o-mini": {"id": "openai/gpt-4o-mini", "name": "GPT-4o mini"}}
		}
	}`)
	defer server.Close()
	modelsDevAPIURL = server.URL
	t.Cleanup(func() { modelsDevAPIURL = origURL })

	resetModelsDevProviderCacheForTest()
	t.Cleanup(resetModelsDevProviderCacheForTest)

	if err := saveProviderAuthState(&providerAuthState{
		Providers: map[string]providerAuthEntry{
			"openrouter": {ProviderID: "openrouter", APIKey: "sk-openrouter"},
		},
	}); err != nil {
		t.Fatalf("saveProviderAuthState() error = %v", err)
	}

	app.cmdSelectModel([]string{"openrouter:openai/gpt-4o-mini"})

	drainUntilEventType(t, app, model.AgentThinking)
	drainUntilEventType(t, app, model.ModelBrowserClose)
	drainUntilEventType(t, app, model.ModelUpdate)
	drainUntilEventType(t, app, model.AgentReply)

	if got, want := app.Config.Model.Provider, "openai-completion"; got != want {
		t.Fatalf("provider = %q, want %q", got, want)
	}
	if got, want := app.Config.Model.Model, "openai/gpt-4o-mini"; got != want {
		t.Fatalf("model = %q, want %q", got, want)
	}
}

func TestCmdDeleteProvider_RemovesAuthStateAndFallsBackToOtherModel(t *testing.T) {
	app := newModelCommandTestApp()
	home := t.TempDir()
	t.Setenv("HOME", home)

	origAuthPath := authStatePathOverride
	authStatePathOverride = filepath.Join(home, ".mscli", "auth.json")
	t.Cleanup(func() { authStatePathOverride = origAuthPath })
	origModelPath := modelStatePathOverride
	modelStatePathOverride = filepath.Join(home, ".mscli", "model.json")
	t.Cleanup(func() { modelStatePathOverride = origModelPath })
	origCache := modelsDevCachePathOverride
	modelsDevCachePathOverride = filepath.Join(home, ".mscli", "cached", "models-dev-api.json")
	t.Cleanup(func() { modelsDevCachePathOverride = origCache })
	origURL := modelsDevAPIURL
	server := newModelsDevTestServer(`{
		"openrouter": {
			"id": "openrouter",
			"name": "OpenRouter",
			"api": "https://openrouter.ai/api/v1",
			"npm": "@openrouter/ai-sdk-provider",
			"models": {"openai/gpt-4o-mini": {"id": "openai/gpt-4o-mini", "name": "GPT-4o mini"}}
		},
		"anthropic": {
			"id": "anthropic",
			"name": "Anthropic",
			"api": "https://api.anthropic.com",
			"npm": "@ai-sdk/anthropic",
			"models": {"claude-3-5-haiku": {"id": "claude-3-5-haiku", "name": "Claude 3.5 Haiku"}}
		}
	}`)
	defer server.Close()
	modelsDevAPIURL = server.URL
	t.Cleanup(func() { modelsDevAPIURL = origURL })
	resetModelsDevProviderCacheForTest()
	t.Cleanup(resetModelsDevProviderCacheForTest)

	if err := saveProviderAuthState(&providerAuthState{
		Providers: map[string]providerAuthEntry{
			"openrouter": {ProviderID: "openrouter", APIKey: "sk-openrouter"},
			"anthropic":  {ProviderID: "anthropic", APIKey: "sk-anthropic"},
		},
	}); err != nil {
		t.Fatalf("saveProviderAuthState() error = %v", err)
	}
	if err := saveModelSelectionState(&modelSelectionState{
		Active: &modelRef{ProviderID: "openrouter", ModelID: "openai/gpt-4o-mini"},
		Recents: []modelRef{
			{ProviderID: "openrouter", ModelID: "openai/gpt-4o-mini"},
			{ProviderID: "anthropic", ModelID: "claude-3-5-haiku"},
		},
		Favorites: []modelRef{
			{ProviderID: "openrouter", ModelID: "openai/gpt-4o-mini"},
		},
	}); err != nil {
		t.Fatalf("saveModelSelectionState() error = %v", err)
	}

	app.cmdDeleteProvider([]string{"openrouter"})

	drainUntilEventType(t, app, model.AgentThinking)
	ev := drainUntilEventType(t, app, model.ModelUpdate)
	if got, want := ev.Message, "claude-3-5-haiku"; got != want {
		t.Fatalf("model update = %q, want %q", got, want)
	}
	drainUntilEventType(t, app, model.ModelBrowserOpen)

	authState, err := loadProviderAuthState()
	if err != nil {
		t.Fatalf("loadProviderAuthState() error = %v", err)
	}
	if _, ok := authState.Providers["openrouter"]; ok {
		t.Fatalf("authState.Providers = %#v, want openrouter removed", authState.Providers)
	}

	modelState, err := loadModelSelectionState()
	if err != nil {
		t.Fatalf("loadModelSelectionState() error = %v", err)
	}
	if modelState.Active == nil || modelState.Active.ProviderID != "anthropic" {
		t.Fatalf("active = %#v, want anthropic fallback", modelState.Active)
	}
	for _, ref := range append(append([]modelRef{}, modelState.Recents...), modelState.Favorites...) {
		if ref.ProviderID == "openrouter" {
			t.Fatalf("model state still references deleted provider: %#v", modelState)
		}
	}
}

func newModelCommandTestApp() *Application {
	cfg := configs.DefaultConfig()
	cfg.Model.Key = "test-key"
	cfg.Server.URL = "https://issues.example"
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
