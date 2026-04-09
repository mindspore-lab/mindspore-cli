package app

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"github.com/mindspore-lab/mindspore-cli/configs"
)

func TestRestoreProviderSelectionUsesPersistentState(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	origConfigPath := appConfigPathOverride
	appConfigPathOverride = filepath.Join(home, ".mscli", "config.json")
	t.Cleanup(func() { appConfigPathOverride = origConfigPath })
	origAuthPath := authStatePathOverride
	authStatePathOverride = filepath.Join(home, ".mscli", "auth.json")
	t.Cleanup(func() { authStatePathOverride = origAuthPath })
	origModelPath := modelStatePathOverride
	modelStatePathOverride = filepath.Join(home, ".mscli", "model.json")
	t.Cleanup(func() { modelStatePathOverride = origModelPath })
	origCachePath := modelsDevCachePathOverride
	modelsDevCachePathOverride = filepath.Join(home, ".mscli", "cached", "models-dev-api.json")
	t.Cleanup(func() { modelsDevCachePathOverride = origCachePath })

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
	origURL := modelsDevAPIURL
	modelsDevAPIURL = server.URL
	t.Cleanup(func() { modelsDevAPIURL = origURL })

	if err := saveProviderAuthState(&providerAuthState{
		Providers: map[string]providerAuthEntry{
			"openrouter": {ProviderID: "openrouter", APIKey: "sk-openrouter"},
		},
	}); err != nil {
		t.Fatalf("saveProviderAuthState() error = %v", err)
	}
	if err := saveModelSelectionState(&modelSelectionState{
		Active: &modelRef{ProviderID: "openrouter", ModelID: "openai/gpt-4o-mini"},
	}); err != nil {
		t.Fatalf("saveModelSelectionState() error = %v", err)
	}

	cfg := configs.DefaultConfig()
	result, err := restoreProviderSelection(cfg)
	if err != nil {
		t.Fatalf("restoreProviderSelection() error = %v", err)
	}
	if !result.Restored {
		t.Fatal("result.Restored = false, want true")
	}
	if got, want := cfg.Model.Provider, "openai-completion"; got != want {
		t.Fatalf("cfg.Model.Provider = %q, want %q", got, want)
	}
	if got, want := cfg.Model.Model, "openai/gpt-4o-mini"; got != want {
		t.Fatalf("cfg.Model.Model = %q, want %q", got, want)
	}
}

func TestRestoreProviderSelectionDoesNotUseClaudeCodeEnvImportWithoutConfirmation(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("ANTHROPIC_BASE_URL", "https://api.kimi.com/coding/")
	t.Setenv("ANTHROPIC_API_KEY", "sk-kimi-env")

	origConfigPath := appConfigPathOverride
	appConfigPathOverride = filepath.Join(home, ".mscli", "config.json")
	t.Cleanup(func() { appConfigPathOverride = origConfigPath })
	origAuthPath := authStatePathOverride
	authStatePathOverride = filepath.Join(home, ".mscli", "auth.json")
	t.Cleanup(func() { authStatePathOverride = origAuthPath })
	origModelPath := modelStatePathOverride
	modelStatePathOverride = filepath.Join(home, ".mscli", "model.json")
	t.Cleanup(func() { modelStatePathOverride = origModelPath })
	origCachePath := modelsDevCachePathOverride
	modelsDevCachePathOverride = filepath.Join(home, ".mscli", "cached", "models-dev-api.json")
	t.Cleanup(func() { modelsDevCachePathOverride = origCachePath })

	server := newModelsDevTestServer(`{
		"kimi-for-coding": {
			"id": "kimi-for-coding",
			"name": "Kimi For Coding",
			"api": "https://api.kimi.com/coding/v1",
			"npm": "@ai-sdk/anthropic",
			"models": {"k2p5": {"id": "k2p5", "name": "Kimi K2.5"}}
		}
	}`)
	defer server.Close()
	origURL := modelsDevAPIURL
	modelsDevAPIURL = server.URL
	t.Cleanup(func() { modelsDevAPIURL = origURL })

	if err := saveModelSelectionState(&modelSelectionState{
		Active: &modelRef{ProviderID: "kimi-for-coding", ModelID: "k2p5"},
	}); err != nil {
		t.Fatalf("saveModelSelectionState() error = %v", err)
	}

	cfg := configs.DefaultConfig()
	result, err := restoreProviderSelection(cfg)
	if err != nil {
		t.Fatalf("restoreProviderSelection() error = %v", err)
	}
	if result.Restored {
		t.Fatal("result.Restored = true, want false")
	}
	if got := cfg.Model.Model; got != "" {
		t.Fatalf("cfg.Model.Model = %q, want empty", got)
	}
	if got := cfg.Model.Key; got != "" {
		t.Fatalf("cfg.Model.Key = %q, want empty", got)
	}
}

func TestRestoreProviderSelectionDefaultsToFreeWhenLoggedIn(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	origConfigPath := appConfigPathOverride
	appConfigPathOverride = filepath.Join(home, ".mscli", "config.json")
	t.Cleanup(func() { appConfigPathOverride = origConfigPath })
	origAuthPath := authStatePathOverride
	authStatePathOverride = filepath.Join(home, ".mscli", "auth.json")
	t.Cleanup(func() { authStatePathOverride = origAuthPath })
	origModelPath := modelStatePathOverride
	modelStatePathOverride = filepath.Join(home, ".mscli", "model.json")
	t.Cleanup(func() { modelStatePathOverride = origModelPath })

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]string{"api_key": "server-kimi-key"})
	}))
	defer srv.Close()

	if err := saveCredentials(&credentials{
		ServerURL: srv.URL,
		Token:     "user-token",
		User:      "alice",
		Role:      "user",
	}); err != nil {
		t.Fatalf("saveCredentials() error = %v", err)
	}

	cfg := configs.DefaultConfig()
	result, err := restoreProviderSelection(cfg)
	if err != nil {
		t.Fatalf("restoreProviderSelection() error = %v", err)
	}
	if !result.Restored {
		t.Fatal("result.Restored = false, want true")
	}
	if got, want := result.ActivePresetID, "kimi-k2.5-free"; got != want {
		t.Fatalf("result.ActivePresetID = %q, want %q", got, want)
	}
	if got, want := cfg.Model.Model, "kimi-k2.5"; got != want {
		t.Fatalf("cfg.Model.Model = %q, want %q", got, want)
	}
}

func TestRestoreProviderSelectionLeavesConfigUnsetWhenNotLoggedIn(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	origConfigPath := appConfigPathOverride
	appConfigPathOverride = filepath.Join(home, ".mscli", "config.json")
	t.Cleanup(func() { appConfigPathOverride = origConfigPath })
	origAuthPath := authStatePathOverride
	authStatePathOverride = filepath.Join(home, ".mscli", "auth.json")
	t.Cleanup(func() { authStatePathOverride = origAuthPath })
	origModelPath := modelStatePathOverride
	modelStatePathOverride = filepath.Join(home, ".mscli", "model.json")
	t.Cleanup(func() { modelStatePathOverride = origModelPath })

	cfg := configs.DefaultConfig()
	result, err := restoreProviderSelection(cfg)
	if err != nil {
		t.Fatalf("restoreProviderSelection() error = %v", err)
	}
	if result.Restored {
		t.Fatal("result.Restored = true, want false")
	}
	if got := cfg.Model.Model; got != "" {
		t.Fatalf("cfg.Model.Model = %q, want empty", got)
	}
}
