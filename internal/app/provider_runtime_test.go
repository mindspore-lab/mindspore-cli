package app

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/mindspore-lab/mindspore-cli/configs"
)

func TestResolveRuntimeSelectionForExtraProviderOpenAIChat(t *testing.T) {
	catalog := &providerCatalog{
		Providers: []providerCatalogEntry{
			{
				ID:       "openrouter",
				Label:    "OpenRouter",
				BaseURL:  "https://openrouter.ai/api/v1",
				Protocol: "openai-chat",
			},
		},
	}
	auth := &providerAuthState{
		Providers: map[string]providerAuthEntry{
			"openrouter": {ProviderID: "openrouter", APIKey: "sk-openrouter"},
		},
	}

	cfg, presetID, err := resolveRuntimeSelection(catalog, auth, modelRef{
		ProviderID: "openrouter",
		ModelID:    "openai/gpt-4o-mini",
	})
	if err != nil {
		t.Fatalf("resolveRuntimeSelection() error = %v", err)
	}
	if presetID != "" {
		t.Fatalf("presetID = %q, want empty", presetID)
	}
	if got, want := cfg.Provider, "openai-completion"; got != want {
		t.Fatalf("cfg.Provider = %q, want %q", got, want)
	}
	if got, want := cfg.URL, "https://openrouter.ai/api/v1"; got != want {
		t.Fatalf("cfg.URL = %q, want %q", got, want)
	}
	if got, want := cfg.Key, "sk-openrouter"; got != want {
		t.Fatalf("cfg.Key = %q, want %q", got, want)
	}
}

func TestResolveRuntimeSelectionForFreeProviderRequiresLogin(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	catalog := &providerCatalog{
		Providers: builtinProviderCatalog(),
	}
	_, _, err := resolveRuntimeSelection(catalog, emptyProviderAuthState(), modelRef{
		ProviderID: mindsporeCLIFreeProviderID,
		ModelID:    "kimi-k2.5",
	})
	if err == nil {
		t.Fatal("resolveRuntimeSelection() error = nil, want login requirement")
	}
}

func TestResolveRuntimeSelectionForFreeProviderUsesPresetCredential(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got, want := r.Header.Get("Authorization"), "Bearer user-token"; got != want {
			t.Fatalf("Authorization = %q, want %q", got, want)
		}
		if r.URL.Path != "/model-presets/kimi-k2.5-free/credential" {
			t.Fatalf("path = %q, want %q", r.URL.Path, "/model-presets/kimi-k2.5-free/credential")
		}
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

	catalog := &providerCatalog{
		Providers: builtinProviderCatalog(),
	}
	cfg, presetID, err := resolveRuntimeSelection(catalog, emptyProviderAuthState(), modelRef{
		ProviderID: mindsporeCLIFreeProviderID,
		ModelID:    "kimi-k2.5",
	})
	if err != nil {
		t.Fatalf("resolveRuntimeSelection() error = %v", err)
	}
	if got, want := presetID, "kimi-k2.5-free"; got != want {
		t.Fatalf("presetID = %q, want %q", got, want)
	}
	if got, want := cfg.Provider, "anthropic"; got != want {
		t.Fatalf("cfg.Provider = %q, want %q", got, want)
	}
	if got, want := cfg.Key, "server-kimi-key"; got != want {
		t.Fatalf("cfg.Key = %q, want %q", got, want)
	}
}

func TestApplyResolvedModelConfigMutatesConfig(t *testing.T) {
	cfg := configs.DefaultConfig()
	cfg.Model.Provider = "openai-completion"
	cfg.Model.URL = "https://api.openai.com/v1"
	cfg.Model.Model = ""
	cfg.Model.Key = ""

	applyResolvedModelConfig(cfg, configs.ModelConfig{
		Provider: "anthropic",
		URL:      "https://api.kimi.com/coding/",
		Model:    "kimi-k2.5",
		Key:      "server-kimi-key",
	})

	if got, want := cfg.Model.Provider, "anthropic"; got != want {
		t.Fatalf("cfg.Model.Provider = %q, want %q", got, want)
	}
	if got, want := cfg.Model.URL, "https://api.kimi.com/coding/"; got != want {
		t.Fatalf("cfg.Model.URL = %q, want %q", got, want)
	}
	if got, want := cfg.Model.Model, "kimi-k2.5"; got != want {
		t.Fatalf("cfg.Model.Model = %q, want %q", got, want)
	}
}
