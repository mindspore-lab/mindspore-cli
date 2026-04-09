package app

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestAppConfigRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	origPath := appConfigPathOverride
	appConfigPathOverride = path
	t.Cleanup(func() { appConfigPathOverride = origPath })

	cfg := &appConfig{
		ModelMode:     "mscli-provided",
		ModelPresetID: "kimi-k2.5-free",
		ModelToken:    "sk-test-token-123",
	}
	if err := saveAppConfig(cfg); err != nil {
		t.Fatalf("saveAppConfig: %v", err)
	}

	loaded, err := loadAppConfig()
	if err != nil {
		t.Fatalf("loadAppConfig: %v", err)
	}
	if loaded.ModelMode != cfg.ModelMode {
		t.Errorf("ModelMode = %q, want %q", loaded.ModelMode, cfg.ModelMode)
	}
	if loaded.ModelPresetID != cfg.ModelPresetID {
		t.Errorf("ModelPresetID = %q, want %q", loaded.ModelPresetID, cfg.ModelPresetID)
	}
	if loaded.ModelToken != cfg.ModelToken {
		t.Errorf("ModelToken = %q, want %q", loaded.ModelToken, cfg.ModelToken)
	}
}

func TestLoadAppConfigMissing(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "nonexistent", "config.json")
	origPath := appConfigPathOverride
	appConfigPathOverride = path
	t.Cleanup(func() { appConfigPathOverride = origPath })

	cfg, err := loadAppConfig()
	if err != nil {
		t.Fatalf("expected nil error for missing file, got: %v", err)
	}
	if cfg.ModelMode != "" {
		t.Errorf("expected empty ModelMode, got %q", cfg.ModelMode)
	}
}

func TestAppConfigRoundTripPreservesExtraProviders(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	origPath := appConfigPathOverride
	appConfigPathOverride = path
	t.Cleanup(func() { appConfigPathOverride = origPath })

	cfg := &appConfig{
		ModelMode:     "mscli-provided",
		ModelPresetID: "kimi-k2.5-free",
		ModelToken:    "deprecated-token",
		ExtraProviders: []extraProviderConfig{
			{
				ID:       "openrouter",
				Label:    "OpenRouter",
				BaseURL:  "https://openrouter.ai/api/v1",
				Protocol: "openai-chat",
				Models: []extraProviderModelConfig{
					{ID: "openai/gpt-4o-mini", Label: "GPT-4o mini"},
				},
			},
		},
	}
	if err := saveAppConfig(cfg); err != nil {
		t.Fatalf("saveAppConfig: %v", err)
	}

	loaded, err := loadAppConfig()
	if err != nil {
		t.Fatalf("loadAppConfig: %v", err)
	}
	if got, want := len(loaded.ExtraProviders), 1; got != want {
		t.Fatalf("len(loaded.ExtraProviders) = %d, want %d", got, want)
	}
	if got, want := loaded.ExtraProviders[0].ID, "openrouter"; got != want {
		t.Fatalf("loaded.ExtraProviders[0].ID = %q, want %q", got, want)
	}
	if got, want := loaded.ExtraProviders[0].Models[0].ID, "openai/gpt-4o-mini"; got != want {
		t.Fatalf("loaded.ExtraProviders[0].Models[0].ID = %q, want %q", got, want)
	}
	if got, want := loaded.ModelToken, "deprecated-token"; got != want {
		t.Fatalf("loaded.ModelToken = %q, want %q", got, want)
	}
}

func TestLoadAppConfigExtraProvidersLabelOptional(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	origPath := appConfigPathOverride
	appConfigPathOverride = path
	t.Cleanup(func() { appConfigPathOverride = origPath })

	data, err := json.MarshalIndent(map[string]any{
		"extra_providers": []map[string]any{
			{
				"id":       "my-gateway",
				"base_url": "https://llm.example.com/v1",
				"protocol": "openai-chat",
			},
		},
	}, "", "  ")
	if err != nil {
		t.Fatalf("json.MarshalIndent() error = %v", err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("os.WriteFile() error = %v", err)
	}

	loaded, err := loadAppConfig()
	if err != nil {
		t.Fatalf("loadAppConfig() error = %v", err)
	}
	if got, want := len(loaded.ExtraProviders), 1; got != want {
		t.Fatalf("len(loaded.ExtraProviders) = %d, want %d", got, want)
	}
	if got, want := loaded.ExtraProviders[0].ID, "my-gateway"; got != want {
		t.Fatalf("loaded.ExtraProviders[0].ID = %q, want %q", got, want)
	}
	if got := loaded.ExtraProviders[0].Label; got != "" {
		t.Fatalf("loaded.ExtraProviders[0].Label = %q, want empty", got)
	}
}
