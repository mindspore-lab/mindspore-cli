package main

import (
	"path/filepath"
	"testing"
)

func TestPersistSessionState_DisableAPIKeyPersistence(t *testing.T) {
	path := filepath.Join(t.TempDir(), "session.yaml")

	app := &Application{
		SessionPath: path,
		SessionState: PersistentState{
			Version: 1,
			APIKeys: PersistedAPIKeys{
				OpenAI:     "legacy-openai",
				OpenRouter: "legacy-openrouter",
			},
		},
		SessionModel: SessionModel{
			Provider: "openai",
			Name:     "gpt-4o-mini",
		},
		Config: Config{
			Providers: ProvidersConfig{
				OpenAI:     ProviderConfig{APIKeyEnv: "OPENAI_API_KEY"},
				OpenRouter: ProviderConfig{APIKeyEnv: "OPENROUTER_API_KEY"},
			},
			Session: SessionConfig{
				PersistAPIKeys: false,
			},
		},
	}

	t.Setenv("OPENAI_API_KEY", "new-openai")
	t.Setenv("OPENROUTER_API_KEY", "new-openrouter")

	if err := app.persistSessionState(); err != nil {
		t.Fatalf("persistSessionState failed: %v", err)
	}
	if app.SessionState.APIKeys.OpenAI != "" || app.SessionState.APIKeys.OpenRouter != "" {
		t.Fatalf("expected in-memory keys to be cleared when persistence disabled, got %+v", app.SessionState.APIKeys)
	}

	stored, err := LoadPersistentState(path)
	if err != nil {
		t.Fatalf("LoadPersistentState failed: %v", err)
	}
	if stored.APIKeys.OpenAI != "" || stored.APIKeys.OpenRouter != "" {
		t.Fatalf("expected persisted keys to be empty, got %+v", stored.APIKeys)
	}
}
