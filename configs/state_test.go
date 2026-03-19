package configs

import "testing"

func TestStateManager_PersistsProvider(t *testing.T) {
	workDir := t.TempDir()

	cfg := DefaultConfig()
	cfg.Model.Provider = "anthropic"
	cfg.Model.Model = "claude-3-5-sonnet"
	cfg.Model.Key = "token-123"

	manager := NewStateManager(workDir)
	manager.SaveFromConfig(cfg)
	if err := manager.Save(); err != nil {
		t.Fatalf("Save() error = %v", err)
	}

	loaded := NewStateManager(workDir)
	if err := loaded.Load(); err != nil {
		t.Fatalf("Load() error = %v", err)
	}

	target := DefaultConfig()
	target.Model.Provider = "openai-compatible"
	target.Model.Model = "gpt-4o-mini"
	target.Model.Key = ""
	loaded.ApplyToConfig(target)

	if got, want := target.Model.Provider, "anthropic"; got != want {
		t.Fatalf("provider = %q, want %q", got, want)
	}
	if got, want := target.Model.Model, "claude-3-5-sonnet"; got != want {
		t.Fatalf("model = %q, want %q", got, want)
	}
	if got, want := target.Model.Key, "token-123"; got != want {
		t.Fatalf("key = %q, want %q", got, want)
	}
}
