package app

import (
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
		ModelMode:            "mscli-provided",
		ModelPresetID:        "kimi-k2.5-free",
		ModelToken:           "sk-test-token-123",
		SessionRetentionDays: 45,
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
	if loaded.SessionRetentionDays != cfg.SessionRetentionDays {
		t.Errorf("SessionRetentionDays = %d, want %d", loaded.SessionRetentionDays, cfg.SessionRetentionDays)
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
	if got, want := cfg.sessionRetentionDays(), defaultSessionRetentionDays; got != want {
		t.Errorf("sessionRetentionDays() = %d, want %d", got, want)
	}
}
