package app

import (
	"path/filepath"
	"testing"
)

func TestDetectModelMode_EnvWins(t *testing.T) {
	t.Setenv("MSCLI_PROVIDER", "openai-completion")
	t.Setenv("MSCLI_API_KEY", "sk-test")
	t.Setenv("MSCLI_MODEL", "gpt-4o")

	mode, _ := detectModelMode()
	if mode != modelModeOwnEnv {
		t.Errorf("expected %q, got %q", modelModeOwnEnv, mode)
	}
}

func TestDetectModelMode_SavedToken(t *testing.T) {
	t.Setenv("MSCLI_PROVIDER", "")
	t.Setenv("MSCLI_API_KEY", "")
	t.Setenv("MSCLI_MODEL", "")

	dir := t.TempDir()
	origPath := appConfigPathOverride
	appConfigPathOverride = filepath.Join(dir, "config.json")
	t.Cleanup(func() { appConfigPathOverride = origPath })

	if err := saveAppConfig(&appConfig{
		ModelMode:     modelModeMSCLIProvided,
		ModelPresetID: "kimi-k2.5-free",
		ModelToken:    "sk-saved",
	}); err != nil {
		t.Fatal(err)
	}

	mode, cfg := detectModelMode()
	if mode != modelModeMSCLIProvided {
		t.Errorf("expected %q, got %q", modelModeMSCLIProvided, mode)
	}
	if cfg == nil || cfg.ModelToken != "sk-saved" {
		t.Error("expected returned config to contain saved token")
	}
}

func TestDetectModelMode_NothingConfigured(t *testing.T) {
	t.Setenv("MSCLI_PROVIDER", "")
	t.Setenv("MSCLI_API_KEY", "")
	t.Setenv("MSCLI_MODEL", "")

	dir := t.TempDir()
	origPath := appConfigPathOverride
	appConfigPathOverride = filepath.Join(dir, "nonexistent", "config.json")
	t.Cleanup(func() { appConfigPathOverride = origPath })

	mode, _ := detectModelMode()
	if mode != "" {
		t.Errorf("expected empty string, got %q", mode)
	}
}

func TestDetectModelMode_BothEnvAndSavedToken_EnvWins(t *testing.T) {
	t.Setenv("MSCLI_PROVIDER", "openai-completion")
	t.Setenv("MSCLI_API_KEY", "sk-env-key")
	t.Setenv("MSCLI_MODEL", "gpt-4o")

	dir := t.TempDir()
	origPath := appConfigPathOverride
	appConfigPathOverride = filepath.Join(dir, "config.json")
	t.Cleanup(func() { appConfigPathOverride = origPath })

	saveAppConfig(&appConfig{
		ModelMode:     modelModeMSCLIProvided,
		ModelPresetID: "kimi-k2.5-free",
		ModelToken:    "sk-saved-token",
	})

	mode, _ := detectModelMode()
	if mode != modelModeOwnEnv {
		t.Errorf("expected %q (env wins), got %q", modelModeOwnEnv, mode)
	}
}
