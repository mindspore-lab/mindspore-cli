package configs

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDefaultConfigProvider(t *testing.T) {
	cfg := DefaultConfig()

	if got, want := cfg.Model.Provider, "openai-compatible"; got != want {
		t.Fatalf("default provider = %q, want %q", got, want)
	}
}

func TestApplyEnvOverridesProvider(t *testing.T) {
	t.Setenv("MSCLI_PROVIDER", "anthropic")

	cfg := DefaultConfig()
	cfg.Model.Provider = "yaml-provider"

	ApplyEnvOverrides(cfg)

	if got, want := cfg.Model.Provider, "anthropic"; got != want {
		t.Fatalf("provider after env override = %q, want %q", got, want)
	}
}

func TestApplyEnvOverridesProviderAware(t *testing.T) {
	t.Setenv("MSCLI_PROVIDER", "anthropic")
	t.Setenv("OPENAI_API_KEY", "openai-key")
	t.Setenv("OPENAI_BASE_URL", "https://openai.example/v1")

	cfg := DefaultConfig()
	cfg.Model.Key = ""
	cfg.Model.URL = "https://api.openai.com/v1"

	ApplyEnvOverrides(cfg)

	if got, want := cfg.Model.Key, ""; got != want {
		t.Fatalf("anthropic config key after env override = %q, want %q", got, want)
	}

	if got, want := cfg.Model.URL, "https://api.openai.com/v1"; got != want {
		t.Fatalf("anthropic config url after env override = %q, want %q", got, want)
	}
}

func TestLoadWithEnvProvider(t *testing.T) {
	t.Run("defaults when yaml provider blank", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "mscli.yaml")

		if err := os.WriteFile(path, []byte("model:\n  model: gpt-4o-mini\n  provider: \"\"\n"), 0600); err != nil {
			t.Fatalf("write yaml: %v", err)
		}

		cfg, err := LoadWithEnv(path)
		if err != nil {
			t.Fatalf("load config: %v", err)
		}

		if got, want := cfg.Model.Provider, "openai-compatible"; got != want {
			t.Fatalf("provider from blank yaml = %q, want %q", got, want)
		}
	})

	t.Run("env overrides yaml provider", func(t *testing.T) {
		t.Setenv("MSCLI_PROVIDER", "anthropic")

		dir := t.TempDir()
		path := filepath.Join(dir, "mscli.yaml")

		if err := os.WriteFile(path, []byte("model:\n  model: gpt-4o-mini\n  provider: yaml-provider\n"), 0600); err != nil {
			t.Fatalf("write yaml: %v", err)
		}

		cfg, err := LoadWithEnv(path)
		if err != nil {
			t.Fatalf("load config: %v", err)
		}

		if got, want := cfg.Model.Provider, "anthropic"; got != want {
			t.Fatalf("provider from env override = %q, want %q", got, want)
		}
	})

	t.Run("accepts anthropic config with blank model url", func(t *testing.T) {
		clearEnv(t)
		t.Setenv("MSCLI_PROVIDER", "anthropic")
		t.Setenv("ANTHROPIC_AUTH_TOKEN", "anthropic-key")

		dir := t.TempDir()
		path := filepath.Join(dir, "mscli.yaml")

		if err := os.WriteFile(path, []byte("model:\n  model: gpt-4o-mini\n  provider: anthropic\n  url: \"\"\n"), 0600); err != nil {
			t.Fatalf("write yaml: %v", err)
		}

		cfg, err := LoadWithEnv(path)
		if err != nil {
			t.Fatalf("load config: %v", err)
		}

		if got, want := cfg.Model.URL, ""; got != want {
			t.Fatalf("model url from blank yaml = %q, want %q", got, want)
		}
	})
}

func clearEnv(t *testing.T) {
	t.Helper()

	for _, key := range []string{
		"MSCLI_PROVIDER",
		"MSCLI_API_KEY",
		"OPENAI_API_KEY",
		"ANTHROPIC_AUTH_TOKEN",
		"ANTHROPIC_API_KEY",
		"MSCLI_BASE_URL",
		"OPENAI_BASE_URL",
		"ANTHROPIC_BASE_URL",
	} {
		t.Setenv(key, "")
	}
}
