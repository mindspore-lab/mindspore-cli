package configs

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDefaultConfigProvider(t *testing.T) {
	cfg := DefaultConfig()
	if got, want := cfg.Model.Provider, "openai-completion"; got != want {
		t.Fatalf("default provider = %q, want %q", got, want)
	}
	if got, want := cfg.Context.ReserveTokens, 20000; got != want {
		t.Fatalf("default context.reserve_tokens = %d, want %d", got, want)
	}
}

func TestLoadWithEnv_UsesDefaultsAndEnvOverrides(t *testing.T) {
	clearEnv(t)

	home := t.TempDir()
	t.Setenv("HOME", home)

	projectDir := t.TempDir()
	t.Chdir(projectDir)

	t.Setenv("MSCLI_PROVIDER", "anthropic")
	t.Setenv("MSCLI_MODEL", "env-model")
	t.Setenv("MSCLI_API_KEY", "env-key")
	t.Setenv("MSCLI_BASE_URL", "https://env.example")
	t.Setenv("MSCLI_TEMPERATURE", "0.2")
	t.Setenv("MSCLI_MAX_TOKENS", "4096")
	t.Setenv("MSCLI_MAX_ITERATIONS", "7")
	t.Setenv("MSCLI_CONTEXT_WINDOW", "16000")
	t.Setenv("MSCLI_UI_ENABLED", "false")

	cfg, err := LoadWithEnv()
	if err != nil {
		t.Fatalf("LoadWithEnv() error = %v", err)
	}

	if got, want := cfg.Model.Provider, "anthropic"; got != want {
		t.Fatalf("provider = %q, want %q", got, want)
	}
	if got, want := cfg.Model.Model, "env-model"; got != want {
		t.Fatalf("model = %q, want %q", got, want)
	}
	if got, want := cfg.Model.Key, "env-key"; got != want {
		t.Fatalf("key = %q, want %q", got, want)
	}
	if got, want := cfg.Model.URL, "https://env.example"; got != want {
		t.Fatalf("url = %q, want %q", got, want)
	}
	if cfg.Request.Temperature == nil {
		t.Fatal("request.temperature = nil, want value")
	}
	if got, want := *cfg.Request.Temperature, 0.2; got != want {
		t.Fatalf("request.temperature = %v, want %v", got, want)
	}
	if cfg.Request.MaxTokens == nil {
		t.Fatal("request.max_tokens = nil, want value")
	}
	if got, want := *cfg.Request.MaxTokens, 4096; got != want {
		t.Fatalf("request.max_tokens = %d, want %d", got, want)
	}
	if cfg.Request.MaxIterations == nil {
		t.Fatal("request.max_iterations = nil, want value")
	}
	if got, want := *cfg.Request.MaxIterations, 7; got != want {
		t.Fatalf("request.max_iterations = %d, want %d", got, want)
	}
	if got, want := cfg.Context.Window, 16000; got != want {
		t.Fatalf("context.window = %d, want %d", got, want)
	}
	if got, want := cfg.Context.ReserveTokens, 1600; got != want {
		t.Fatalf("context.reserve_tokens = %d, want %d", got, want)
	}
	if got, want := cfg.UI.Enabled, false; got != want {
		t.Fatalf("ui.enabled = %v, want %v", got, want)
	}
}

func TestLoadWithEnv_IgnoresConfigFiles(t *testing.T) {
	clearEnv(t)

	home := t.TempDir()
	t.Setenv("HOME", home)

	projectDir := t.TempDir()
	t.Chdir(projectDir)

	userPath := filepath.Join(home, ".mscli", "config.yaml")
	if err := os.MkdirAll(filepath.Dir(userPath), 0755); err != nil {
		t.Fatalf("mkdir user config dir: %v", err)
	}
	if err := os.WriteFile(userPath, []byte("model: [\n"), 0600); err != nil {
		t.Fatalf("write user config: %v", err)
	}

	projectPath := filepath.Join(projectDir, ".mscli", "config.yaml")
	if err := os.MkdirAll(filepath.Dir(projectPath), 0755); err != nil {
		t.Fatalf("mkdir project dir: %v", err)
	}
	if err := os.WriteFile(projectPath, []byte("model: [\n"), 0600); err != nil {
		t.Fatalf("write project config: %v", err)
	}

	cfg, err := LoadWithEnv()
	if err != nil {
		t.Fatalf("LoadWithEnv() error = %v", err)
	}
	if got, want := cfg.Model.Model, ""; got != want {
		t.Fatalf("model = %q, want %q", got, want)
	}
	if cfg.Request.MaxIterations == nil {
		t.Fatal("request.max_iterations = nil, want default value")
	}
	if got, want := *cfg.Request.MaxIterations, DefaultRequestMaxIterations; got != want {
		t.Fatalf("request.max_iterations = %d, want %d", got, want)
	}
}

func TestApplyEnvOverrides_OnlyMSCLIVariables(t *testing.T) {
	clearEnv(t)
	t.Setenv("OPENAI_MODEL", "openai-model")
	t.Setenv("OPENAI_API_KEY", "openai-key")
	t.Setenv("OPENAI_BASE_URL", "https://openai.example")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "anthropic-token")
	t.Setenv("ANTHROPIC_API_KEY", "anthropic-api-key")
	t.Setenv("ANTHROPIC_BASE_URL", "https://anthropic.example")

	cfg := DefaultConfig()
	ApplyEnvOverrides(cfg)

	if got, want := cfg.Model.Model, ""; got != want {
		t.Fatalf("model after non-MSCLI env overrides = %q, want %q", got, want)
	}
	if got, want := cfg.Model.Key, ""; got != want {
		t.Fatalf("key after non-MSCLI env overrides = %q, want %q", got, want)
	}
	if got, want := cfg.Model.URL, "https://api.openai.com/v1"; got != want {
		t.Fatalf("url after non-MSCLI env overrides = %q, want %q", got, want)
	}
}

func TestLoadWithEnv_RejectsUnsupportedProviderFromEnv(t *testing.T) {
	clearEnv(t)
	t.Setenv("MSCLI_PROVIDER", "unsupported")
	_, err := LoadWithEnv()
	if err == nil {
		t.Fatal("LoadWithEnv() error = nil, want validation error for unsupported provider")
	}
}

func TestLoadWithEnv_RejectsNegativeMaxIterationsFromEnv(t *testing.T) {
	clearEnv(t)
	t.Setenv("MSCLI_MAX_ITERATIONS", "-1")

	_, err := LoadWithEnv()
	if err == nil {
		t.Fatal("LoadWithEnv() error = nil, want validation error for negative max_iterations")
	}
}

func TestLoadWithEnv_IgnoresWhitespaceOnlyModelEnv(t *testing.T) {
	clearEnv(t)
	t.Setenv("MSCLI_MODEL", "   ")

	cfg, err := LoadWithEnv()
	if err != nil {
		t.Fatalf("LoadWithEnv() error = %v", err)
	}
	if got, want := cfg.Model.Model, ""; got != want {
		t.Fatalf("model = %q, want %q", got, want)
	}
}

func TestLoadWithEnv_AutoTokenLimitsForEnvModelOverride(t *testing.T) {
	clearEnv(t)

	home := t.TempDir()
	t.Setenv("HOME", home)
	projectDir := t.TempDir()
	t.Chdir(projectDir)

	t.Setenv("MSCLI_MODEL", "gpt-5.4")

	cfg, err := LoadWithEnv()
	if err != nil {
		t.Fatalf("LoadWithEnv() error = %v", err)
	}

	if got, want := cfg.Context.Window, 1050000; got != want {
		t.Fatalf("context.window = %d, want %d", got, want)
	}
}

func clearEnv(t *testing.T) {
	t.Helper()

	for _, key := range []string{
		"MSCLI_PROVIDER",
		"MSCLI_API_KEY",
		"MSCLI_BASE_URL",
		"MSCLI_MODEL",
		"MSCLI_TEMPERATURE",
		"MSCLI_MAX_TOKENS",
		"MSCLI_MAX_ITERATIONS",
		"MSCLI_TIMEOUT",
		"MSCLI_CONTEXT_WINDOW",
		"MSCLI_CONTEXT_RESERVE",
		"MSCLI_UI_ENABLED",
		"MSCLI_PERMISSIONS_SKIP",
		"MSCLI_PERMISSIONS_DEFAULT",
		"MSCLI_MEMORY_ENABLED",
		"MSCLI_MEMORY_PATH",
		"MSCLI_SERVER_URL",
		"OPENAI_API_KEY",
		"OPENAI_MODEL",
		"OPENAI_BASE_URL",
		"ANTHROPIC_AUTH_TOKEN",
		"ANTHROPIC_API_KEY",
		"ANTHROPIC_BASE_URL",
	} {
		t.Setenv(key, "")
	}
}
