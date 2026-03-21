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
	if got, want := cfg.Permissions.DefaultMode, "default"; got != want {
		t.Fatalf("default permission mode = %q, want %q", got, want)
	}
}

func TestLoadWithEnv_MergesFixedLayers(t *testing.T) {
	clearEnv(t)

	home := t.TempDir()
	t.Setenv("HOME", home)

	projectDir := t.TempDir()
	t.Chdir(projectDir)

	userPath := filepath.Join(home, ".ms-cli", "config.yaml")
	if err := os.MkdirAll(filepath.Dir(userPath), 0755); err != nil {
		t.Fatalf("mkdir user config dir: %v", err)
	}
	if err := os.WriteFile(userPath, []byte(`model:
  provider: openai
  model: user-model
  temperature: 0.2
context:
  max_tokens: 16000
`), 0600); err != nil {
		t.Fatalf("write user config: %v", err)
	}

	projectPath := filepath.Join(projectDir, ".ms-cli", "config.yaml")
	if err := os.MkdirAll(filepath.Dir(projectPath), 0755); err != nil {
		t.Fatalf("mkdir project config dir: %v", err)
	}
	if err := os.WriteFile(projectPath, []byte(`model:
  model: project-model
ui:
  enabled: false
`), 0600); err != nil {
		t.Fatalf("write project config: %v", err)
	}

	t.Setenv("MSCLI_PROVIDER", "anthropic")
	t.Setenv("MSCLI_MODEL", "env-model")
	t.Setenv("MSCLI_API_KEY", "env-key")
	t.Setenv("MSCLI_BASE_URL", "https://env.example")

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
	if got, want := cfg.Model.Temperature, 0.2; got != want {
		t.Fatalf("temperature = %v, want %v", got, want)
	}
	if got, want := cfg.Context.MaxTokens, 16000; got != want {
		t.Fatalf("context.max_tokens = %d, want %d", got, want)
	}
	if got, want := cfg.UI.Enabled, false; got != want {
		t.Fatalf("ui.enabled = %v, want %v", got, want)
	}
}

func TestLoadWithEnv_UsesFixedProjectPathOnly(t *testing.T) {
	clearEnv(t)

	home := t.TempDir()
	t.Setenv("HOME", home)

	projectDir := t.TempDir()
	t.Chdir(projectDir)

	projectPath := filepath.Join(projectDir, ".ms-cli", "config.yaml")
	if err := os.MkdirAll(filepath.Dir(projectPath), 0755); err != nil {
		t.Fatalf("mkdir project dir: %v", err)
	}
	if err := os.WriteFile(projectPath, []byte("model:\n  model: project-model\n"), 0600); err != nil {
		t.Fatalf("write project config: %v", err)
	}

	ignoredPath := filepath.Join(projectDir, "custom", "config.yaml")
	if err := os.MkdirAll(filepath.Dir(ignoredPath), 0755); err != nil {
		t.Fatalf("mkdir ignored dir: %v", err)
	}
	if err := os.WriteFile(ignoredPath, []byte("model:\n  model: ignored-model\n"), 0600); err != nil {
		t.Fatalf("write ignored config: %v", err)
	}

	cfg, err := LoadWithEnv()
	if err != nil {
		t.Fatalf("LoadWithEnv() error = %v", err)
	}
	if got, want := cfg.Model.Model, "project-model"; got != want {
		t.Fatalf("model = %q, want %q", got, want)
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

	if got, want := cfg.Model.Model, "gpt-4o-mini"; got != want {
		t.Fatalf("model after non-MSCLI env overrides = %q, want %q", got, want)
	}
	if got, want := cfg.Model.Key, ""; got != want {
		t.Fatalf("key after non-MSCLI env overrides = %q, want %q", got, want)
	}
	if got, want := cfg.Model.URL, "https://api.openai.com/v1"; got != want {
		t.Fatalf("url after non-MSCLI env overrides = %q, want %q", got, want)
	}
}

func TestLoadWithEnvRejectsWhitespaceOnlyModel(t *testing.T) {
	clearEnv(t)

	dir := t.TempDir()
	path := filepath.Join(dir, ".ms-cli", "config.yaml")
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatalf("mkdir config dir: %v", err)
	}

	if err := os.WriteFile(path, []byte("model:\n  model: \"   \"\n"), 0600); err != nil {
		t.Fatalf("write yaml: %v", err)
	}

	t.Chdir(dir)
	_, err := LoadWithEnv()
	if err == nil {
		t.Fatal("LoadWithEnv() error = nil, want validation error for whitespace-only model")
	}
}

func TestLoadWithEnvRejectsUnsupportedProvider(t *testing.T) {
	clearEnv(t)

	dir := t.TempDir()
	path := filepath.Join(dir, ".ms-cli", "config.yaml")
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatalf("mkdir config dir: %v", err)
	}

	if err := os.WriteFile(path, []byte("model:\n  model: gpt-4o-mini\n  provider: unsupported\n"), 0600); err != nil {
		t.Fatalf("write yaml: %v", err)
	}

	t.Chdir(dir)
	_, err := LoadWithEnv()
	if err == nil {
		t.Fatal("LoadWithEnv() error = nil, want validation error for unsupported provider")
	}
}

func TestLoadWithEnv_IgnoresLegacyDotMscliPaths(t *testing.T) {
	clearEnv(t)

	home := t.TempDir()
	t.Setenv("HOME", home)
	projectDir := t.TempDir()
	t.Chdir(projectDir)

	legacyUserPath := filepath.Join(home, ".mscli", "config.yaml")
	if err := os.MkdirAll(filepath.Dir(legacyUserPath), 0755); err != nil {
		t.Fatalf("mkdir legacy user config dir: %v", err)
	}
	if err := os.WriteFile(legacyUserPath, []byte("model:\n  model: legacy-user\n"), 0600); err != nil {
		t.Fatalf("write legacy user config: %v", err)
	}

	legacyProjectPath := filepath.Join(projectDir, ".mscli", "config.yaml")
	if err := os.MkdirAll(filepath.Dir(legacyProjectPath), 0755); err != nil {
		t.Fatalf("mkdir legacy project config dir: %v", err)
	}
	if err := os.WriteFile(legacyProjectPath, []byte("model:\n  model: legacy-project\n"), 0600); err != nil {
		t.Fatalf("write legacy project config: %v", err)
	}

	cfg, err := LoadWithEnv()
	if err != nil {
		t.Fatalf("LoadWithEnv() error = %v", err)
	}
	if got, want := cfg.Model.Model, "gpt-4o-mini"; got != want {
		t.Fatalf("model = %q, want %q (legacy .mscli path should be ignored)", got, want)
	}
}

func TestLoadWithEnv_PermissionsRuleBuckets(t *testing.T) {
	clearEnv(t)

	projectDir := t.TempDir()
	t.Chdir(projectDir)

	projectPath := filepath.Join(projectDir, ".ms-cli", "config.yaml")
	if err := os.MkdirAll(filepath.Dir(projectPath), 0755); err != nil {
		t.Fatalf("mkdir project config dir: %v", err)
	}
	if err := os.WriteFile(projectPath, []byte(`permissions:
  default_mode: default
  default_level: ask
  allow:
    - "Bash(npm test *)"
  ask:
    - "Bash(git push *)"
  deny:
    - "Agent(Plan)"
`), 0600); err != nil {
		t.Fatalf("write project config: %v", err)
	}

	cfg, err := LoadWithEnv()
	if err != nil {
		t.Fatalf("LoadWithEnv() error = %v", err)
	}
	if len(cfg.Permissions.Allow) != 1 || cfg.Permissions.Allow[0] != "Bash(npm test *)" {
		t.Fatalf("permissions.allow = %#v, want Bash(npm test *)", cfg.Permissions.Allow)
	}
	if len(cfg.Permissions.Ask) != 1 || cfg.Permissions.Ask[0] != "Bash(git push *)" {
		t.Fatalf("permissions.ask = %#v, want Bash(git push *)", cfg.Permissions.Ask)
	}
	if len(cfg.Permissions.Deny) != 1 || cfg.Permissions.Deny[0] != "Agent(Plan)" {
		t.Fatalf("permissions.deny = %#v, want Agent(Plan)", cfg.Permissions.Deny)
	}
	if got := cfg.Permissions.RuleSources["Bash(npm test *)"]; got != "project" {
		t.Fatalf("permissions source allow = %q, want project", got)
	}
	if got := cfg.Permissions.RuleSources["Bash(git push *)"]; got != "project" {
		t.Fatalf("permissions source ask = %q, want project", got)
	}
	if got := cfg.Permissions.RuleSources["Agent(Plan)"]; got != "project" {
		t.Fatalf("permissions source deny = %q, want project", got)
	}
}

func TestLoadWithEnv_PermissionsRuleSourcesUserAndProject(t *testing.T) {
	clearEnv(t)

	home := t.TempDir()
	t.Setenv("HOME", home)
	projectDir := t.TempDir()
	t.Chdir(projectDir)

	userPath := filepath.Join(home, ".ms-cli", "config.yaml")
	if err := os.MkdirAll(filepath.Dir(userPath), 0755); err != nil {
		t.Fatalf("mkdir user config dir: %v", err)
	}
	if err := os.WriteFile(userPath, []byte(`permissions:
  allow:
    - "Read"
  ask:
    - "Bash(git push *)"
`), 0600); err != nil {
		t.Fatalf("write user config: %v", err)
	}

	projectPath := filepath.Join(projectDir, ".ms-cli", "config.yaml")
	if err := os.MkdirAll(filepath.Dir(projectPath), 0755); err != nil {
		t.Fatalf("mkdir project config dir: %v", err)
	}
	if err := os.WriteFile(projectPath, []byte(`permissions:
  deny:
    - "Agent(Plan)"
`), 0600); err != nil {
		t.Fatalf("write project config: %v", err)
	}

	cfg, err := LoadWithEnv()
	if err != nil {
		t.Fatalf("LoadWithEnv() error = %v", err)
	}

	if got := cfg.Permissions.RuleSources["Read"]; got != "user" {
		t.Fatalf("source(Read) = %q, want user", got)
	}
	if got := cfg.Permissions.RuleSources["Bash(git push *)"]; got != "user" {
		t.Fatalf("source(Bash(git push *)) = %q, want user", got)
	}
	if got := cfg.Permissions.RuleSources["Agent(Plan)"]; got != "project" {
		t.Fatalf("source(Agent(Plan)) = %q, want project", got)
	}
}

func TestLoadWithEnv_PermissionsRuleSourcesManagedLayer(t *testing.T) {
	clearEnv(t)

	projectDir := t.TempDir()
	t.Chdir(projectDir)

	managedPath := filepath.Join(t.TempDir(), "managed.yaml")
	if err := os.WriteFile(managedPath, []byte(`permissions:
  allow:
    - "Read"
  deny:
    - "Agent(Plan)"
`), 0600); err != nil {
		t.Fatalf("write managed config: %v", err)
	}
	t.Setenv("MSCLI_MANAGED_CONFIG", managedPath)

	home := t.TempDir()
	t.Setenv("HOME", home)
	userPath := filepath.Join(home, ".ms-cli", "config.yaml")
	if err := os.MkdirAll(filepath.Dir(userPath), 0755); err != nil {
		t.Fatalf("mkdir user config dir: %v", err)
	}
	if err := os.WriteFile(userPath, []byte(`permissions:
  allow:
    - "Read"
  ask:
    - "Bash(git push *)"
`), 0600); err != nil {
		t.Fatalf("write user config: %v", err)
	}

	projectPath := filepath.Join(projectDir, ".ms-cli", "config.yaml")
	if err := os.MkdirAll(filepath.Dir(projectPath), 0755); err != nil {
		t.Fatalf("mkdir project config dir: %v", err)
	}
	if err := os.WriteFile(projectPath, []byte(`permissions:
  ask:
    - "Read"
`), 0600); err != nil {
		t.Fatalf("write project config: %v", err)
	}

	cfg, err := LoadWithEnv()
	if err != nil {
		t.Fatalf("LoadWithEnv() error = %v", err)
	}

	// effective value should follow merge precedence: project > user > managed.
	if len(cfg.Permissions.Ask) == 0 || cfg.Permissions.Ask[0] != "Read" {
		t.Fatalf("permissions.ask = %#v, want Read from project override", cfg.Permissions.Ask)
	}
	// source map still tracks origin layer per declared rule.
	if got := cfg.Permissions.RuleSources["Agent(Plan)"]; got != "managed" {
		t.Fatalf("source(Agent(Plan)) = %q, want managed", got)
	}
}

func clearEnv(t *testing.T) {
	t.Helper()

	for _, key := range []string{
		"MSCLI_PROVIDER",
		"MSCLI_API_KEY",
		"MSCLI_BASE_URL",
		"MSCLI_MODEL",
		"MSCLI_MANAGED_CONFIG",
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
