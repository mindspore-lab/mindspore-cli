package configs

import "testing"

func TestDefaultConfigProvider(t *testing.T) {
	cfg := DefaultConfig()

	if got, want := cfg.Model.Provider, "openai-compatible"; got != want {
		t.Fatalf("default provider = %q, want %q", got, want)
	}
}

func TestApplyEnvOverridesProvider(t *testing.T) {
	t.Setenv("MSCLI_PROVIDER", "anthropic-compatible")

	cfg := DefaultConfig()
	cfg.Model.Provider = "yaml-provider"

	ApplyEnvOverrides(cfg)

	if got, want := cfg.Model.Provider, "anthropic-compatible"; got != want {
		t.Fatalf("provider after env override = %q, want %q", got, want)
	}
}
