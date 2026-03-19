package app

import (
	"errors"
	"testing"

	"github.com/vigo999/ms-cli/configs"
)

func TestInitProviderAnthropic(t *testing.T) {
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "anthropic-token")
	t.Setenv("MSCLI_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")

	provider, err := initProvider(configs.ModelConfig{
		Provider: "anthropic",
		Model:    "claude-3-5-sonnet",
	})
	if err != nil {
		t.Fatalf("initProvider() error = %v", err)
	}
	if provider == nil {
		t.Fatal("initProvider() provider = nil, want provider")
	}
	if got, want := provider.Name(), "anthropic"; got != want {
		t.Fatalf("provider.Name() = %q, want %q", got, want)
	}
}

func TestInitProviderOpenAICompatibleDefault(t *testing.T) {
	t.Setenv("MSCLI_PROVIDER", "")
	t.Setenv("MSCLI_API_KEY", "mscli-token")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "")

	provider, err := initProvider(configs.ModelConfig{Model: "gpt-4o-mini"})
	if err != nil {
		t.Fatalf("initProvider() error = %v", err)
	}
	if provider == nil {
		t.Fatal("initProvider() provider = nil, want provider")
	}
	if got, want := provider.Name(), "openai-compatible"; got != want {
		t.Fatalf("provider.Name() = %q, want %q", got, want)
	}
}

func TestInitProviderMapsMissingKeyToAppSentinel(t *testing.T) {
	t.Setenv("MSCLI_PROVIDER", "")
	t.Setenv("MSCLI_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "")

	_, err := initProvider(configs.ModelConfig{Model: "gpt-4o-mini"})
	if err == nil {
		t.Fatal("initProvider() error = nil, want missing api key error")
	}
	if !errors.Is(err, errAPIKeyNotFound) {
		t.Fatalf("initProvider() error = %v, want errAPIKeyNotFound", err)
	}
}
