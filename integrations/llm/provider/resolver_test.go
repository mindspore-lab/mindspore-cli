package provider

import (
	"errors"
	"testing"

	"github.com/vigo999/ms-cli/configs"
)

func TestResolveConfig_ProviderPriority(t *testing.T) {
	clearResolverEnv(t)
	t.Setenv("MSCLI_PROVIDER", "  ANTHROPIC  ")

	cfg := configs.ModelConfig{
		Provider: "openai",
		Key:      "cfg-key",
	}

	got, err := ResolveConfig(cfg)
	if err != nil {
		t.Fatalf("ResolveConfig() error = %v", err)
	}

	if got.Kind != ProviderAnthropic {
		t.Fatalf("ResolveConfig() kind = %q, want %q", got.Kind, ProviderAnthropic)
	}
}

func TestResolveConfig_ProviderFallbacks(t *testing.T) {
	t.Run("cfg provider used when env absent", func(t *testing.T) {
		clearResolverEnv(t)

		got, err := ResolveConfig(configs.ModelConfig{Provider: "  anthropic  ", Key: "cfg-key"})
		if err != nil {
			t.Fatalf("ResolveConfig() error = %v", err)
		}

		if got.Kind != ProviderAnthropic {
			t.Fatalf("ResolveConfig() kind = %q, want %q", got.Kind, ProviderAnthropic)
		}
	})

	t.Run("default openai compatible when env and cfg absent", func(t *testing.T) {
		clearResolverEnv(t)

		got, err := ResolveConfig(configs.ModelConfig{Key: "cfg-key"})
		if err != nil {
			t.Fatalf("ResolveConfig() error = %v", err)
		}

		if got.Kind != ProviderOpenAICompatible {
			t.Fatalf("ResolveConfig() kind = %q, want %q", got.Kind, ProviderOpenAICompatible)
		}
	})
}

func TestResolveConfig_OpenAIKeyPriority(t *testing.T) {
	clearResolverEnv(t)
	t.Setenv("MSCLI_API_KEY", "MsCli-Key")
	t.Setenv("OPENAI_API_KEY", "OpenAI-Key")

	cfg := configs.ModelConfig{
		Key: "cfg-key",
	}

	got, err := ResolveConfig(cfg)
	if err != nil {
		t.Fatalf("ResolveConfig() error = %v", err)
	}

	if got.APIKey != "MsCli-Key" {
		t.Fatalf("ResolveConfig() APIKey = %q, want %q", got.APIKey, "MsCli-Key")
	}
}

func TestResolveConfig_OpenAIKeyFallbacks(t *testing.T) {
	t.Run("OPENAI_API_KEY used when MSCLI_API_KEY absent", func(t *testing.T) {
		clearResolverEnv(t)
		t.Setenv("OPENAI_API_KEY", "OpenAI-Key")

		got, err := ResolveConfig(configs.ModelConfig{})
		if err != nil {
			t.Fatalf("ResolveConfig() error = %v", err)
		}

		if got.APIKey != "OpenAI-Key" {
			t.Fatalf("ResolveConfig() APIKey = %q, want %q", got.APIKey, "OpenAI-Key")
		}
	})

	t.Run("cfg key used when env absent", func(t *testing.T) {
		clearResolverEnv(t)

		got, err := ResolveConfig(configs.ModelConfig{Key: "Cfg-Key"})
		if err != nil {
			t.Fatalf("ResolveConfig() error = %v", err)
		}

		if got.APIKey != "Cfg-Key" {
			t.Fatalf("ResolveConfig() APIKey = %q, want %q", got.APIKey, "Cfg-Key")
		}
	})
}

func TestResolveConfig_AnthropicKeyPriority(t *testing.T) {
	clearResolverEnv(t)
	t.Setenv("MSCLI_PROVIDER", "anthropic")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "token-key")
	t.Setenv("ANTHROPIC_API_KEY", "api-key")

	cfg := configs.ModelConfig{
		Key: "cfg-key",
	}

	got, err := ResolveConfig(cfg)
	if err != nil {
		t.Fatalf("ResolveConfig() error = %v", err)
	}

	if got.APIKey != "token-key" {
		t.Fatalf("ResolveConfig() APIKey = %q, want %q", got.APIKey, "token-key")
	}
}

func TestResolveConfig_AnthropicKeyFallbacks(t *testing.T) {
	t.Run("ANTHROPIC_API_KEY used when auth token absent", func(t *testing.T) {
		clearResolverEnv(t)
		t.Setenv("MSCLI_PROVIDER", "anthropic")
		t.Setenv("ANTHROPIC_API_KEY", "Anthropic-Key")

		got, err := ResolveConfig(configs.ModelConfig{})
		if err != nil {
			t.Fatalf("ResolveConfig() error = %v", err)
		}

		if got.APIKey != "Anthropic-Key" {
			t.Fatalf("ResolveConfig() APIKey = %q, want %q", got.APIKey, "Anthropic-Key")
		}
	})

	t.Run("cfg key used when env absent", func(t *testing.T) {
		clearResolverEnv(t)
		t.Setenv("MSCLI_PROVIDER", "anthropic")

		got, err := ResolveConfig(configs.ModelConfig{Key: "Cfg-Anthropic-Key"})
		if err != nil {
			t.Fatalf("ResolveConfig() error = %v", err)
		}

		if got.APIKey != "Cfg-Anthropic-Key" {
			t.Fatalf("ResolveConfig() APIKey = %q, want %q", got.APIKey, "Cfg-Anthropic-Key")
		}
	})
}

func TestResolveConfig_OpenAIBaseURLPriority(t *testing.T) {
	clearResolverEnv(t)
	t.Setenv("MSCLI_BASE_URL", "HTTPS://MsCli.Example/V1")
	t.Setenv("OPENAI_BASE_URL", "HTTPS://OpenAI.Example/V1")

	cfg := configs.ModelConfig{
		URL: "HTTPS://Cfg.Example/V1",
		Key: "cfg-key",
	}

	got, err := ResolveConfig(cfg)
	if err != nil {
		t.Fatalf("ResolveConfig() error = %v", err)
	}

	if got.BaseURL != "HTTPS://MsCli.Example/V1" {
		t.Fatalf("ResolveConfig() BaseURL = %q, want %q", got.BaseURL, "HTTPS://MsCli.Example/V1")
	}
}

func TestResolveConfig_OpenAIBaseURLFallbacks(t *testing.T) {
	t.Run("OPENAI_BASE_URL used when MSCLI_BASE_URL absent", func(t *testing.T) {
		clearResolverEnv(t)
		t.Setenv("OPENAI_BASE_URL", "HTTPS://OpenAI.Example/V1")

		got, err := ResolveConfig(configs.ModelConfig{Key: "cfg-key"})
		if err != nil {
			t.Fatalf("ResolveConfig() error = %v", err)
		}

		if got.BaseURL != "HTTPS://OpenAI.Example/V1" {
			t.Fatalf("ResolveConfig() BaseURL = %q, want %q", got.BaseURL, "HTTPS://OpenAI.Example/V1")
		}
	})

	t.Run("cfg url used when env absent", func(t *testing.T) {
		clearResolverEnv(t)

		got, err := ResolveConfig(configs.ModelConfig{
			URL: "HTTPS://Cfg.Example/V1",
			Key: "cfg-key",
		})
		if err != nil {
			t.Fatalf("ResolveConfig() error = %v", err)
		}

		if got.BaseURL != "HTTPS://Cfg.Example/V1" {
			t.Fatalf("ResolveConfig() BaseURL = %q, want %q", got.BaseURL, "HTTPS://Cfg.Example/V1")
		}
	})

	t.Run("default openai url used when none set", func(t *testing.T) {
		clearResolverEnv(t)

		got, err := ResolveConfig(configs.ModelConfig{Key: "cfg-key"})
		if err != nil {
			t.Fatalf("ResolveConfig() error = %v", err)
		}

		if got.BaseURL != defaultOpenAIBaseURL {
			t.Fatalf("ResolveConfig() BaseURL = %q, want %q", got.BaseURL, defaultOpenAIBaseURL)
		}
	})
}

func TestResolveConfig_AnthropicBaseURLPriority(t *testing.T) {
	clearResolverEnv(t)
	t.Setenv("MSCLI_PROVIDER", "anthropic")
	t.Setenv("MSCLI_BASE_URL", "HTTPS://MsCli.Example/V1")
	t.Setenv("ANTHROPIC_BASE_URL", "HTTPS://Anthropic.Example/V1")

	cfg := configs.ModelConfig{
		URL: "HTTPS://Cfg.Example/V1",
		Key: "cfg-key",
	}

	got, err := ResolveConfig(cfg)
	if err != nil {
		t.Fatalf("ResolveConfig() error = %v", err)
	}

	if got.BaseURL != "HTTPS://MsCli.Example/V1" {
		t.Fatalf("ResolveConfig() BaseURL = %q, want %q", got.BaseURL, "HTTPS://MsCli.Example/V1")
	}
}

func TestResolveConfig_AnthropicBaseURLFallbacks(t *testing.T) {
	t.Run("ANTHROPIC_BASE_URL used when MSCLI_BASE_URL absent", func(t *testing.T) {
		clearResolverEnv(t)
		t.Setenv("MSCLI_PROVIDER", "anthropic")
		t.Setenv("ANTHROPIC_BASE_URL", "HTTPS://Anthropic.Example/V1")

		got, err := ResolveConfig(configs.ModelConfig{Key: "cfg-key"})
		if err != nil {
			t.Fatalf("ResolveConfig() error = %v", err)
		}

		if got.BaseURL != "HTTPS://Anthropic.Example/V1" {
			t.Fatalf("ResolveConfig() BaseURL = %q, want %q", got.BaseURL, "HTTPS://Anthropic.Example/V1")
		}
	})

	t.Run("cfg url used when env absent", func(t *testing.T) {
		clearResolverEnv(t)
		t.Setenv("MSCLI_PROVIDER", "anthropic")

		got, err := ResolveConfig(configs.ModelConfig{
			URL: "HTTPS://Cfg.Anthropic.Example/V1",
			Key: "cfg-key",
		})
		if err != nil {
			t.Fatalf("ResolveConfig() error = %v", err)
		}

		if got.BaseURL != "HTTPS://Cfg.Anthropic.Example/V1" {
			t.Fatalf("ResolveConfig() BaseURL = %q, want %q", got.BaseURL, "HTTPS://Cfg.Anthropic.Example/V1")
		}
	})

	t.Run("default anthropic url used when none set", func(t *testing.T) {
		clearResolverEnv(t)
		t.Setenv("MSCLI_PROVIDER", "anthropic")

		got, err := ResolveConfig(configs.ModelConfig{Key: "cfg-key"})
		if err != nil {
			t.Fatalf("ResolveConfig() error = %v", err)
		}

		if got.BaseURL != defaultAnthropicBaseURL {
			t.Fatalf("ResolveConfig() BaseURL = %q, want %q", got.BaseURL, defaultAnthropicBaseURL)
		}
	})
}

func TestResolveConfig_AnthropicUsesDefaultURLWithInheritedOpenAIDefault(t *testing.T) {
	clearResolverEnv(t)
	t.Setenv("MSCLI_PROVIDER", "anthropic")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "Anthropic-Key")

	cfg := configs.DefaultConfig()
	cfg.Model.Provider = "anthropic"
	cfg.Model.URL = "HTTPS://API.OPENAI.COM/v1/"

	got, err := ResolveConfig(cfg.Model)
	if err != nil {
		t.Fatalf("ResolveConfig() error = %v", err)
	}

	if got.BaseURL != defaultAnthropicBaseURL {
		t.Fatalf("ResolveConfig() BaseURL = %q, want %q", got.BaseURL, defaultAnthropicBaseURL)
	}
}

func TestResolveConfig_AnthropicKeyPreservesCase(t *testing.T) {
	clearResolverEnv(t)
	t.Setenv("MSCLI_PROVIDER", "anthropic")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "AnThRoPiC-ToKeN")

	got, err := ResolveConfig(configs.ModelConfig{})
	if err != nil {
		t.Fatalf("ResolveConfig() error = %v", err)
	}

	if got.APIKey != "AnThRoPiC-ToKeN" {
		t.Fatalf("ResolveConfig() APIKey = %q, want %q", got.APIKey, "AnThRoPiC-ToKeN")
	}
}

func TestResolveConfig_AnthropicBaseURLPreservesCase(t *testing.T) {
	clearResolverEnv(t)
	t.Setenv("MSCLI_PROVIDER", "anthropic")
	t.Setenv("ANTHROPIC_BASE_URL", "HTTPS://Anthropic.Example/Path/V1")

	got, err := ResolveConfig(configs.ModelConfig{Key: "cfg-key"})
	if err != nil {
		t.Fatalf("ResolveConfig() error = %v", err)
	}

	if got.BaseURL != "HTTPS://Anthropic.Example/Path/V1" {
		t.Fatalf("ResolveConfig() BaseURL = %q, want %q", got.BaseURL, "HTTPS://Anthropic.Example/Path/V1")
	}
}

func TestResolveConfig_AnthropicHeaderConflict(t *testing.T) {
	clearResolverEnv(t)
	t.Setenv("MSCLI_PROVIDER", "anthropic")

	cfg := configs.ModelConfig{
		Key: "cfg-key",
		Headers: map[string]string{
			"X-API-KEY":         "user-key",
			"Anthropic-Version": "2024-01-01",
			"X-Trace-ID":        "trace-123",
		},
	}

	got, err := ResolveConfig(cfg)
	if err != nil {
		t.Fatalf("ResolveConfig() error = %v", err)
	}

	if got.AuthHeaderName != "x-api-key" {
		t.Fatalf("ResolveConfig() AuthHeaderName = %q, want %q", got.AuthHeaderName, "x-api-key")
	}

	if got.Headers["x-api-key"] != "cfg-key" {
		t.Fatalf("ResolveConfig() x-api-key = %q, want %q", got.Headers["x-api-key"], "cfg-key")
	}

	if got.Headers["anthropic-version"] != "2023-06-01" {
		t.Fatalf("ResolveConfig() anthropic-version = %q, want %q", got.Headers["anthropic-version"], "2023-06-01")
	}

	if got.Headers["X-Trace-ID"] != "trace-123" {
		t.Fatalf("ResolveConfig() X-Trace-ID = %q, want %q", got.Headers["X-Trace-ID"], "trace-123")
	}
}

func TestResolveConfig_OpenAIHeaderConflict(t *testing.T) {
	clearResolverEnv(t)
	t.Setenv("MSCLI_PROVIDER", "openai")

	got, err := ResolveConfig(configs.ModelConfig{
		Key: "cfg-key",
		Headers: map[string]string{
			"authorization": "Basic user-secret",
			"X-Trace-ID":    "trace-123",
		},
	})
	if err != nil {
		t.Fatalf("ResolveConfig() error = %v", err)
	}

	if got.AuthHeaderName != "Authorization" {
		t.Fatalf("ResolveConfig() AuthHeaderName = %q, want %q", got.AuthHeaderName, "Authorization")
	}

	if got.Headers["Authorization"] != "Bearer cfg-key" {
		t.Fatalf("ResolveConfig() Authorization = %q, want %q", got.Headers["Authorization"], "Bearer cfg-key")
	}

	if got.Headers["X-Trace-ID"] != "trace-123" {
		t.Fatalf("ResolveConfig() X-Trace-ID = %q, want %q", got.Headers["X-Trace-ID"], "trace-123")
	}
}

func TestResolveConfig_MissingAPIKeyWrapsSentinel(t *testing.T) {
	clearResolverEnv(t)

	_, err := ResolveConfig(configs.ModelConfig{})
	if err == nil {
		t.Fatal("ResolveConfig() error = nil, want missing api key error")
	}

	if !errors.Is(err, ErrMissingAPIKey) {
		t.Fatalf("ResolveConfig() error = %v, want ErrMissingAPIKey", err)
	}
}

func TestResolveConfig_AnthropicAcceptsExplicitKeyMatchingOpenAIEnv(t *testing.T) {
	clearResolverEnv(t)
	t.Setenv("MSCLI_PROVIDER", "anthropic")
	t.Setenv("OPENAI_API_KEY", "Shared-Key")

	got, err := ResolveConfig(configs.ModelConfig{
		Provider: "anthropic",
		Key:      "Shared-Key",
	})
	if err != nil {
		t.Fatalf("ResolveConfig() error = %v", err)
	}

	if got.APIKey != "Shared-Key" {
		t.Fatalf("ResolveConfig() APIKey = %q, want %q", got.APIKey, "Shared-Key")
	}
}

func TestResolveConfig_AnthropicIgnoresOpenAIEnvFallbacksFromApplyEnvOverrides(t *testing.T) {
	clearResolverEnv(t)
	t.Setenv("MSCLI_PROVIDER", "anthropic")
	t.Setenv("OPENAI_API_KEY", "OpenAI-Key")
	t.Setenv("OPENAI_BASE_URL", "HTTPS://OpenAI.Example/V1")

	cfg := configs.DefaultConfig()
	configs.ApplyEnvOverrides(cfg)

	got, err := ResolveConfig(cfg.Model)
	if err == nil {
		t.Fatal("ResolveConfig() error = nil, want missing api key error")
	}
	if !errors.Is(err, ErrMissingAPIKey) {
		t.Fatalf("ResolveConfig() error = %v, want ErrMissingAPIKey", err)
	}

	_ = got
}

func TestResolveConfig_AnthropicIgnoresOpenAIBaseURLFallbackFromApplyEnvOverrides(t *testing.T) {
	clearResolverEnv(t)
	t.Setenv("MSCLI_PROVIDER", "anthropic")
	t.Setenv("OPENAI_BASE_URL", "HTTPS://OpenAI.Example/V1")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "Anthropic-Key")

	cfg := configs.DefaultConfig()
	configs.ApplyEnvOverrides(cfg)

	got, err := ResolveConfig(cfg.Model)
	if err != nil {
		t.Fatalf("ResolveConfig() error = %v", err)
	}

	if got.BaseURL != defaultAnthropicBaseURL {
		t.Fatalf("ResolveConfig() BaseURL = %q, want %q", got.BaseURL, defaultAnthropicBaseURL)
	}

	if got.APIKey != "Anthropic-Key" {
		t.Fatalf("ResolveConfig() APIKey = %q, want %q", got.APIKey, "Anthropic-Key")
	}
}

func TestResolveConfigWithOptions_PreferConfigAPIKey(t *testing.T) {
	clearResolverEnv(t)
	t.Setenv("MSCLI_API_KEY", "env-key")

	got, err := ResolveConfigWithOptions(configs.ModelConfig{
		Model: "gpt-4o-mini",
		Key:   "cfg-key",
	}, ResolveOptions{
		PreferConfigAPIKey: true,
	})
	if err != nil {
		t.Fatalf("ResolveConfigWithOptions() error = %v", err)
	}

	if got.APIKey != "cfg-key" {
		t.Fatalf("ResolveConfigWithOptions() APIKey = %q, want %q", got.APIKey, "cfg-key")
	}
}

func TestResolveConfigWithOptions_PreferConfigBaseURL(t *testing.T) {
	clearResolverEnv(t)
	t.Setenv("MSCLI_BASE_URL", "https://env.example/v1")

	got, err := ResolveConfigWithOptions(configs.ModelConfig{
		Model: "gpt-4o-mini",
		Key:   "cfg-key",
		URL:   "https://cfg.example/v1",
	}, ResolveOptions{
		PreferConfigBaseURL: true,
	})
	if err != nil {
		t.Fatalf("ResolveConfigWithOptions() error = %v", err)
	}

	if got.BaseURL != "https://cfg.example/v1" {
		t.Fatalf("ResolveConfigWithOptions() BaseURL = %q, want %q", got.BaseURL, "https://cfg.example/v1")
	}
}

func clearResolverEnv(t *testing.T) {
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
