package provider

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/vigo999/ms-cli/configs"
)

const (
	defaultOpenAIBaseURL    = "https://api.openai.com/v1"
	defaultAnthropicBaseURL = "https://api.anthropic.com"
	anthropicVersionHeader  = "2023-06-01"
)

// ErrMissingAPIKey indicates the resolved provider configuration has no API key.
var ErrMissingAPIKey = errors.New("missing api key")

// ResolveConfig resolves provider, base URL, API key, and required headers
// from explicit environment and config precedence rules.
func ResolveConfig(cfg configs.ModelConfig) (ResolvedConfig, error) {
	return ResolveConfigWithOptions(cfg, ResolveOptions{})
}

// ResolveConfigWithOptions resolves provider configuration using optional
// precedence overrides for explicit runtime inputs.
func ResolveConfigWithOptions(cfg configs.ModelConfig, opts ResolveOptions) (ResolvedConfig, error) {
	kind, err := resolveProviderKind(cfg.Provider)
	if err != nil {
		return ResolvedConfig{}, err
	}

	apiKey, err := resolveAPIKey(kind, cfg.Key, opts)
	if err != nil {
		return ResolvedConfig{}, err
	}

	baseURL := resolveBaseURL(kind, cfg.URL, opts)
	if baseURL == "" {
		baseURL = defaultBaseURL(kind)
	}

	requiredHeaders, authHeaderName := requiredHeadersFor(kind, apiKey)

	return ResolvedConfig{
		Kind:           kind,
		Model:          strings.TrimSpace(cfg.Model),
		BaseURL:        baseURL,
		APIKey:         apiKey,
		AuthHeaderName: authHeaderName,
		Headers:        mergeHeaders(requiredHeaders, cfg.Headers),
		Timeout:        resolvedTimeout(cfg.TimeoutSec),
	}, nil
}

func resolveProviderKind(cfgProvider string) (ProviderKind, error) {
	if raw := normalizedProviderEnv("MSCLI_PROVIDER"); raw != "" {
		return parseProviderKind(raw)
	}
	if raw := NormalizeProvider(cfgProvider); raw != "" {
		return parseProviderKind(raw)
	}
	return ProviderOpenAICompatible, nil
}

func resolveAPIKey(kind ProviderKind, cfgKey string, opts ResolveOptions) (string, error) {
	if opts.PreferConfigAPIKey {
		if raw := strings.TrimSpace(cfgKey); raw != "" {
			return raw, nil
		}
	}

	switch kind {
	case ProviderAnthropic:
		if raw := trimmedEnv("ANTHROPIC_AUTH_TOKEN"); raw != "" {
			return raw, nil
		}
		if raw := trimmedEnv("ANTHROPIC_API_KEY"); raw != "" {
			return raw, nil
		}
	default:
		if raw := trimmedEnv("MSCLI_API_KEY"); raw != "" {
			return raw, nil
		}
		if raw := trimmedEnv("OPENAI_API_KEY"); raw != "" {
			return raw, nil
		}
	}

	if raw := strings.TrimSpace(cfgKey); raw != "" {
		return raw, nil
	}

	return "", fmt.Errorf("%w for provider %s", ErrMissingAPIKey, kind)
}

func resolveBaseURL(kind ProviderKind, cfgURL string, opts ResolveOptions) string {
	if opts.PreferConfigBaseURL {
		if raw := strings.TrimSpace(cfgURL); raw != "" {
			return raw
		}
	}

	switch kind {
	case ProviderAnthropic:
		if raw := trimmedEnv("MSCLI_BASE_URL"); raw != "" {
			return raw
		}
		if raw := trimmedEnv("ANTHROPIC_BASE_URL"); raw != "" {
			return raw
		}
		if raw := strings.TrimSpace(cfgURL); raw != "" && normalizeURLForComparison(raw) != normalizeURLForComparison(defaultOpenAIBaseURL) {
			return raw
		}
		return defaultAnthropicBaseURL
	default:
		if raw := trimmedEnv("MSCLI_BASE_URL"); raw != "" {
			return raw
		}
		if raw := trimmedEnv("OPENAI_BASE_URL"); raw != "" {
			return raw
		}
		if raw := strings.TrimSpace(cfgURL); raw != "" {
			return raw
		}
		return defaultOpenAIBaseURL
	}
}

func requiredHeadersFor(kind ProviderKind, apiKey string) (map[string]string, string) {
	switch kind {
	case ProviderAnthropic:
		return map[string]string{
			"x-api-key":         apiKey,
			"anthropic-version": anthropicVersionHeader,
		}, "x-api-key"
	default:
		return map[string]string{
			"Authorization": "Bearer " + apiKey,
		}, "Authorization"
	}
}

func mergeHeaders(required, user map[string]string) map[string]string {
	if len(required) == 0 && len(user) == 0 {
		return map[string]string{}
	}

	result := make(map[string]string, len(required)+len(user))
	requiredKeys := make(map[string]struct{}, len(required))
	for key := range required {
		requiredKeys[strings.ToLower(strings.TrimSpace(key))] = struct{}{}
	}

	for key, value := range user {
		trimmedKey := strings.TrimSpace(key)
		if trimmedKey == "" {
			continue
		}
		if _, conflict := requiredKeys[strings.ToLower(trimmedKey)]; conflict {
			continue
		}
		result[trimmedKey] = value
	}

	for key, value := range required {
		result[key] = value
	}

	return result
}

func defaultBaseURL(kind ProviderKind) string {
	if kind == ProviderAnthropic {
		return defaultAnthropicBaseURL
	}
	return defaultOpenAIBaseURL
}

func resolvedTimeout(timeoutSec int) time.Duration {
	if timeoutSec <= 0 {
		return 180 * time.Second
	}
	return time.Duration(timeoutSec) * time.Second
}

func normalizedProviderEnv(key string) string {
	return normalizeProviderName(os.Getenv(key))
}

func trimmedEnv(key string) string {
	return strings.TrimSpace(os.Getenv(key))
}

func normalizeProviderName(v string) string {
	return NormalizeProvider(v)
}

// NormalizeProvider canonicalizes provider input for comparisons.
func NormalizeProvider(v string) string {
	return strings.ToLower(strings.TrimSpace(v))
}

// IsSupportedProvider reports whether the provider identifier is one of the
// explicitly supported provider kinds.
func IsSupportedProvider(v string) bool {
	switch NormalizeProvider(v) {
	case string(ProviderOpenAI), string(ProviderOpenAICompatible), string(ProviderAnthropic):
		return true
	default:
		return false
	}
}

func normalizeURLForComparison(v string) string {
	return strings.ToLower(strings.TrimRight(strings.TrimSpace(v), "/"))
}

func parseProviderKind(raw string) (ProviderKind, error) {
	switch normalizeProviderName(raw) {
	case string(ProviderOpenAI):
		return ProviderOpenAI, nil
	case string(ProviderOpenAICompatible):
		return ProviderOpenAICompatible, nil
	case string(ProviderAnthropic):
		return ProviderAnthropic, nil
	default:
		return "", fmt.Errorf("unsupported provider %q", strings.TrimSpace(raw))
	}
}
