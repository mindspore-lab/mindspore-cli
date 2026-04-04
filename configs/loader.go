package configs

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

// LoadWithEnv loads built-in defaults and applies environment variable overrides.
func LoadWithEnv() (*Config, error) {
	cfg := DefaultConfig()
	defaultContextWindow := cfg.Context.Window

	// model-derived context defaults > env overrides > built-in defaults
	applyModelTokenDefaults(cfg, defaultContextWindow)
	previousModel := cfg.Model.Model
	ApplyEnvOverrides(cfg)
	RefreshModelTokenDefaults(cfg, previousModel)
	cfg.normalize()

	// Validate
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("validate config: %w", err)
	}

	return cfg, nil
}

// ApplyEnvOverrides applies environment variable overrides to the config.
// Unified MSCLI_* overrides are applied on top of built-in defaults.
func ApplyEnvOverrides(cfg *Config) {
	// Model settings
	if v := strings.TrimSpace(os.Getenv("MSCLI_MODEL")); v != "" {
		cfg.Model.Model = v
	}
	if v := strings.TrimSpace(os.Getenv("MSCLI_API_KEY")); v != "" {
		cfg.Model.Key = v
	}
	if v := strings.TrimSpace(os.Getenv("MSCLI_BASE_URL")); v != "" {
		cfg.Model.URL = v
	}
	if v := strings.TrimSpace(os.Getenv("MSCLI_PROVIDER")); v != "" {
		cfg.Model.Provider = v
	}
	if v := os.Getenv("MSCLI_TEMPERATURE"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			cfg.Request.Temperature = &f
		}
	}
	if v := os.Getenv("MSCLI_MAX_TOKENS"); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			cfg.Request.MaxTokens = &i
		}
	}
	if v := os.Getenv("MSCLI_MAX_ITERATIONS"); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			cfg.Request.MaxIterations = &i
		}
	}
	if v := os.Getenv("MSCLI_TIMEOUT"); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			cfg.Model.TimeoutSec = i
		}
	}

	// UI settings
	if v := os.Getenv("MSCLI_UI_ENABLED"); v != "" {
		if b, err := strconv.ParseBool(v); err == nil {
			cfg.UI.Enabled = b
		}
	}
	if v := os.Getenv("MSCLI_THEME"); v != "" {
		cfg.UI.Theme = v
	}

	// Permissions
	if v := os.Getenv("MSCLI_PERMISSIONS_SKIP"); v != "" {
		if b, err := strconv.ParseBool(v); err == nil {
			cfg.Permissions.SkipRequests = b
		}
	}
	if v := os.Getenv("MSCLI_PERMISSIONS_DEFAULT"); v != "" {
		cfg.Permissions.DefaultLevel = v
	}

	// Context settings
	if v := os.Getenv("MSCLI_CONTEXT_WINDOW"); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			cfg.Context.Window = i
		}
	}
	if v := os.Getenv("MSCLI_CONTEXT_RESERVE"); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			cfg.Context.ReserveTokens = i
		}
	}

	// Issues server
	if v := strings.TrimSpace(os.Getenv("MSCLI_SERVER_URL")); v != "" {
		cfg.Server.URL = v
	}

	// Memory settings
	if v := os.Getenv("MSCLI_MEMORY_ENABLED"); v != "" {
		if b, err := strconv.ParseBool(v); err == nil {
			cfg.Memory.Enabled = b
		}
	}
	if v := os.Getenv("MSCLI_MEMORY_PATH"); v != "" {
		cfg.Memory.StorePath = v
	}
}

// StringSliceEnv splits an environment variable by comma.
func StringSliceEnv(key string) []string {
	v := os.Getenv(key)
	if v == "" {
		return nil
	}
	parts := strings.Split(v, ",")
	for i, p := range parts {
		parts[i] = strings.TrimSpace(p)
	}
	return parts
}
