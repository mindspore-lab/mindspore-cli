package configs

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// LoadWithEnv loads configuration from file and applies environment variable overrides.
func LoadWithEnv() (*Config, error) {
	cfg := DefaultConfig()

	// Auto-generate user config on first run if it doesn't exist.
	ensureUserConfig(cfg)

	// Fixed config layers: defaults -> user -> project -> env.
	if err := mergeConfigFile(cfg, userConfigPath()); err != nil {
		return nil, err
	}

	projectPath := filepath.Join(".ms-cli", "config.yaml")
	if err := mergeConfigFile(cfg, projectPath); err != nil {
		return nil, err
	}

	// ENV > project > user > default
	ApplyEnvOverrides(cfg)
	cfg.normalize()

	// Validate
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("validate config: %w", err)
	}

	return cfg, nil
}

// ApplyEnvOverrides applies environment variable overrides to the config.
// Config-layer precedence uses unified MSCLI_* overrides.
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
			cfg.Model.Temperature = f
		}
	}
	if v := os.Getenv("MSCLI_MAX_TOKENS"); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			cfg.Model.MaxTokens = i
		}
	}
	if v := os.Getenv("MSCLI_TIMEOUT"); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			cfg.Model.TimeoutSec = i
		}
	}

	// Budget settings
	if v := os.Getenv("MSCLI_BUDGET_TOKENS"); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			cfg.Budget.MaxTokens = i
		}
	}
	if v := os.Getenv("MSCLI_BUDGET_COST"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			cfg.Budget.MaxCostUSD = f
		}
	}

	// UI settings
	if v := os.Getenv("MSCLI_UI_ENABLED"); v != "" {
		if b, err := strconv.ParseBool(v); err == nil {
			cfg.UI.Enabled = b
		}
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
			cfg.Context.MaxTokens = i
		}
	}
	if v := os.Getenv("MSCLI_CONTEXT_RESERVE"); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			cfg.Context.ReserveTokens = i
		}
	}

	// Issues server
	if v := strings.TrimSpace(os.Getenv("MSCLI_SERVER_URL")); v != "" {
		cfg.Issues.ServerURL = v
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

// SaveToFile saves the configuration to a YAML file.
func SaveToFile(cfg *Config, path string) error {
	if path == "" {
		return fmt.Errorf("config path is required")
	}

	// Expand home directory
	if strings.HasPrefix(path, "~/") {
		home, err := os.UserHomeDir()
		if err == nil {
			path = filepath.Join(home, path[2:])
		}
	}

	// Ensure directory exists
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("create config directory: %w", err)
	}

	data, err := yaml.Marshal(cfg)
	if err != nil {
		return fmt.Errorf("marshal config: %w", err)
	}

	if err := os.WriteFile(path, data, 0600); err != nil {
		return fmt.Errorf("write config file: %w", err)
	}

	return nil
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

func ensureUserConfig(cfg *Config) {
	path := userConfigPath()
	if path == "" {
		return
	}
	if _, err := os.Stat(path); err == nil {
		return // already exists
	}
	_ = SaveToFile(cfg, path)
}

func userConfigPath() string {
	home, err := os.UserHomeDir()
	if err != nil || strings.TrimSpace(home) == "" {
		return ""
	}
	return filepath.Join(home, ".ms-cli", "config.yaml")
}

func mergeConfigFile(cfg *Config, path string) error {
	path = strings.TrimSpace(path)
	if path == "" {
		return nil
	}

	if strings.HasPrefix(path, "~/") {
		home, err := os.UserHomeDir()
		if err == nil {
			path = filepath.Join(home, path[2:])
		}
	}

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("read config file %q: %w", path, err)
	}
	if err := yaml.Unmarshal(data, cfg); err != nil {
		return fmt.Errorf("parse config file %q: %w", path, err)
	}

	return nil
}
