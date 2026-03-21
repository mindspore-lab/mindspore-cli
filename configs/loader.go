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

	// Fixed config layers: defaults -> managed -> user -> project -> env.
	if err := mergeConfigFile(cfg, managedConfigPath()); err != nil {
		return nil, err
	}

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
	cfg.Permissions.RuleSources = derivePermissionRuleSources(cfg, managedConfigPath(), userConfigPath(), projectPath)

	// Validate
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("validate config: %w", err)
	}

	return cfg, nil
}

func derivePermissionRuleSources(cfg *Config, managedPath, userPath, projectPath string) map[string]string {
	result := make(map[string]string)
	managedBuckets := readPermissionBuckets(managedPath)
	userBuckets := readPermissionBuckets(userPath)
	projectBuckets := readPermissionBuckets(projectPath)

	assign := func(rules []string, source string) {
		for _, rule := range rules {
			r := strings.TrimSpace(rule)
			if r == "" {
				continue
			}
			result[r] = source
		}
	}

	assign(managedBuckets.Allow, "managed")
	assign(managedBuckets.Ask, "managed")
	assign(managedBuckets.Deny, "managed")
	assign(userBuckets.Allow, "user")
	assign(userBuckets.Ask, "user")
	assign(userBuckets.Deny, "user")
	assign(projectBuckets.Allow, "project")
	assign(projectBuckets.Ask, "project")
	assign(projectBuckets.Deny, "project")

	for _, rule := range cfg.Permissions.Allow {
		r := strings.TrimSpace(rule)
		if r == "" {
			continue
		}
		if _, ok := result[r]; !ok {
			result[r] = "config"
		}
	}
	for _, rule := range cfg.Permissions.Ask {
		r := strings.TrimSpace(rule)
		if r == "" {
			continue
		}
		if _, ok := result[r]; !ok {
			result[r] = "config"
		}
	}
	for _, rule := range cfg.Permissions.Deny {
		r := strings.TrimSpace(rule)
		if r == "" {
			continue
		}
		if _, ok := result[r]; !ok {
			result[r] = "config"
		}
	}
	return result
}

func managedConfigPath() string {
	path := strings.TrimSpace(os.Getenv("MSCLI_MANAGED_CONFIG"))
	if path == "" {
		return ""
	}
	return path
}

type permissionBuckets struct {
	Allow []string
	Ask   []string
	Deny  []string
}

func readPermissionBuckets(path string) permissionBuckets {
	path = strings.TrimSpace(path)
	if path == "" {
		return permissionBuckets{}
	}
	if strings.HasPrefix(path, "~/") {
		home, err := os.UserHomeDir()
		if err == nil {
			path = filepath.Join(home, path[2:])
		}
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return permissionBuckets{}
	}

	var raw map[string]any
	if err := yaml.Unmarshal(data, &raw); err != nil {
		return permissionBuckets{}
	}
	permsRaw, ok := raw["permissions"].(map[string]any)
	if !ok {
		return permissionBuckets{}
	}
	return permissionBuckets{
		Allow: anyStringSlice(permsRaw["allow"]),
		Ask:   anyStringSlice(permsRaw["ask"]),
		Deny:  anyStringSlice(permsRaw["deny"]),
	}
}

func anyStringSlice(v any) []string {
	items, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]string, 0, len(items))
	for _, item := range items {
		if s, ok := item.(string); ok {
			s = strings.TrimSpace(s)
			if s != "" {
				out = append(out, s)
			}
		}
	}
	return out
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
	if v := os.Getenv("MSCLI_PERMISSIONS_MODE"); v != "" {
		cfg.Permissions.DefaultMode = v
	}

	// Context settings
	if v := os.Getenv("MSCLI_CONTEXT_MAX"); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			cfg.Context.MaxTokens = i
		}
	}
	if v := os.Getenv("MSCLI_CONTEXT_RESERVE"); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			cfg.Context.ReserveTokens = i
		}
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
