// Package configs provides configuration management for mscli.
package configs

import (
	"fmt"
	"strings"
)

// Config holds the complete application configuration.
type Config struct {
	Model         ModelConfig                  `yaml:"model"`
	ModelProfiles map[string]ModelTokenProfile `yaml:"model_profiles,omitempty"`
	Request       RequestConfig                `yaml:"-"`
	UI            UIConfig                     `yaml:"ui"`
	Permissions   PermissionsConfig            `yaml:"permissions"`
	Context       ContextConfig                `yaml:"context"`
	Memory        MemoryConfig                 `yaml:"memory"`
	Execution     ExecutionConfig              `yaml:"execution"`
	Server        RemoteConfig                 `yaml:"server"`
}

// RemoteConfig holds the client-side server connection config.
type RemoteConfig struct {
	URL       string `yaml:"url,omitempty"`
	TokenPath string `yaml:"token_path,omitempty"`
}

const DefaultServerURL = "https://mscli.dev"
const DefaultRequestMaxIterations = 100

func (c *Config) normalize() {
	if strings.TrimSpace(c.Model.Provider) == "" {
		c.Model.Provider = "openai-completion"
	}
}

// ModelConfig holds the LLM model configuration.
type ModelConfig struct {
	URL        string            `yaml:"url,omitempty"`
	Key        string            `yaml:"key,omitempty"`
	Provider   string            `yaml:"provider,omitempty"`
	Model      string            `yaml:"model"`
	TimeoutSec int               `yaml:"timeout_sec"`
	Headers    map[string]string `yaml:"headers,omitempty"`
}

// RequestConfig holds optional per-request overrides sourced from env.
type RequestConfig struct {
	Temperature   *float64 `yaml:"-"`
	MaxTokens     *int     `yaml:"-"`
	MaxIterations *int     `yaml:"-"`
}

// UIConfig holds the UI configuration.
type UIConfig struct {
	Enabled      bool   `yaml:"enabled"`
	Theme        string `yaml:"theme,omitempty"`
	ShowTokenBar bool   `yaml:"show_token_bar"`
	Animation    bool   `yaml:"animation"`
}

// PermissionsConfig holds the permission control configuration.
type PermissionsConfig struct {
	SkipRequests bool              `yaml:"skip_requests"`
	DefaultLevel string            `yaml:"default_level"`
	Allow        []string          `yaml:"allow,omitempty"`
	Ask          []string          `yaml:"ask,omitempty"`
	Deny         []string          `yaml:"deny,omitempty"`
	ToolPolicies map[string]string `yaml:"tool_policies,omitempty"`
	AllowedTools []string          `yaml:"allowed_tools"`
	BlockedTools []string          `yaml:"blocked_tools,omitempty"`
	RuleSources  map[string]string `yaml:"-"`
}

// ContextConfig holds the context management configuration.
type ContextConfig struct {
	Window              int     `yaml:"window"`
	ReserveTokens       int     `yaml:"reserve_tokens"`
	CompactionThreshold float64 `yaml:"compaction_threshold"`
}

// MemoryConfig holds the memory system configuration.
type MemoryConfig struct {
	Enabled   bool   `yaml:"enabled"`
	StorePath string `yaml:"store_path,omitempty"`
	MaxItems  int    `yaml:"max_items"`
	MaxBytes  int64  `yaml:"max_bytes"`
	TTLHours  int    `yaml:"ttl_hours"`
}

// ExecutionConfig holds the execution configuration.
type ExecutionConfig struct {
	Mode           string       `yaml:"mode"`
	TimeoutSec     int          `yaml:"timeout_sec"`
	MaxConcurrency int          `yaml:"max_concurrency"`
	Docker         DockerConfig `yaml:"docker,omitempty"`
}

// DockerConfig holds the Docker execution configuration.
type DockerConfig struct {
	Image   string            `yaml:"image"`
	CPU     string            `yaml:"cpu"`
	Memory  string            `yaml:"memory"`
	Network string            `yaml:"network"`
	Env     map[string]string `yaml:"env,omitempty"`
}

// DefaultConfig returns a configuration with default values.
func DefaultConfig() *Config {
	defaultMaxIterations := DefaultRequestMaxIterations
	cfg := &Config{
		Model: ModelConfig{
			URL:        "https://api.openai.com/v1",
			Provider:   "openai-completion",
			Model:      "",
			TimeoutSec: 300, // 5 minutes for longer conversations
			Headers:    make(map[string]string),
		},
		UI: UIConfig{
			Enabled:      true,
			Theme:        "default",
			ShowTokenBar: true,
			Animation:    true,
		},
		Permissions: PermissionsConfig{
			SkipRequests: false,
			DefaultLevel: "ask",
			Allow:        []string{},
			Ask:          []string{},
			Deny:         []string{},
			ToolPolicies: make(map[string]string),
			AllowedTools: []string{},
			BlockedTools: []string{},
			RuleSources:  map[string]string{},
		},
		Context: ContextConfig{
			Window:              200000,
			ReserveTokens:       4000,
			CompactionThreshold: 0.9,
		},
		Memory: MemoryConfig{
			Enabled:   true,
			StorePath: "",
			MaxItems:  200,
			MaxBytes:  2 * 1024 * 1024, // 2MB
			TTLHours:  168,             // 7 days
		},
		ModelProfiles: make(map[string]ModelTokenProfile),
		Request: RequestConfig{
			MaxIterations: &defaultMaxIterations,
		},
		Server: RemoteConfig{
			URL: DefaultServerURL,
		},
		Execution: ExecutionConfig{
			Mode:           "local",
			TimeoutSec:     1800,
			MaxConcurrency: 2,
			Docker: DockerConfig{
				Image:   "ubuntu:22.04",
				CPU:     "2",
				Memory:  "4g",
				Network: "none",
				Env:     make(map[string]string),
			},
		},
	}
	cfg.normalize()
	return cfg
}

// Validate validates the configuration.
func (c *Config) Validate() error {
	c.normalize()

	if provider := strings.ToLower(strings.TrimSpace(c.Model.Provider)); provider != "" {
		switch provider {
		case "openai-completion", "openai-responses", "anthropic":
		default:
			return fmt.Errorf("unsupported provider %q", strings.TrimSpace(c.Model.Provider))
		}
	}

	if c.Request.Temperature != nil {
		if *c.Request.Temperature < 0 || *c.Request.Temperature > 2 {
			return fmt.Errorf("temperature must be between 0 and 2")
		}
	}

	if c.Request.MaxTokens != nil && *c.Request.MaxTokens < 0 {
		return fmt.Errorf("max_tokens must be non-negative")
	}

	if c.Request.MaxIterations != nil && *c.Request.MaxIterations < 0 {
		return fmt.Errorf("max_iterations must be non-negative")
	}

	if c.Context.Window < c.Context.ReserveTokens {
		return fmt.Errorf("window must be greater than reserve_tokens")
	}

	return nil
}

// Merge merges another config into this one (overwriting values).
func (c *Config) Merge(other *Config) {
	if other.Model.URL != "" {
		c.Model.URL = other.Model.URL
	}
	if other.Model.Key != "" {
		c.Model.Key = other.Model.Key
	}
	if other.Model.Provider != "" {
		c.Model.Provider = other.Model.Provider
	}
	if other.Model.Model != "" {
		c.Model.Model = other.Model.Model
	}
	if other.Model.TimeoutSec != 0 {
		c.Model.TimeoutSec = other.Model.TimeoutSec
	}
	if len(other.Model.Headers) > 0 {
		c.Model.Headers = other.Model.Headers
	}

	if len(other.ModelProfiles) > 0 {
		c.ModelProfiles = other.ModelProfiles
	}

	if other.Request.Temperature != nil {
		v := *other.Request.Temperature
		c.Request.Temperature = &v
	}
	if other.Request.MaxTokens != nil {
		v := *other.Request.MaxTokens
		c.Request.MaxTokens = &v
	}
	if other.Request.MaxIterations != nil {
		v := *other.Request.MaxIterations
		c.Request.MaxIterations = &v
	}

	if other.Context.Window != 0 {
		c.Context.Window = other.Context.Window
	}
	if other.Context.ReserveTokens != 0 {
		c.Context.ReserveTokens = other.Context.ReserveTokens
	}
	if other.Context.CompactionThreshold != 0 {
		c.Context.CompactionThreshold = other.Context.CompactionThreshold
	}
}

// UnmarshalYAML supports both context.window and legacy context.max_tokens.
func (c *ContextConfig) UnmarshalYAML(unmarshal func(interface{}) error) error {
	type rawContextConfig struct {
		Window              int     `yaml:"window"`
		LegacyMaxTokens     int     `yaml:"max_tokens"`
		ReserveTokens       int     `yaml:"reserve_tokens"`
		CompactionThreshold float64 `yaml:"compaction_threshold"`
	}

	var raw rawContextConfig
	if err := unmarshal(&raw); err != nil {
		return err
	}

	c.Window = raw.Window
	if c.Window == 0 {
		c.Window = raw.LegacyMaxTokens
	}
	c.ReserveTokens = raw.ReserveTokens
	c.CompactionThreshold = raw.CompactionThreshold

	return nil
}
