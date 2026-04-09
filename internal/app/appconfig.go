package app

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
)

const (
	modelModeMSCLIProvided = "mscli-provided"
	modelModeOwn           = "own"
	modelModeOwnEnv        = "own-env"
	modelSetupToken        = "__model_setup"
	connectProviderToken   = "__connect_provider__"
	selectModelToken       = "__select_model__"
	deleteProviderToken    = "__delete_provider__"
)

// appConfig holds persistent local settings stored in ~/.mscli/config.json.
// Separate from credentials.json (issue server auth) and configs/ (YAML + env).
type appConfig struct {
	ModelMode      string                `json:"model_mode,omitempty"`      // "mscli-provided" or "own" or ""
	ModelPresetID  string                `json:"model_preset_id,omitempty"` // e.g. "kimi-k2.5-free"
	ModelToken     string                `json:"model_token,omitempty"`     // API token for mscli-provided models
	ExtraProviders []extraProviderConfig `json:"extra_providers,omitempty"`
}

type extraProviderConfig struct {
	ID       string                     `json:"id"`
	Label    string                     `json:"label,omitempty"`
	BaseURL  string                     `json:"base_url"`
	Protocol string                     `json:"protocol"`
	Models   []extraProviderModelConfig `json:"models,omitempty"`
}

type extraProviderModelConfig struct {
	ID    string `json:"id"`
	Label string `json:"label,omitempty"`
}

// appConfigPathOverride allows tests to redirect the config path.
var appConfigPathOverride string

func appConfigPath() string {
	if appConfigPathOverride != "" {
		return appConfigPathOverride
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".mscli", "config.json")
}

func loadAppConfig() (*appConfig, error) {
	data, err := os.ReadFile(appConfigPath())
	if err != nil {
		if os.IsNotExist(err) {
			return &appConfig{}, nil
		}
		return nil, err
	}
	var cfg appConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	cfg.normalize()
	return &cfg, nil
}

func saveAppConfig(cfg *appConfig) error {
	path := appConfigPath()
	cfg.normalize()
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o600)
}

func (c *appConfig) normalize() {
	if c == nil {
		return
	}
	if len(c.ExtraProviders) == 0 {
		c.ExtraProviders = nil
		return
	}

	out := make([]extraProviderConfig, 0, len(c.ExtraProviders))
	for _, provider := range c.ExtraProviders {
		provider.ID = strings.TrimSpace(provider.ID)
		provider.Label = strings.TrimSpace(provider.Label)
		provider.BaseURL = strings.TrimSpace(provider.BaseURL)
		provider.Protocol = strings.TrimSpace(provider.Protocol)
		if provider.ID == "" {
			continue
		}

		models := make([]extraProviderModelConfig, 0, len(provider.Models))
		for _, model := range provider.Models {
			model.ID = strings.TrimSpace(model.ID)
			model.Label = strings.TrimSpace(model.Label)
			if model.ID == "" {
				continue
			}
			models = append(models, model)
		}
		provider.Models = models
		out = append(out, provider)
	}
	c.ExtraProviders = out
}
