package app

import (
	"encoding/json"
	"os"
	"path/filepath"
)

const (
	modelModeMSCLIProvided = "mscli-provided"
	modelModeOwn            = "own"
	modelModeOwnEnv         = "own-env"
	modelSetupToken         = "__model_setup"
)

// appConfig holds persistent local settings stored in ~/.mscli/config.json.
// Separate from credentials.json (issue server auth) and configs/ (YAML + env).
type appConfig struct {
	ModelMode     string `json:"model_mode,omitempty"`      // "mscli-provided" or "own" or ""
	ModelPresetID string `json:"model_preset_id,omitempty"` // e.g. "kimi-k2.5-free"
	ModelToken    string `json:"model_token,omitempty"`     // API token for mscli-provided models
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
	return &cfg, nil
}

func saveAppConfig(cfg *appConfig) error {
	path := appConfigPath()
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o600)
}
