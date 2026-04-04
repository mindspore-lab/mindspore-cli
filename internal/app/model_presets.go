package app

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/vigo999/mindspore-cli/configs"
)

type modelCredentialStrategy string

const (
	credentialStrategyStatic       modelCredentialStrategy = "static"
	credentialStrategyMSCLIServer modelCredentialStrategy = "mscli-server"
)

type modelCredentialSpec struct {
	Strategy  modelCredentialStrategy
	StaticKey string
	Path      string
}

type builtinModelPreset struct {
	ID         string
	Label      string
	Provider   string
	BaseURL    string
	Model      string
	Aliases    []string
	Credential modelCredentialSpec
	ComingSoon bool
}

var builtinModelPresets = []builtinModelPreset{
	{
		ID:       "kimi-k2.5-free",
		Label:    "kimi-k2.5 [free]",
		Provider: "anthropic",
		BaseURL:  "https://api.kimi.com/coding/",
		Model:    "kimi-k2.5",
		Aliases:  []string{"kimi-k2.5", "kimi-k2.5 [free]"},
		Credential: modelCredentialSpec{
			Strategy: credentialStrategyMSCLIServer,
			Path:     "/model-presets/kimi-k2.5-free/credential",
		},
	},
	{
		ID:       "deepseek-v3",
		Label:    "deepseek-v3",
		Provider: "openai-completion",
		BaseURL:  "https://api.deepseek.com/v1",
		Model:    "deepseek-chat",
		Aliases:  []string{"deepseek-v3", "deepseek-chat", "deepseek"},
		Credential: modelCredentialSpec{
			Strategy: credentialStrategyMSCLIServer,
			Path:     "/model-presets/deepseek-v3/credential",
		},
	},
	{
		ID:         "glm-4.7",
		Label:      "glm-4.7 (coming soon)",
		Model:      "glm-4.7",
		ComingSoon: true,
	},
	{
		ID:         "minimax-m2.7",
		Label:      "minimax-m2.7 (coming soon)",
		Model:      "minimax-m2.7",
		ComingSoon: true,
	},
}

func listBuiltinModelPresets() []builtinModelPreset {
	out := make([]builtinModelPreset, len(builtinModelPresets))
	copy(out, builtinModelPresets)
	return out
}

func resolveBuiltinModelPreset(input string) (builtinModelPreset, bool) {
	needle := strings.ToLower(strings.TrimSpace(input))
	if needle == "" {
		return builtinModelPreset{}, false
	}

	for _, preset := range builtinModelPresets {
		if preset.ComingSoon {
			continue
		}
		if strings.EqualFold(needle, preset.ID) || strings.EqualFold(needle, preset.Label) {
			return preset, true
		}
		for _, alias := range preset.Aliases {
			if strings.EqualFold(needle, alias) {
				return preset, true
			}
		}
	}
	return builtinModelPreset{}, false
}

// fetchPresetAPIKey fetches the API key for a preset without needing an Application instance.
// Used at startup before the Application is created.
func fetchPresetAPIKey(preset builtinModelPreset) (string, error) {
	switch preset.Credential.Strategy {
	case credentialStrategyStatic:
		return strings.TrimSpace(preset.Credential.StaticKey), nil
	case credentialStrategyMSCLIServer:
		cred, err := loadCredentials()
		if err != nil || strings.TrimSpace(cred.Token) == "" || strings.TrimSpace(cred.ServerURL) == "" {
			return "", fmt.Errorf("not logged in")
		}
		path := strings.TrimSpace(preset.Credential.Path)
		if path == "" {
			path = "/model-presets/" + preset.ID + "/credential"
		}
		url := strings.TrimRight(strings.TrimSpace(cred.ServerURL), "/") + path
		req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, url, nil)
		if err != nil {
			return "", err
		}
		req.Header.Set("Authorization", "Bearer "+cred.Token)
		client := &http.Client{Timeout: 5 * time.Second}
		resp, err := client.Do(req)
		if err != nil {
			return "", err
		}
		defer resp.Body.Close()
		data, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusOK {
			return "", fmt.Errorf("status %d", resp.StatusCode)
		}
		var payload struct {
			APIKey string `json:"api_key"`
		}
		if err := json.Unmarshal(data, &payload); err != nil {
			return "", err
		}
		return strings.TrimSpace(payload.APIKey), nil
	default:
		return "", fmt.Errorf("unsupported strategy %q", preset.Credential.Strategy)
	}
}

func (a *Application) resolveModelPresetAPIKey(ctx context.Context, preset builtinModelPreset) (string, error) {
	switch preset.Credential.Strategy {
	case credentialStrategyStatic:
		if strings.TrimSpace(preset.Credential.StaticKey) == "" {
			return "", fmt.Errorf("preset %s static credential is empty", preset.ID)
		}
		return strings.TrimSpace(preset.Credential.StaticKey), nil
	case credentialStrategyMSCLIServer:
		return a.fetchPresetAPIKeyFromServer(ctx, preset)
	default:
		return "", fmt.Errorf("unsupported credential strategy %q", preset.Credential.Strategy)
	}
}

func (a *Application) fetchPresetAPIKeyFromServer(ctx context.Context, preset builtinModelPreset) (string, error) {
	cred, err := loadCredentials()
	if err != nil || strings.TrimSpace(cred.Token) == "" || strings.TrimSpace(cred.ServerURL) == "" {
		return "", fmt.Errorf("not logged in. run /login <token> first")
	}
	path := strings.TrimSpace(preset.Credential.Path)
	if path == "" {
		path = "/model-presets/" + preset.ID + "/credential"
	}
	url := strings.TrimRight(strings.TrimSpace(cred.ServerURL), "/") + path

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", fmt.Errorf("build credential request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+cred.Token)

	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("request preset credential: %w", err)
	}
	defer resp.Body.Close()

	data, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("request preset credential failed: status %d", resp.StatusCode)
	}
	var payload struct {
		APIKey string `json:"api_key"`
	}
	if err := json.Unmarshal(data, &payload); err != nil {
		return "", fmt.Errorf("decode preset credential response: %w", err)
	}
	if strings.TrimSpace(payload.APIKey) == "" {
		return "", fmt.Errorf("preset credential response is empty")
	}
	return strings.TrimSpace(payload.APIKey), nil
}

func copyModelConfig(cfg configs.ModelConfig) *configs.ModelConfig {
	c := cfg
	if cfg.Headers != nil {
		c.Headers = make(map[string]string, len(cfg.Headers))
		for k, v := range cfg.Headers {
			c.Headers[k] = v
		}
	}
	return &c
}
