package app

import (
	"fmt"
	"strings"

	"github.com/mindspore-lab/mindspore-cli/configs"
)

type providerRestoreResult struct {
	Restored       bool
	ActivePresetID string
}

func restoreProviderSelection(cfg *configs.Config) (providerRestoreResult, error) {
	if cfg == nil {
		return providerRestoreResult{}, nil
	}

	appCfg, err := loadAppConfig()
	if err != nil {
		return providerRestoreResult{}, err
	}
	catalog, err := loadProviderCatalogBlocking(appCfg.ExtraProviders)
	if err != nil {
		return providerRestoreResult{}, err
	}

	authState, err := loadProviderAuthState()
	if err != nil {
		return providerRestoreResult{}, err
	}
	modelState, err := loadModelSelectionState()
	if err != nil {
		return providerRestoreResult{}, err
	}

	if modelState.Active != nil {
		resolved, presetID, err := resolveRuntimeSelection(catalog, authState, *modelState.Active)
		if err == nil {
			applyResolvedModelConfig(cfg, resolved)
			return providerRestoreResult{Restored: true, ActivePresetID: presetID}, nil
		}
	}

	if strings.TrimSpace(appCfg.ModelPresetID) != "" {
		if preset, ok := resolveBuiltinModelPreset(appCfg.ModelPresetID); ok {
			apiKey, err := fetchPresetAPIKey(preset)
			if err == nil {
				applyResolvedModelConfig(cfg, configs.ModelConfig{
					Provider: preset.Provider,
					URL:      preset.BaseURL,
					Model:    preset.Model,
					Key:      apiKey,
				})
				return providerRestoreResult{Restored: true, ActivePresetID: preset.ID}, nil
			}
		}
	}

	if isLoggedIn() {
		resolved, presetID, err := resolveRuntimeSelection(catalog, authState, modelRef{
			ProviderID: mindsporeCLIFreeProviderID,
			ModelID:    "kimi-k2.5",
		})
		if err == nil {
			applyResolvedModelConfig(cfg, resolved)
			return providerRestoreResult{Restored: true, ActivePresetID: presetID}, nil
		}
	}

	cfg.Model.Model = ""
	cfg.Model.Key = ""
	return providerRestoreResult{}, nil
}

func resolveRuntimeSelection(catalog *providerCatalog, authState *providerAuthState, ref modelRef) (configs.ModelConfig, string, error) {
	provider, ok := catalog.Provider(ref.ProviderID)
	if !ok {
		return configs.ModelConfig{}, "", fmt.Errorf("unknown provider %q", ref.ProviderID)
	}

	switch strings.TrimSpace(provider.Protocol) {
	case "mindspore-cli-free":
		preset, ok := freeProviderPresetForModel(ref.ModelID)
		if !ok {
			return configs.ModelConfig{}, "", fmt.Errorf("unknown free model %q", ref.ModelID)
		}
		apiKey, err := fetchPresetAPIKey(preset)
		if err != nil {
			return configs.ModelConfig{}, "", err
		}
		return configs.ModelConfig{
			Provider: preset.Provider,
			URL:      preset.BaseURL,
			Model:    preset.Model,
			Key:      apiKey,
		}, preset.ID, nil
	case "openai-chat":
		return resolveAPIKeyBackedRuntimeConfig(provider, authState, ref, "openai-completion")
	case "openai-responses":
		return resolveAPIKeyBackedRuntimeConfig(provider, authState, ref, "openai-responses")
	case "anthropic-messages":
		return resolveAPIKeyBackedRuntimeConfig(provider, authState, ref, "anthropic")
	default:
		return configs.ModelConfig{}, "", fmt.Errorf("unsupported provider protocol %q", provider.Protocol)
	}
}

func resolveAPIKeyBackedRuntimeConfig(provider providerCatalogEntry, authState *providerAuthState, ref modelRef, runtimeProvider string) (configs.ModelConfig, string, error) {
	if authState == nil {
		authState = emptyProviderAuthState()
	}
	entry, ok := authState.Providers[provider.ID]
	if !ok || strings.TrimSpace(entry.APIKey) == "" {
		return configs.ModelConfig{}, "", fmt.Errorf("provider %s is not connected", provider.ID)
	}
	return configs.ModelConfig{
		Provider: runtimeProvider,
		URL:      provider.BaseURL,
		Model:    strings.TrimSpace(ref.ModelID),
		Key:      strings.TrimSpace(entry.APIKey),
	}, "", nil
}

func applyResolvedModelConfig(cfg *configs.Config, resolved configs.ModelConfig) {
	if cfg == nil {
		return
	}
	previousModel := cfg.Model.Model
	cfg.Model.Provider = strings.TrimSpace(resolved.Provider)
	cfg.Model.URL = strings.TrimSpace(resolved.URL)
	cfg.Model.Model = strings.TrimSpace(resolved.Model)
	cfg.Model.Key = strings.TrimSpace(resolved.Key)
	configs.RefreshModelTokenDefaults(cfg, previousModel)
}

func freeProviderPresetForModel(modelID string) (builtinModelPreset, bool) {
	switch strings.TrimSpace(modelID) {
	case "kimi-k2.5":
		return resolveBuiltinModelPreset("kimi-k2.5-free")
	case "deepseek-chat":
		return resolveBuiltinModelPreset("deepseek-v3")
	default:
		return builtinModelPreset{}, false
	}
}

func isLoggedIn() bool {
	cred, err := loadCredentials()
	if err != nil {
		return false
	}
	return strings.TrimSpace(cred.Token) != "" && strings.TrimSpace(cred.ServerURL) != ""
}
