package app

import (
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	mindsporeCLIFreeProviderID = "mindspore-cli-free"
	defaultModelsDevAPIURL     = "https://models.dev/api.json"
)

var (
	connectPopularProviderIDs = []string{
		"anthropic",
		"openai",
		"kimi-for-coding",
		"deepseek",
	}
	defaultPopularProviderIDs = append(
		[]string{mindsporeCLIFreeProviderID},
		append(append([]string(nil), connectPopularProviderIDs...), "openrouter", "google", "groq")...,
	)

	modelsDevAPIURL            = defaultModelsDevAPIURL
	modelsDevCachePathOverride string

	modelsDevProvidersCacheMu      sync.Mutex
	modelsDevProvidersCache        []providerCatalogEntry
	modelsDevProvidersCacheLoaded  bool
	modelsDevProvidersRefreshAt    time.Time
	modelsDevProvidersRefreshAlive bool
)

const modelsDevRefreshTTL = 5 * time.Minute

type providerCatalog struct {
	Providers []providerCatalogEntry
}

type providerCatalogEntry struct {
	ID          string
	Label       string
	BaseURL     string
	Protocol    string
	Description string
	Builtin     bool
	Popular     bool
	Models      []modelCatalogEntry
}

type modelCatalogEntry struct {
	ProviderID      string
	ID              string
	Label           string
	Description     string
	ContextWindow   int
	MaxOutputTokens int
	Tags            []string
}

type modelsDevProvider struct {
	ID     string                    `json:"id"`
	API    string                    `json:"api"`
	Name   string                    `json:"name"`
	Doc    string                    `json:"doc"`
	NPM    string                    `json:"npm"`
	Models map[string]modelsDevModel `json:"models"`
}

type modelsDevModel struct {
	ID    string `json:"id"`
	Name  string `json:"name"`
	Limit struct {
		Context int `json:"context"`
		Output  int `json:"output"`
	} `json:"limit"`
}

func loadProviderCatalog(httpClient *http.Client, extras []extraProviderConfig) (*providerCatalog, error) {
	providers := make(map[string]providerCatalogEntry)

	for _, provider := range builtinProviderCatalog() {
		providers[provider.ID] = provider
	}

	var remote []providerCatalogEntry
	var ok bool
	if httpClient == nil {
		remote, ok = loadModelsDevProvidersNonBlocking()
	} else {
		remote, ok = loadModelsDevProviders(httpClient)
	}
	if ok {
		for _, provider := range remote {
			providers[provider.ID] = provider
		}
	}

	for _, provider := range extraProvidersToCatalog(extras) {
		providers[provider.ID] = provider
	}

	out := make([]providerCatalogEntry, 0, len(providers))
	for _, provider := range providers {
		out = append(out, provider)
	}
	sort.Slice(out, func(i, j int) bool {
		left, right := out[i], out[j]
		if left.Popular != right.Popular {
			return left.Popular
		}
		if left.Builtin != right.Builtin {
			return left.Builtin
		}
		return left.Label < right.Label
	})

	return &providerCatalog{Providers: out}, nil
}

func loadProviderCatalogBlocking(extras []extraProviderConfig) (*providerCatalog, error) {
	return loadProviderCatalog(&http.Client{Timeout: 5 * time.Second}, extras)
}

func loadModelsDevProvidersNonBlocking() ([]providerCatalogEntry, bool) {
	if cached, ok := cachedModelsDevProviders(); ok {
		maybeRefreshModelsDevProvidersAsync()
		return cached, true
	}
	if disk, ok := loadModelsDevProvidersFromDisk(); ok {
		setCachedModelsDevProvidersFromDisk(disk)
		maybeRefreshModelsDevProvidersAsync()
		return disk, true
	}
	maybeRefreshModelsDevProvidersAsync()
	return nil, false
}

func (c *providerCatalog) Provider(id string) (providerCatalogEntry, bool) {
	if c == nil {
		return providerCatalogEntry{}, false
	}
	needle := normalizedProviderID(id)
	for _, provider := range c.Providers {
		if provider.ID == needle {
			return provider, true
		}
	}
	return providerCatalogEntry{}, false
}

func builtinProviderCatalog() []providerCatalogEntry {
	return []providerCatalogEntry{
		{
			ID:       mindsporeCLIFreeProviderID,
			Label:    "MindSpore CLI Free",
			Protocol: "mindspore-cli-free",
			Builtin:  true,
			Popular:  true,
			Models: []modelCatalogEntry{
				{
					ProviderID:      mindsporeCLIFreeProviderID,
					ID:              "kimi-k2.5",
					Label:           "Kimi K2.5",
					ContextWindow:   262144,
					MaxOutputTokens: 262144,
				},
				{
					ProviderID:      mindsporeCLIFreeProviderID,
					ID:              "deepseek-chat",
					Label:           "DeepSeek V3",
					ContextWindow:   128000,
					MaxOutputTokens: 8000,
				},
			},
		},
	}
}

func loadModelsDevProviders(httpClient *http.Client) ([]providerCatalogEntry, bool) {
	body, ok := fetchModelsDevCatalog(httpClient)
	if ok {
		_ = writeModelsDevCache(body)
	} else {
		cached, err := readModelsDevCache()
		if err != nil {
			return nil, false
		}
		body = cached
	}

	parsed, err := parseModelsDevCatalog(body)
	if err != nil {
		return nil, false
	}
	setCachedModelsDevProviders(parsed)
	return parsed, true
}

func loadModelsDevProvidersFromDisk() ([]providerCatalogEntry, bool) {
	cached, err := readModelsDevCache()
	if err != nil {
		return nil, false
	}
	parsed, err := parseModelsDevCatalog(cached)
	if err != nil {
		return nil, false
	}
	return parsed, true
}

func cachedModelsDevProviders() ([]providerCatalogEntry, bool) {
	modelsDevProvidersCacheMu.Lock()
	defer modelsDevProvidersCacheMu.Unlock()
	if !modelsDevProvidersCacheLoaded || len(modelsDevProvidersCache) == 0 {
		return nil, false
	}
	return cloneProviderCatalogEntries(modelsDevProvidersCache), true
}

func setCachedModelsDevProviders(providers []providerCatalogEntry) {
	modelsDevProvidersCacheMu.Lock()
	defer modelsDevProvidersCacheMu.Unlock()
	modelsDevProvidersCache = cloneProviderCatalogEntries(providers)
	modelsDevProvidersCacheLoaded = true
	modelsDevProvidersRefreshAt = time.Now()
}

func setCachedModelsDevProvidersFromDisk(providers []providerCatalogEntry) {
	modelsDevProvidersCacheMu.Lock()
	defer modelsDevProvidersCacheMu.Unlock()
	modelsDevProvidersCache = cloneProviderCatalogEntries(providers)
	modelsDevProvidersCacheLoaded = true
}

func maybeRefreshModelsDevProvidersAsync() {
	httpClient := &http.Client{Timeout: 5 * time.Second}

	modelsDevProvidersCacheMu.Lock()
	if modelsDevProvidersRefreshAlive {
		modelsDevProvidersCacheMu.Unlock()
		return
	}
	if !modelsDevProvidersRefreshAt.IsZero() && time.Since(modelsDevProvidersRefreshAt) < modelsDevRefreshTTL {
		modelsDevProvidersCacheMu.Unlock()
		return
	}
	modelsDevProvidersRefreshAlive = true
	modelsDevProvidersRefreshAt = time.Now()
	modelsDevProvidersCacheMu.Unlock()

	go func() {
		defer func() {
			modelsDevProvidersCacheMu.Lock()
			modelsDevProvidersRefreshAlive = false
			modelsDevProvidersCacheMu.Unlock()
		}()

		body, ok := fetchModelsDevCatalog(httpClient)
		if !ok {
			return
		}
		parsed, err := parseModelsDevCatalog(body)
		if err != nil {
			return
		}
		_ = writeModelsDevCache(body)
		setCachedModelsDevProviders(parsed)
	}()
}

func cloneProviderCatalogEntries(in []providerCatalogEntry) []providerCatalogEntry {
	if len(in) == 0 {
		return nil
	}
	out := make([]providerCatalogEntry, len(in))
	for i, provider := range in {
		cloned := provider
		if len(provider.Models) > 0 {
			cloned.Models = append([]modelCatalogEntry(nil), provider.Models...)
		}
		out[i] = cloned
	}
	return out
}

func resetModelsDevProviderCacheForTest() {
	modelsDevProvidersCacheMu.Lock()
	defer modelsDevProvidersCacheMu.Unlock()
	modelsDevProvidersCache = nil
	modelsDevProvidersCacheLoaded = false
	modelsDevProvidersRefreshAt = time.Time{}
	modelsDevProvidersRefreshAlive = false
}

func fetchModelsDevCatalog(httpClient *http.Client) ([]byte, bool) {
	req, err := http.NewRequest(http.MethodGet, modelsDevAPIURL, nil)
	if err != nil {
		return nil, false
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, false
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, 16*1024*1024))
	if err != nil {
		return nil, false
	}
	return data, true
}

func parseModelsDevCatalog(body []byte) ([]providerCatalogEntry, error) {
	var payload map[string]modelsDevProvider
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, err
	}

	out := make([]providerCatalogEntry, 0, len(payload))
	for _, provider := range payload {
		id := normalizedProviderID(provider.ID)
		if id == "" {
			continue
		}
		models := make([]modelCatalogEntry, 0, len(provider.Models))
		for modelID, model := range provider.Models {
			normalizedModelID := strings.TrimSpace(model.ID)
			if normalizedModelID == "" {
				normalizedModelID = strings.TrimSpace(modelID)
			}
			if normalizedModelID == "" {
				continue
			}
			label := strings.TrimSpace(model.Name)
			if label == "" {
				label = normalizedModelID
			}
			models = append(models, modelCatalogEntry{
				ProviderID:      id,
				ID:              normalizedModelID,
				Label:           label,
				ContextWindow:   model.Limit.Context,
				MaxOutputTokens: model.Limit.Output,
			})
		}
		sort.Slice(models, func(i, j int) bool { return models[i].Label < models[j].Label })

		label := strings.TrimSpace(provider.Name)
		if label == "" {
			label = id
		}
		label = canonicalProviderLabel(id, label)
		out = append(out, providerCatalogEntry{
			ID:          id,
			Label:       label,
			BaseURL:     strings.TrimSpace(provider.API),
			Protocol:    inferProtocolFromModelsDev(provider),
			Description: strings.TrimSpace(provider.Doc),
			Popular:     isPopularProviderID(id),
			Models:      models,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Label < out[j].Label })
	return out, nil
}

func extraProvidersToCatalog(extras []extraProviderConfig) []providerCatalogEntry {
	out := make([]providerCatalogEntry, 0, len(extras))
	for _, provider := range extras {
		id := normalizedProviderID(provider.ID)
		if id == "" {
			continue
		}
		label := strings.TrimSpace(provider.Label)
		if label == "" {
			label = provider.ID
		}
		label = canonicalProviderLabel(id, label)
		models := make([]modelCatalogEntry, 0, len(provider.Models))
		for _, model := range provider.Models {
			modelID := strings.TrimSpace(model.ID)
			if modelID == "" {
				continue
			}
			modelLabel := strings.TrimSpace(model.Label)
			if modelLabel == "" {
				modelLabel = modelID
			}
			models = append(models, modelCatalogEntry{
				ProviderID: id,
				ID:         modelID,
				Label:      modelLabel,
			})
		}
		out = append(out, providerCatalogEntry{
			ID:       id,
			Label:    label,
			BaseURL:  strings.TrimSpace(provider.BaseURL),
			Protocol: strings.TrimSpace(provider.Protocol),
			Popular:  isPopularProviderID(id),
			Models:   models,
		})
	}
	return out
}

func inferProtocolFromModelsDev(provider modelsDevProvider) string {
	npm := strings.ToLower(strings.TrimSpace(provider.NPM))
	switch {
	case strings.Contains(npm, "anthropic"):
		return "anthropic-messages"
	case strings.Contains(npm, "responses"):
		return "openai-responses"
	default:
		return "openai-chat"
	}
}

func isPopularProviderID(id string) bool {
	needle := normalizedProviderID(id)
	for _, candidate := range defaultPopularProviderIDs {
		if needle == candidate {
			return true
		}
	}
	return false
}

func canonicalProviderLabel(id, fallback string) string {
	return strings.TrimSpace(fallback)
}

func modelsDevCachePath() string {
	if modelsDevCachePathOverride != "" {
		return modelsDevCachePathOverride
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".mscli", "cached", "models-dev-api.json")
}

func writeModelsDevCache(body []byte) error {
	path := modelsDevCachePath()
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	return os.WriteFile(path, body, 0o600)
}

func readModelsDevCache() ([]byte, error) {
	return os.ReadFile(modelsDevCachePath())
}
