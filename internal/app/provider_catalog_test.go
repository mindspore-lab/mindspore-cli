package app

import (
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
	"time"
)

func TestProviderCatalogIncludesBuiltinFreeProvider(t *testing.T) {
	server := newModelsDevTestServer(`{}`)
	defer server.Close()

	origURL := modelsDevAPIURL
	modelsDevAPIURL = server.URL
	t.Cleanup(func() { modelsDevAPIURL = origURL })

	dir := t.TempDir()
	origCache := modelsDevCachePathOverride
	modelsDevCachePathOverride = filepath.Join(dir, "models-dev-api.json")
	t.Cleanup(func() { modelsDevCachePathOverride = origCache })

	catalog, err := loadProviderCatalog(http.DefaultClient, nil)
	if err != nil {
		t.Fatalf("loadProviderCatalog() error = %v", err)
	}

	provider, ok := catalog.Provider(mindsporeCLIFreeProviderID)
	if !ok {
		t.Fatalf("catalog.Provider(%q) ok = false", mindsporeCLIFreeProviderID)
	}
	if got, want := provider.Label, "MindSpore CLI Free"; got != want {
		t.Fatalf("provider.Label = %q, want %q", got, want)
	}
	if got, want := provider.Models[0].ID, "kimi-k2.5"; got != want {
		t.Fatalf("provider.Models[0].ID = %q, want %q", got, want)
	}
}

func TestProviderCatalogMergesModelsDevAndExtraProviders(t *testing.T) {
	server := newModelsDevTestServer(`{
		"openrouter": {
			"id": "openrouter",
			"name": "OpenRouter",
			"api": "https://openrouter.ai/api/v1",
			"npm": "@openrouter/ai-sdk-provider",
			"models": {
				"openai/gpt-4o-mini": {
					"id": "openai/gpt-4o-mini",
					"name": "GPT-4o mini",
					"limit": {"context": 128000, "output": 16384}
				}
			}
		}
	}`)
	defer server.Close()

	origURL := modelsDevAPIURL
	modelsDevAPIURL = server.URL
	t.Cleanup(func() { modelsDevAPIURL = origURL })

	dir := t.TempDir()
	origCache := modelsDevCachePathOverride
	modelsDevCachePathOverride = filepath.Join(dir, "models-dev-api.json")
	t.Cleanup(func() { modelsDevCachePathOverride = origCache })

	catalog, err := loadProviderCatalog(http.DefaultClient, []extraProviderConfig{
		{
			ID:       "my-gateway",
			Label:    "My Gateway",
			BaseURL:  "https://llm.example.com/v1",
			Protocol: "openai-chat",
			Models: []extraProviderModelConfig{
				{ID: "my-model", Label: "My Model"},
			},
		},
	})
	if err != nil {
		t.Fatalf("loadProviderCatalog() error = %v", err)
	}

	if _, ok := catalog.Provider("openrouter"); !ok {
		t.Fatal("expected openrouter in merged catalog")
	}
	custom, ok := catalog.Provider("my-gateway")
	if !ok {
		t.Fatal("expected my-gateway in merged catalog")
	}
	if got, want := custom.Models[0].ID, "my-model"; got != want {
		t.Fatalf("custom.Models[0].ID = %q, want %q", got, want)
	}
}

func TestProviderCatalogLocalOverridesTakePrecedence(t *testing.T) {
	server := newModelsDevTestServer(`{
		"openrouter": {
			"id": "openrouter",
			"name": "OpenRouter",
			"api": "https://openrouter.ai/api/v1",
			"npm": "@openrouter/ai-sdk-provider",
			"models": {
				"openai/gpt-4o-mini": {"id": "openai/gpt-4o-mini", "name": "GPT-4o mini"}
			}
		}
	}`)
	defer server.Close()

	origURL := modelsDevAPIURL
	modelsDevAPIURL = server.URL
	t.Cleanup(func() { modelsDevAPIURL = origURL })

	dir := t.TempDir()
	origCache := modelsDevCachePathOverride
	modelsDevCachePathOverride = filepath.Join(dir, "models-dev-api.json")
	t.Cleanup(func() { modelsDevCachePathOverride = origCache })

	catalog, err := loadProviderCatalog(http.DefaultClient, []extraProviderConfig{
		{
			ID:       "openrouter",
			Label:    "OpenRouter Custom",
			BaseURL:  "https://custom.example/v1",
			Protocol: "openai-chat",
			Models: []extraProviderModelConfig{
				{ID: "custom-model", Label: "Custom Model"},
			},
		},
	})
	if err != nil {
		t.Fatalf("loadProviderCatalog() error = %v", err)
	}

	provider, ok := catalog.Provider("openrouter")
	if !ok {
		t.Fatal("expected openrouter in catalog")
	}
	if got, want := provider.Label, "OpenRouter Custom"; got != want {
		t.Fatalf("provider.Label = %q, want %q", got, want)
	}
	if got, want := provider.BaseURL, "https://custom.example/v1"; got != want {
		t.Fatalf("provider.BaseURL = %q, want %q", got, want)
	}
	if got, want := provider.Models[0].ID, "custom-model"; got != want {
		t.Fatalf("provider.Models[0].ID = %q, want %q", got, want)
	}
}

func TestProviderCatalogFallsBackToCacheOnFetchFailure(t *testing.T) {
	dir := t.TempDir()
	origCache := modelsDevCachePathOverride
	modelsDevCachePathOverride = filepath.Join(dir, "models-dev-api.json")
	t.Cleanup(func() { modelsDevCachePathOverride = origCache })

	cached := `{
		"anthropic": {
			"id": "anthropic",
			"name": "Anthropic",
			"npm": "@ai-sdk/anthropic",
			"models": {
				"claude-sonnet-4-5": {"id": "claude-sonnet-4-5", "name": "Claude Sonnet 4.5"}
			}
		}
	}`
	if err := writeModelsDevCache([]byte(cached)); err != nil {
		t.Fatalf("writeModelsDevCache() error = %v", err)
	}

	origURL := modelsDevAPIURL
	modelsDevAPIURL = "http://127.0.0.1:1"
	t.Cleanup(func() { modelsDevAPIURL = origURL })

	catalog, err := loadProviderCatalog(http.DefaultClient, nil)
	if err != nil {
		t.Fatalf("loadProviderCatalog() error = %v", err)
	}
	if _, ok := catalog.Provider("anthropic"); !ok {
		t.Fatal("expected cached anthropic provider")
	}
}

func TestProviderCatalogFallsBackToBuiltinAndExtraWithoutCache(t *testing.T) {
	dir := t.TempDir()
	origCache := modelsDevCachePathOverride
	modelsDevCachePathOverride = filepath.Join(dir, "missing", "models-dev-api.json")
	t.Cleanup(func() { modelsDevCachePathOverride = origCache })

	origURL := modelsDevAPIURL
	modelsDevAPIURL = "http://127.0.0.1:1"
	t.Cleanup(func() { modelsDevAPIURL = origURL })

	catalog, err := loadProviderCatalog(http.DefaultClient, []extraProviderConfig{
		{
			ID:       "my-gateway",
			BaseURL:  "https://llm.example.com/v1",
			Protocol: "openai-chat",
		},
	})
	if err != nil {
		t.Fatalf("loadProviderCatalog() error = %v", err)
	}

	if _, ok := catalog.Provider(mindsporeCLIFreeProviderID); !ok {
		t.Fatal("expected builtin free provider")
	}
	if _, ok := catalog.Provider("my-gateway"); !ok {
		t.Fatal("expected extra provider fallback")
	}
}

func TestProviderCatalogPreservesProviderIDAndName(t *testing.T) {
	server := newModelsDevTestServer(`{
		"acmeai": {
			"id": "acmeai",
			"name": "Acme AI",
			"api": "https://api.acme.ai/v1",
			"npm": "@ai-sdk/openai-compatible",
			"models": {}
		}
	}`)
	defer server.Close()

	origURL := modelsDevAPIURL
	modelsDevAPIURL = server.URL
	t.Cleanup(func() { modelsDevAPIURL = origURL })

	dir := t.TempDir()
	origCache := modelsDevCachePathOverride
	modelsDevCachePathOverride = filepath.Join(dir, "models-dev-api.json")
	t.Cleanup(func() { modelsDevCachePathOverride = origCache })

	catalog, err := loadProviderCatalog(http.DefaultClient, nil)
	if err != nil {
		t.Fatalf("loadProviderCatalog() error = %v", err)
	}
	provider, ok := catalog.Provider("acmeai")
	if !ok {
		t.Fatal("expected acmeai in catalog")
	}
	if got, want := provider.ID, "acmeai"; got != want {
		t.Fatalf("provider.ID = %q, want %q", got, want)
	}
	if got, want := provider.Label, "Acme AI"; got != want {
		t.Fatalf("provider.Label = %q, want %q", got, want)
	}
}

func TestProviderCatalogUsesCacheWithoutBlockingWhenNilClient(t *testing.T) {
	resetModelsDevProviderCacheForTest()

	dir := t.TempDir()
	origCache := modelsDevCachePathOverride
	modelsDevCachePathOverride = filepath.Join(dir, "models-dev-api.json")
	t.Cleanup(func() { modelsDevCachePathOverride = origCache })

	cached := `{
		"anthropic": {
			"id": "anthropic",
			"name": "Anthropic",
			"npm": "@ai-sdk/anthropic",
			"models": {}
		}
	}`
	if err := writeModelsDevCache([]byte(cached)); err != nil {
		t.Fatalf("writeModelsDevCache() error = %v", err)
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(1100 * time.Millisecond)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{}`))
	}))
	defer server.Close()

	origURL := modelsDevAPIURL
	modelsDevAPIURL = server.URL
	t.Cleanup(func() { modelsDevAPIURL = origURL })

	start := time.Now()
	catalog, err := loadProviderCatalog(nil, nil)
	if err != nil {
		t.Fatalf("loadProviderCatalog() error = %v", err)
	}
	if elapsed := time.Since(start); elapsed >= 500*time.Millisecond {
		t.Fatalf("loadProviderCatalog() took %v, want under 500ms", elapsed)
	}
	if _, ok := catalog.Provider("anthropic"); !ok {
		t.Fatal("expected cached anthropic provider")
	}
}

func TestProviderCatalogNilClientReturnsBuiltinAndExtraWithoutBlockingWhenCacheMissing(t *testing.T) {
	resetModelsDevProviderCacheForTest()

	dir := t.TempDir()
	origCache := modelsDevCachePathOverride
	modelsDevCachePathOverride = filepath.Join(dir, "missing", "models-dev-api.json")
	t.Cleanup(func() { modelsDevCachePathOverride = origCache })

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(1100 * time.Millisecond)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"openrouter": {
				"id": "openrouter",
				"name": "OpenRouter",
				"npm": "@openrouter/ai-sdk-provider",
				"models": {}
			}
		}`))
	}))
	defer server.Close()

	origURL := modelsDevAPIURL
	modelsDevAPIURL = server.URL
	t.Cleanup(func() { modelsDevAPIURL = origURL })

	start := time.Now()
	catalog, err := loadProviderCatalog(nil, []extraProviderConfig{
		{
			ID:       "my-gateway",
			BaseURL:  "https://llm.example.com/v1",
			Protocol: "openai-chat",
		},
	})
	if err != nil {
		t.Fatalf("loadProviderCatalog() error = %v", err)
	}
	if elapsed := time.Since(start); elapsed >= 500*time.Millisecond {
		t.Fatalf("loadProviderCatalog() took %v, want under 500ms", elapsed)
	}
	if _, ok := catalog.Provider(mindsporeCLIFreeProviderID); !ok {
		t.Fatal("expected builtin free provider")
	}
	if _, ok := catalog.Provider("my-gateway"); !ok {
		t.Fatal("expected extra provider")
	}
}

func newModelsDevTestServer(body string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
}
