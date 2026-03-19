package provider

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/vigo999/ms-cli/integrations/llm"
)

type testProvider struct {
	name string
}

func (p *testProvider) Name() string { return p.name }
func (p *testProvider) Complete(ctx context.Context, req *llm.CompletionRequest) (*llm.CompletionResponse, error) {
	return nil, nil
}
func (p *testProvider) CompleteStream(ctx context.Context, req *llm.CompletionRequest) (llm.StreamIterator, error) {
	return nil, nil
}
func (p *testProvider) SupportsTools() bool              { return false }
func (p *testProvider) AvailableModels() []llm.ModelInfo { return nil }

func TestManagerBuild_CacheHitReturnsSameInstance(t *testing.T) {
	m := NewManager()

	var builds int
	if err := m.Register(ProviderOpenAICompatible, func(cfg ResolvedConfig) (llm.Provider, error) {
		builds++
		return &testProvider{name: "one"}, nil
	}); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	cfg := ResolvedConfig{
		Kind:           ProviderOpenAICompatible,
		BaseURL:        "https://example.invalid/v1",
		Model:          "gpt-test",
		Timeout:        30 * time.Second,
		Headers:        map[string]string{"X-Test": "1"},
		AuthHeaderName: "Authorization",
		APIKey:         "secret",
	}

	first, err := m.Build(cfg)
	if err != nil {
		t.Fatalf("Build() first error = %v", err)
	}
	second, err := m.Build(cfg)
	if err != nil {
		t.Fatalf("Build() second error = %v", err)
	}

	if first != second {
		t.Fatalf("Build() provider instances differ: %p vs %p", first, second)
	}
	if builds != 1 {
		t.Fatalf("builder called %d times, want 1", builds)
	}
}

func TestManagerBuild_DifferentConfigProducesDifferentInstances(t *testing.T) {
	m := NewManager()

	var builds int
	if err := m.Register(ProviderOpenAICompatible, func(cfg ResolvedConfig) (llm.Provider, error) {
		builds++
		return &testProvider{name: cfg.Model}, nil
	}); err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	base := ResolvedConfig{
		Kind:           ProviderOpenAICompatible,
		BaseURL:        "https://example.invalid/v1",
		Model:          "gpt-test-a",
		Timeout:        30 * time.Second,
		Headers:        map[string]string{"X-Test": "1"},
		AuthHeaderName: "Authorization",
		APIKey:         "secret",
	}
	other := base
	other.Model = "gpt-test-b"

	first, err := m.Build(base)
	if err != nil {
		t.Fatalf("Build() first error = %v", err)
	}
	second, err := m.Build(other)
	if err != nil {
		t.Fatalf("Build() second error = %v", err)
	}

	if first == second {
		t.Fatal("Build() returned same instance for different configs")
	}
	if builds != 2 {
		t.Fatalf("builder called %d times, want 2", builds)
	}
}

func TestManagerBuild_UnregisteredKindReturnsError(t *testing.T) {
	m := NewManager()

	_, err := m.Build(ResolvedConfig{Kind: ProviderAnthropic})
	if err == nil {
		t.Fatal("Build() error = nil, want unregistered provider error")
	}
	if got := err.Error(); got == "" {
		t.Fatal("Build() error message is empty")
	}
	if errors.Is(err, errProviderNotImplemented) {
		t.Fatal("Build() returned not-implemented error for unregistered kind")
	}
}
