package provider

import (
	"fmt"
	"sync"

	"github.com/vigo999/ms-cli/integrations/llm"
)

// Builder constructs a provider from a resolved configuration.
type Builder func(ResolvedConfig) (llm.Provider, error)

// Registry stores provider builders by provider kind.
type Registry struct {
	mu       sync.RWMutex
	builders map[ProviderKind]Builder
}

// NewRegistry creates an empty provider registry.
func NewRegistry() *Registry {
	return &Registry{
		builders: make(map[ProviderKind]Builder),
	}
}

// Register associates a provider kind with a builder.
func (r *Registry) Register(kind ProviderKind, builder Builder) error {
	if builder == nil {
		return fmt.Errorf("builder is nil")
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if kind == "" {
		return fmt.Errorf("provider kind is empty")
	}
	if _, exists := r.builders[kind]; exists {
		return fmt.Errorf("provider kind %q already registered", kind)
	}

	r.builders[kind] = builder
	return nil
}

// Build constructs a provider for the resolved configuration.
func (r *Registry) Build(cfg ResolvedConfig) (llm.Provider, error) {
	r.mu.RLock()
	builder, ok := r.builders[cfg.Kind]
	r.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("provider kind %q not registered", cfg.Kind)
	}

	provider, err := builder(cfg)
	if err != nil {
		return nil, err
	}
	if provider == nil {
		return nil, fmt.Errorf("provider kind %q builder returned nil provider", cfg.Kind)
	}

	return provider, nil
}
