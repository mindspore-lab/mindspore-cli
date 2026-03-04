package llm

import (
	"fmt"
	"sync"
)

// Registry manages all available providers.
type Registry struct {
	mu        sync.RWMutex
	providers map[string]Provider
	default_  string
}

// NewRegistry creates a new provider registry.
func NewRegistry() *Registry {
	return &Registry{
		providers: make(map[string]Provider),
	}
}

// Register registers a provider.
func (r *Registry) Register(p Provider) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	name := p.Name()
	if name == "" {
		return fmt.Errorf("provider name cannot be empty")
	}

	if _, exists := r.providers[name]; exists {
		return fmt.Errorf("provider %q already registered", name)
	}

	r.providers[name] = p

	// Set as default if it's the first provider
	if r.default_ == "" {
		r.default_ = name
	}

	return nil
}

// Get retrieves a provider by name.
func (r *Registry) Get(name string) (Provider, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	p, ok := r.providers[name]
	if !ok {
		return nil, fmt.Errorf("provider %q not found", name)
	}

	return p, nil
}

// List returns all registered providers.
func (r *Registry) List() []Provider {
	r.mu.RLock()
	defer r.mu.RUnlock()

	list := make([]Provider, 0, len(r.providers))
	for _, p := range r.providers {
		list = append(list, p)
	}
	return list
}

// Default returns the default provider.
func (r *Registry) Default() (Provider, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if r.default_ == "" {
		return nil, fmt.Errorf("no providers registered")
	}

	p, ok := r.providers[r.default_]
	if !ok {
		return nil, fmt.Errorf("default provider %q not found", r.default_)
	}

	return p, nil
}

// SetDefault sets the default provider.
func (r *Registry) SetDefault(name string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, ok := r.providers[name]; !ok {
		return fmt.Errorf("provider %q not found", name)
	}

	r.default_ = name
	return nil
}

// Names returns all registered provider names.
func (r *Registry) Names() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.providers))
	for name := range r.providers {
		names = append(names, name)
	}
	return names
}

// GlobalRegistry is the global provider registry.
var GlobalRegistry = NewRegistry()

// Register registers a provider to the global registry.
func Register(p Provider) error {
	return GlobalRegistry.Register(p)
}

// Get retrieves a provider from the global registry.
func Get(name string) (Provider, error) {
	return GlobalRegistry.Get(name)
}

// Default returns the default provider from the global registry.
func Default() (Provider, error) {
	return GlobalRegistry.Default()
}
