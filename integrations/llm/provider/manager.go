package provider

import (
	"errors"
	"fmt"
	"sync"

	"github.com/vigo999/ms-cli/integrations/llm"
)

var errProviderNotImplemented = errors.New("provider builder not implemented")

// Manager coordinates provider construction and instance caching.
type Manager struct {
	registry *Registry
	cache    *cache
	mu       sync.Mutex
}

var (
	defaultManagerOnce sync.Once
	defaultManager     *Manager
)

// NewManager creates a manager with an empty registry and cache.
func NewManager() *Manager {
	return &Manager{
		registry: NewRegistry(),
		cache:    newCache(),
	}
}

// DefaultManager returns the singleton default manager.
func DefaultManager() *Manager {
	defaultManagerOnce.Do(func() {
		defaultManager = NewManager()
		// Stub builders preserve the default registration surface until concrete
		// provider clients land in the follow-up task.
		mustRegisterDefaultProviders(defaultManager)
	})

	return defaultManager
}

// Register adds a provider builder to the manager registry.
func (m *Manager) Register(kind ProviderKind, builder Builder) error {
	return m.registry.Register(kind, builder)
}

// Build returns a cached provider instance for the resolved configuration.
func (m *Manager) Build(cfg ResolvedConfig) (llm.Provider, error) {
	key := cacheKey(cfg)

	m.mu.Lock()
	defer m.mu.Unlock()

	if cached, ok := m.cache.get(key); ok {
		provider, ok := cached.(llm.Provider)
		if !ok {
			return nil, fmt.Errorf("cached value for provider kind %q has unexpected type", cfg.Kind)
		}
		return provider, nil
	}

	provider, err := m.registry.Build(cfg)
	if err != nil {
		return nil, err
	}

	m.cache.set(key, provider)
	return provider, nil
}

func mustRegisterDefaultProviders(m *Manager) {
	builders := map[ProviderKind]Builder{
		ProviderOpenAI:           NewOpenAIProvider,
		ProviderOpenAICompatible: NewOpenAICompatibleProvider,
		ProviderAnthropic:        notImplementedBuilder(ProviderAnthropic),
	}
	for kind, builder := range builders {
		if err := m.Register(kind, builder); err != nil {
			panic(err)
		}
	}
}

func notImplementedBuilder(kind ProviderKind) Builder {
	return func(ResolvedConfig) (llm.Provider, error) {
		return nil, fmt.Errorf("provider %q builder not implemented", kind)
	}
}
