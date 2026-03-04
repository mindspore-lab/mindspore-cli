package tools

import (
	"fmt"
	"sync"

	"github.com/vigo999/ms-cli/integrations/llm"
)

// Registry manages all available tools.
type Registry struct {
	mu    sync.RWMutex
	tools map[string]Tool
	order []string // Maintain registration order
}

// NewRegistry creates a new tool registry.
func NewRegistry() *Registry {
	return &Registry{
		tools: make(map[string]Tool),
		order: make([]string, 0),
	}
}

// Register registers a tool.
func (r *Registry) Register(t Tool) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	name := t.Name()
	if name == "" {
		return fmt.Errorf("tool name cannot be empty")
	}

	if _, exists := r.tools[name]; exists {
		return fmt.Errorf("tool %q already registered", name)
	}

	r.tools[name] = t
	r.order = append(r.order, name)
	return nil
}

// MustRegister registers a tool, panicking on error.
func (r *Registry) MustRegister(t Tool) {
	if err := r.Register(t); err != nil {
		panic(err)
	}
}

// Get retrieves a tool by name.
func (r *Registry) Get(name string) (Tool, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	t, ok := r.tools[name]
	return t, ok
}

// List returns all registered tools.
func (r *Registry) List() []Tool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	list := make([]Tool, 0, len(r.tools))
	for _, name := range r.order {
		if t, ok := r.tools[name]; ok {
			list = append(list, t)
		}
	}
	return list
}

// Names returns all registered tool names.
func (r *Registry) Names() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, len(r.order))
	copy(names, r.order)
	return names
}

// Count returns the number of registered tools.
func (r *Registry) Count() int {
	r.mu.RLock()
	defer r.mu.RUnlock()

	return len(r.tools)
}

// ToLLMTools converts all tools to LLM tool format.
func (r *Registry) ToLLMTools() []llm.Tool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make([]llm.Tool, 0, len(r.tools))
	for _, name := range r.order {
		if t, ok := r.tools[name]; ok {
			result = append(result, llm.Tool{
				Type: "function",
				Function: llm.ToolFunction{
					Name:        t.Name(),
					Description: t.Description(),
					Parameters:  t.Schema(),
				},
			})
		}
	}
	return result
}

// GetLLMTool returns a specific tool in LLM format.
func (r *Registry) GetLLMTool(name string) (llm.Tool, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	t, ok := r.tools[name]
	if !ok {
		return llm.Tool{}, false
	}

	return llm.Tool{
		Type: "function",
		Function: llm.ToolFunction{
			Name:        t.Name(),
			Description: t.Description(),
			Parameters:  t.Schema(),
		},
	}, true
}

// GlobalRegistry is the global tool registry.
var GlobalRegistry = NewRegistry()

// Register registers a tool to the global registry.
func Register(t Tool) error {
	return GlobalRegistry.Register(t)
}

// MustRegister registers a tool to the global registry, panicking on error.
func MustRegister(t Tool) {
	GlobalRegistry.MustRegister(t)
}

// Get retrieves a tool from the global registry.
func Get(name string) (Tool, bool) {
	return GlobalRegistry.Get(name)
}
