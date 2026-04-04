// Package slash provides slash command definitions and autocomplete functionality.
package slash

import (
	"sort"
	"strings"
	"sync"
)

// Command represents a slash command.
type Command struct {
	Name        string
	Description string
	Usage       string
	Hidden      bool // hidden from /help and autocomplete, but still executable
	Handler     func(args []string) string
}

// Registry holds all available slash commands.
type Registry struct {
	commands map[string]Command
	mu       sync.RWMutex
}

// NewRegistry creates a new slash command registry.
func NewRegistry() *Registry {
	r := &Registry{
		commands: make(map[string]Command),
	}
	r.registerDefaults()
	return r
}

// Register adds a command to the registry.
func (r *Registry) Register(cmd Command) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.commands[cmd.Name] = cmd
}

// Get retrieves a command by name.
func (r *Registry) Get(name string) (Command, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	cmd, ok := r.commands[name]
	return cmd, ok
}

// List returns all registered commands.
func (r *Registry) List() []Command {
	r.mu.RLock()
	defer r.mu.RUnlock()
	cmds := make([]Command, 0, len(r.commands))
	for _, cmd := range r.commands {
		if !cmd.Hidden {
			cmds = append(cmds, cmd)
		}
	}
	return cmds
}

// Match returns commands that match the given prefix.
func (r *Registry) Match(prefix string) []Command {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if prefix == "" {
		cmds := make([]Command, 0, len(r.commands))
		for _, cmd := range r.commands {
			if !cmd.Hidden {
				cmds = append(cmds, cmd)
			}
		}
		return cmds
	}

	var matches []Command
	for name, cmd := range r.commands {
		if !cmd.Hidden && strings.HasPrefix(name, prefix) {
			matches = append(matches, cmd)
		}
	}
	return matches
}

// Suggestions returns command names that match the given input.
// Exact matches are listed first, then sorted alphabetically.
func (r *Registry) Suggestions(input string) []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var matches []string
	if input == "/" {
		for name, cmd := range r.commands {
			if !cmd.Hidden {
				matches = append(matches, name)
			}
		}
	} else {
		for name, cmd := range r.commands {
			if !cmd.Hidden && strings.HasPrefix(name, input) {
				matches = append(matches, name)
			}
		}
	}

	sort.Slice(matches, func(i, j int) bool {
		// Exact match goes first
		iExact := matches[i] == input
		jExact := matches[j] == input
		if iExact != jExact {
			return iExact
		}
		// Then shorter names first (closer matches)
		if len(matches[i]) != len(matches[j]) {
			return len(matches[i]) < len(matches[j])
		}
		return matches[i] < matches[j]
	})

	return matches
}

// IsSlashCommand checks if input starts with "/".
func IsSlashCommand(input string) bool {
	return strings.HasPrefix(strings.TrimSpace(input), "/")
}

// Parse parses a slash command input into name and arguments.
func Parse(input string) (string, []string) {
	input = strings.TrimSpace(input)
	if !IsSlashCommand(input) {
		return "", nil
	}

	parts := strings.Fields(input)
	if len(parts) == 0 {
		return "", nil
	}

	name := parts[0]
	args := []string{}
	if len(parts) > 1 {
		args = parts[1:]
	}

	return name, args
}

func (r *Registry) registerDefaults() {
	r.Register(Command{
		Name:        "/model",
		Description: "Open model setup or switch model",
		Usage:       "/model [preset-id|provider:model|model]",
	})

	r.Register(Command{
		Name:        "/exit",
		Description: "Exit the application",
		Usage:       "/exit",
	})

	r.Register(Command{
		Name:        "/compact",
		Description: "Compact conversation context",
		Usage:       "/compact",
	})

	r.Register(Command{
		Name:        "/clear",
		Description: "Clear the chat history",
		Usage:       "/clear",
	})

	r.Register(Command{
		Name:        "/test",
		Description: "Test API connectivity",
		Usage:       "/test",
	})

	r.Register(Command{
		Name:        "/permissions",
		Description: "Open permissions view",
		Usage:       "/permissions",
	})

	r.Register(Command{
		Name:        "/yolo",
		Description: "Toggle yolo mode (auto-approve all)",
		Usage:       "/yolo",
	})

	r.Register(Command{
		Name:        "/train",
		Description: "Start or control the train HUD workflow",
		Usage:       "/train <model> <method> | /train <action>",
	})

	r.Register(Command{
		Name:        "/project",
		Description: "Show or edit project status data",
		Usage:       "/project [status|add|update|rm]",
		Hidden:      true,
	})

	r.Register(Command{
		Name:        "/skill",
		Description: "Load a skill and start it",
		Usage:       "/skill <name> [request...]",
	})

	r.Register(Command{
		Name:        "/skill-add",
		Description: "Add local or remote skills into ~/.mscli/skills",
		Usage:       "/skill-add <path|git-url|owner/repo>",
	})

	r.Register(Command{
		Name:        "/skill-update",
		Description: "Update shared skills repo",
		Usage:       "/skill-update",
	})

	r.Register(Command{
		Name:        "/login",
		Description: "Log in to the bug server",
		Usage:       "/login <token>",
	})

	r.Register(Command{
		Name:        "/report",
		Description: "Report a bug or issue",
		Usage:       "/report [tags] <title> | /report acc|fail|perf <title>",
	})

	r.Register(Command{
		Name:        "/issues",
		Description: "List issues",
		Usage:       "/issues [status]",
	})

	r.Register(Command{
		Name:        "/status",
		Description: "Update issue status",
		Usage:       "/status <ISSUE-id> <ready|doing|closed>",
	})

	r.Register(Command{
		Name:        "/diagnose",
		Description: "Diagnose a problem or issue",
		Usage:       "/diagnose <problem text|ISSUE-id>",
	})

	r.Register(Command{
		Name:        "/fix",
		Description: "Fix a problem or issue",
		Usage:       "/fix <problem text|ISSUE-id>",
	})

	r.Register(Command{
		Name:        "/migrate",
		Description: "Migrate a model to MindSpore",
		Usage:       "/migrate <description or repo URL>",
	})

	r.Register(Command{
		Name:        "/bugs",
		Description: "List bugs",
		Usage:       "/bugs [status]",
	})

	r.Register(Command{
		Name:        "/claim",
		Description: "Claim a bug as your lead",
		Usage:       "/claim <id>",
	})

	r.Register(Command{
		Name:        "/close",
		Description: "Close a resolved bug",
		Usage:       "/close <id>",
	})

	r.Register(Command{
		Name:        "/dock",
		Description: "Show bug dashboard",
		Usage:       "/dock",
	})

	r.Register(Command{
		Name:        "/help",
		Description: "Show available commands",
		Usage:       "/help",
	})
}

// DefaultRegistry is the global slash command registry.
var DefaultRegistry = NewRegistry()

// Register registers a command to the default registry.
func Register(cmd Command) {
	DefaultRegistry.Register(cmd)
}

// Get retrieves a command from the default registry.
func Get(name string) (Command, bool) {
	return DefaultRegistry.Get(name)
}

// List returns all commands from the default registry.
func List() []Command {
	return DefaultRegistry.List()
}

// Suggestions returns suggestions from the default registry.
func Suggestions(input string) []string {
	return DefaultRegistry.Suggestions(input)
}
