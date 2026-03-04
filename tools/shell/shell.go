package shell

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/vigo999/ms-cli/integrations/llm"
	"github.com/vigo999/ms-cli/tools"
)

// ShellTool wraps shell execution as a Tool.
type ShellTool struct {
	runner *Runner
}

// NewShellTool creates a new shell tool.
func NewShellTool(runner *Runner) *ShellTool {
	return &ShellTool{runner: runner}
}

// Name returns the tool name.
func (t *ShellTool) Name() string {
	return "shell"
}

// Description returns the tool description.
func (t *ShellTool) Description() string {
	return "Execute a shell command. Use this for running tests, building, git operations, etc. Commands have a timeout and destructive operations may require confirmation."
}

// Schema returns the tool parameter schema.
func (t *ShellTool) Schema() llm.ToolSchema {
	return llm.ToolSchema{
		Type: "object",
		Properties: map[string]llm.Property{
			"command": {
				Type:        "string",
				Description: "The shell command to execute (e.g., 'go test ./...', 'git status')",
			},
			"timeout": {
				Type:        "integer",
				Description: "Timeout in seconds (default: 60, max: 1800)",
			},
		},
		Required: []string{"command"},
	}
}

type shellParams struct {
	Command string `json:"command"`
	Timeout int    `json:"timeout"`
}

// Execute executes the shell tool.
func (t *ShellTool) Execute(ctx context.Context, params json.RawMessage) (*tools.Result, error) {
	var p shellParams
	if err := tools.ParseParams(params, &p); err != nil {
		return tools.ErrorResult(err), nil
	}

	command := strings.TrimSpace(p.Command)
	if command == "" {
		return tools.ErrorResultf("command is required"), nil
	}

	// Apply custom timeout if specified
	if p.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeoutFromInt(p.Timeout))
		defer cancel()
	}

	// Run command
	result, err := t.runner.Run(ctx, command)
	if err != nil {
		return tools.ErrorResultf("execute command: %w", err), nil
	}

	// Build output
	var parts []string
	parts = append(parts, fmt.Sprintf("$ %s", command))

	if result.Stdout != "" {
		parts = append(parts, result.Stdout)
	}

	if result.Stderr != "" {
		parts = append(parts, fmt.Sprintf("[stderr]\n%s", result.Stderr))
	}

	parts = append(parts, fmt.Sprintf("exit status %d", result.ExitCode))

	output := strings.Join(parts, "\n")

	// Summary
	summary := fmt.Sprintf("exit %d", result.ExitCode)
	if result.Error != nil {
		summary = fmt.Sprintf("error: %s", result.Error.Error())
	}

	return tools.StringResultWithSummary(output, summary), nil
}

func timeoutFromInt(seconds int) time.Duration {
	if seconds < 1 {
		return 60 * time.Second
	}
	if seconds > 1800 {
		return 1800 * time.Second
	}
	return time.Duration(seconds) * time.Second
}
