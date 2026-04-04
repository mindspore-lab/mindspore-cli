// Package shell provides the LLM-callable shell tool.
// Actual command execution is delegated to runtime/shell.
package shell

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
	rshell "github.com/mindspore-lab/mindspore-cli/runtime/shell"
	"github.com/mindspore-lab/mindspore-cli/tools"
)

// ShellTool wraps shell execution as an LLM-callable Tool.
type ShellTool struct {
	runner *rshell.Runner
}

// NewShellTool creates a new shell tool backed by a runtime shell runner.
func NewShellTool(runner *rshell.Runner) *ShellTool {
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
	return t.ExecuteStream(ctx, params, nil)
}

// ExecuteStream executes the shell tool and emits live command output updates.
func (t *ShellTool) ExecuteStream(ctx context.Context, params json.RawMessage, emit func(tools.StreamEvent)) (*tools.Result, error) {
	var p shellParams
	if err := tools.ParseParams(params, &p); err != nil {
		return tools.ErrorResult(err), nil
	}

	command := strings.TrimSpace(p.Command)
	if command == "" {
		return tools.ErrorResultf("command is required"), nil
	}

	if p.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeoutFromInt(p.Timeout))
		defer cancel()
	}

	if emit != nil {
		emit(tools.StreamEvent{Type: tools.StreamEventStarted})
	}

	result, err := t.runner.RunStream(ctx, command, func(chunk rshell.OutputChunk) {
		if emit == nil {
			return
		}
		line := chunk.Text
		if chunk.Stream == rshell.StreamStderr {
			line = "[stderr] " + line
		}
		emit(tools.StreamEvent{
			Type:    tools.StreamEventOutput,
			Message: line,
		})
	})
	if err != nil {
		return tools.ErrorResultf("execute command: %w", err), nil
	}

	var parts []string
	if result.Stdout != "" {
		parts = append(parts, result.Stdout)
	}

	if result.Stderr != "" {
		parts = append(parts, fmt.Sprintf("[stderr]\n%s", result.Stderr))
	}
	if len(parts) == 0 {
		parts = append(parts, "(No output)")
	}

	output := strings.Join(parts, "\n")

	summary := "completed"
	if result.ExitCode != 0 {
		summary = fmt.Sprintf("exit %d", result.ExitCode)
	}
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
