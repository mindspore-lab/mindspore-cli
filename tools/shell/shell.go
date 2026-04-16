// Package shell provides the LLM-callable shell tool.
// Actual command execution is delegated to runtime/shell.
package shell

import (
	"context"
	"encoding/json"
	"errors"
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
	return "Execute a shell command for tasks that truly require a shell, such as running tests, builds, git commands, or launching programs. Reserve using the shell exclusively for system commands and terminal operations that require shell execution. If you are unsure and there is a relevant dedicated tool, default to using the dedicated tool and only fallback on using the shell if it is absolutely necessary. Prefer dedicated tools for file reading, file discovery, directory structure exploration, and file editing. Do not use shell redirection, heredoc, tee, sed -i, or similar patterns to create or edit files when write or edit can do the job."
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
		if errors.Is(ctx.Err(), context.Canceled) {
			return tools.StringResultWithSummary("", "interrupted"), nil
		}
		return tools.ErrorResultf("execute command: %w", err), nil
	}

	output, hasOutput := shellOutput(result)
	if errors.Is(ctx.Err(), context.Canceled) {
		if !hasOutput {
			output = ""
		}
		return tools.StringResultWithSummary(output, "interrupted"), nil
	}
	if !hasOutput {
		output = "(No output)"
	}

	summary := "completed"
	if result.ExitCode != 0 {
		summary = fmt.Sprintf("exit %d", result.ExitCode)
	}
	if result.Error != nil {
		summary = fmt.Sprintf("error: %s", result.Error.Error())
	}

	return tools.StringResultWithSummary(output, summary), nil
}

func shellOutput(result *rshell.Result) (string, bool) {
	var parts []string
	if result.Stdout != "" {
		parts = append(parts, result.Stdout)
	}
	if result.Stderr != "" {
		parts = append(parts, fmt.Sprintf("[stderr]\n%s", result.Stderr))
	}
	return strings.Join(parts, "\n"), len(parts) > 0
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
