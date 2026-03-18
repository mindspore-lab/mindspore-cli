package remote

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/vigo999/ms-cli/integrations/llm"
	"github.com/vigo999/ms-cli/runtime/ssh"
	"github.com/vigo999/ms-cli/tools"
)

// RemoteShellTool provides remote shell execution via SSH.
type RemoteShellTool struct {
	pool *ssh.Pool
}

// NewShellTool creates a new remote shell tool.
func NewShellTool(pool *ssh.Pool) *RemoteShellTool {
	return &RemoteShellTool{pool: pool}
}

// Name returns the tool name.
func (t *RemoteShellTool) Name() string {
	return "remote_shell"
}

// Description returns the tool description.
func (t *RemoteShellTool) Description() string {
	return `Execute shell commands on a remote host via SSH.

This tool allows you to run commands on remote machines, with automatic session state management.
The connection is kept alive and reused across multiple commands to the same host.

State preservation:
- Working directory is maintained between commands (cd commands update the cached state)
- Environment variables set via export are cached and applied to subsequent commands

Use pre-configured hosts from config.yaml by name, or specify connection details dynamically.`
}

// Schema returns the JSON schema for tool parameters.
func (t *RemoteShellTool) Schema() llm.ToolSchema {
	return llm.ToolSchema{
		Type: "object",
		Properties: map[string]llm.Property{
			"host": {
				Type:        "string",
				Description: "Target host address or pre-configured alias (e.g., 'gpu-server-1')",
			},
			"command": {
				Type:        "string",
				Description: "Shell command to execute on the remote host",
			},
			"timeout": {
				Type:        "integer",
				Description: "Command timeout in seconds (optional, overrides config)",
			},
			"user": {
				Type:        "string",
				Description: "SSH username (optional, overrides config)",
			},
			"key_path": {
				Type:        "string",
				Description: "Path to SSH private key file (optional, overrides config)",
			},
			"password": {
				Type:        "string",
				Description: "SSH password (optional, overrides config, not recommended for production)",
			},
		},
		Required: []string{"host", "command"},
	}
}

// shellParams represents the parameters for remote_shell tool.
type shellParams struct {
	Host     string `json:"host"`
	Command  string `json:"command"`
	Timeout  int    `json:"timeout,omitempty"`
	User     string `json:"user,omitempty"`
	KeyPath  string `json:"key_path,omitempty"`
	Password string `json:"password,omitempty"`
}

// Execute runs the remote shell command.
func (t *RemoteShellTool) Execute(ctx context.Context, params json.RawMessage) (*tools.Result, error) {
	var args shellParams
	if err := tools.ParseParams(params, &args); err != nil {
		return nil, fmt.Errorf("failed to parse params: %w", err)
	}

	if args.Host == "" {
		return tools.ErrorResultf("host is required"), nil
	}
	if args.Command == "" {
		return tools.ErrorResultf("command is required"), nil
	}

	// Build connection options (dynamic overrides pre-configured)
	opts := ssh.ConnectOptions{
		Host:     args.Host,
		User:     args.User,
		Password: args.Password,
		KeyPath:  args.KeyPath,
	}

	if args.Timeout > 0 {
		opts.Timeout = time.Duration(args.Timeout) * time.Second
	}

	// Get or create session
	session, err := t.pool.Get(opts)
	if err != nil {
		return tools.ErrorResultf("failed to connect to %s: %v", args.Host, err), nil
	}

	// Execute command with context timeout if specified
	if args.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, time.Duration(args.Timeout)*time.Second)
		defer cancel()
	}

	result, err := session.Run(ctx, args.Command)
	if err != nil {
		return tools.ErrorResultf("command execution failed: %v", err), nil
	}

	// Build summary
	summary := fmt.Sprintf("exit=%d", result.ExitCode)
	if result.Duration > 0 {
		summary += fmt.Sprintf(" time=%v", result.Duration.Round(time.Millisecond))
	}

	// Build content
	content := result.Combined()
	if content == "" {
		content = "(no output)"
	}

	return &tools.Result{
		Content: content,
		Summary: summary,
		Error:   nil,
	}, nil
}
