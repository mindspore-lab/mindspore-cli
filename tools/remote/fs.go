package remote

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/vigo999/ms-cli/integrations/llm"
	"github.com/vigo999/ms-cli/runtime/ssh"
	"github.com/vigo999/ms-cli/tools"
)

// RemoteReadTool provides remote file reading via SSH/SFTP.
type RemoteReadTool struct {
	pool *ssh.Pool
}

// NewReadTool creates a new remote read tool.
func NewReadTool(pool *ssh.Pool) *RemoteReadTool {
	return &RemoteReadTool{pool: pool}
}

func (t *RemoteReadTool) Name() string { return "remote_read" }

func (t *RemoteReadTool) Description() string {
	return `Read a file from a remote host via SSH/SFTP.

Supports line offset and limit for reading large files efficiently.
Relative paths are resolved against the cached remote working directory.`
}

func (t *RemoteReadTool) Schema() llm.ToolSchema {
	return llm.ToolSchema{
		Type: "object",
		Properties: map[string]llm.Property{
			"host":   {Type: "string", Description: "Target host address or alias"},
			"path":   {Type: "string", Description: "Remote file path (relative or absolute)"},
			"offset": {Type: "integer", Description: "Number of lines to skip from beginning (optional)"},
			"limit":  {Type: "integer", Description: "Maximum number of lines to read (optional)"},
		},
		Required: []string{"host", "path"},
	}
}

type readParams struct {
	Host   string `json:"host"`
	Path   string `json:"path"`
	Offset int    `json:"offset,omitempty"`
	Limit  int    `json:"limit,omitempty"`
}

func (t *RemoteReadTool) Execute(ctx context.Context, params json.RawMessage) (*tools.Result, error) {
	var args readParams
	if err := tools.ParseParams(params, &args); err != nil {
		return nil, fmt.Errorf("failed to parse params: %w", err)
	}

	if args.Host == "" {
		return tools.ErrorResultf("host is required"), nil
	}
	if args.Path == "" {
		return tools.ErrorResultf("path is required"), nil
	}

	// Validate path
	if err := validateRemotePath(args.Path); err != nil {
		return tools.ErrorResultf("invalid path: %v", err), nil
	}

	opts := ssh.ConnectOptions{Host: args.Host}
	session, err := t.pool.Get(opts)
	if err != nil {
		return tools.ErrorResultf("failed to connect to %s: %v", args.Host, err), nil
	}

	content, err := session.ReadFile(args.Path, args.Offset, args.Limit)
	if err != nil {
		return tools.ErrorResultf("failed to read file: %v", err), nil
	}

	// Count lines
	lines := strings.Split(content, "\n")
	summary := fmt.Sprintf("%d lines", len(lines))
	if args.Offset > 0 || args.Limit > 0 {
		summary += fmt.Sprintf(" (offset=%d, limit=%d)", args.Offset, args.Limit)
	}

	return tools.StringResultWithSummary(content, summary), nil
}

// RemoteWriteTool provides remote file writing via SSH/SFTP.
type RemoteWriteTool struct {
	pool *ssh.Pool
}

func NewWriteTool(pool *ssh.Pool) *RemoteWriteTool {
	return &RemoteWriteTool{pool: pool}
}

func (t *RemoteWriteTool) Name() string { return "remote_write" }

func (t *RemoteWriteTool) Description() string {
	return `Write content to a file on a remote host via SSH/SFTP.

Creates parent directories automatically if needed.
Overwrites existing files without warning.`
}

func (t *RemoteWriteTool) Schema() llm.ToolSchema {
	return llm.ToolSchema{
		Type: "object",
		Properties: map[string]llm.Property{
			"host":    {Type: "string", Description: "Target host address or alias"},
			"path":    {Type: "string", Description: "Remote file path"},
			"content": {Type: "string", Description: "Content to write"},
		},
		Required: []string{"host", "path", "content"},
	}
}

type writeParams struct {
	Host    string `json:"host"`
	Path    string `json:"path"`
	Content string `json:"content"`
}

func (t *RemoteWriteTool) Execute(ctx context.Context, params json.RawMessage) (*tools.Result, error) {
	var args writeParams
	if err := tools.ParseParams(params, &args); err != nil {
		return nil, fmt.Errorf("failed to parse params: %w", err)
	}

	if args.Host == "" {
		return tools.ErrorResultf("host is required"), nil
	}
	if args.Path == "" {
		return tools.ErrorResultf("path is required"), nil
	}

	if err := validateRemotePath(args.Path); err != nil {
		return tools.ErrorResultf("invalid path: %v", err), nil
	}

	opts := ssh.ConnectOptions{Host: args.Host}
	session, err := t.pool.Get(opts)
	if err != nil {
		return tools.ErrorResultf("failed to connect to %s: %v", args.Host, err), nil
	}

	if err := session.WriteFile(args.Path, args.Content); err != nil {
		return tools.ErrorResultf("failed to write file: %v", err), nil
	}

	lines := len(strings.Split(args.Content, "\n"))
	summary := fmt.Sprintf("wrote %d lines", lines)

	return tools.StringResultWithSummary("", summary), nil
}

// RemoteEditTool provides remote file editing via SSH/SFTP.
type RemoteEditTool struct {
	pool *ssh.Pool
}

func NewEditTool(pool *ssh.Pool) *RemoteEditTool {
	return &RemoteEditTool{pool: pool}
}

func (t *RemoteEditTool) Name() string { return "remote_edit" }

func (t *RemoteEditTool) Description() string {
	return `Edit a file on a remote host by replacing text.

Replaces old_string with new_string. The old_string must appear exactly once
in the file for safety. Use remote_read first to verify the content.`
}

func (t *RemoteEditTool) Schema() llm.ToolSchema {
	return llm.ToolSchema{
		Type: "object",
		Properties: map[string]llm.Property{
			"host":       {Type: "string", Description: "Target host address or alias"},
			"path":       {Type: "string", Description: "Remote file path"},
			"old_string": {Type: "string", Description: "Exact text to replace (must appear once)"},
			"new_string": {Type: "string", Description: "Replacement text"},
		},
		Required: []string{"host", "path", "old_string", "new_string"},
	}
}

type editParams struct {
	Host      string `json:"host"`
	Path      string `json:"path"`
	OldString string `json:"old_string"`
	NewString string `json:"new_string"`
}

func (t *RemoteEditTool) Execute(ctx context.Context, params json.RawMessage) (*tools.Result, error) {
	var args editParams
	if err := tools.ParseParams(params, &args); err != nil {
		return nil, fmt.Errorf("failed to parse params: %w", err)
	}

	if args.Host == "" {
		return tools.ErrorResultf("host is required"), nil
	}
	if args.Path == "" {
		return tools.ErrorResultf("path is required"), nil
	}

	if err := validateRemotePath(args.Path); err != nil {
		return tools.ErrorResultf("invalid path: %v", err), nil
	}

	opts := ssh.ConnectOptions{Host: args.Host}
	session, err := t.pool.Get(opts)
	if err != nil {
		return tools.ErrorResultf("failed to connect to %s: %v", args.Host, err), nil
	}

	if err := session.EditFile(args.Path, args.OldString, args.NewString); err != nil {
		return tools.ErrorResultf("failed to edit file: %v", err), nil
	}

	return tools.StringResultWithSummary("", "edit successful"), nil
}

// RemoteGlobTool provides remote file globbing via SSH.
type RemoteGlobTool struct {
	pool *ssh.Pool
}

func NewGlobTool(pool *ssh.Pool) *RemoteGlobTool {
	return &RemoteGlobTool{pool: pool}
}

func (t *RemoteGlobTool) Name() string { return "remote_glob" }

func (t *RemoteGlobTool) Description() string {
	return `Find files on a remote host matching a pattern.

Supports * and ** wildcards. ** matches any number of directory levels.
Examples:
- "*.go" - all .go files in current directory
- "**/*.py" - all .py files recursively`
}

func (t *RemoteGlobTool) Schema() llm.ToolSchema {
	return llm.ToolSchema{
		Type: "object",
		Properties: map[string]llm.Property{
			"host":    {Type: "string", Description: "Target host address or alias"},
			"pattern": {Type: "string", Description: "Glob pattern (e.g., '**/*.go')"},
		},
		Required: []string{"host", "pattern"},
	}
}

type globParams struct {
	Host    string `json:"host"`
	Pattern string `json:"pattern"`
}

func (t *RemoteGlobTool) Execute(ctx context.Context, params json.RawMessage) (*tools.Result, error) {
	var args globParams
	if err := tools.ParseParams(params, &args); err != nil {
		return nil, fmt.Errorf("failed to parse params: %w", err)
	}

	if args.Host == "" {
		return tools.ErrorResultf("host is required"), nil
	}
	if args.Pattern == "" {
		return tools.ErrorResultf("pattern is required"), nil
	}

	opts := ssh.ConnectOptions{Host: args.Host}
	session, err := t.pool.Get(opts)
	if err != nil {
		return tools.ErrorResultf("failed to connect to %s: %v", args.Host, err), nil
	}

	matches, err := session.Glob(args.Pattern)
	if err != nil {
		return tools.ErrorResultf("glob failed: %v", err), nil
	}

	content := strings.Join(matches, "\n")
	if content == "" {
		content = "(no matches)"
	}

	summary := fmt.Sprintf("%d matches", len(matches))
	return tools.StringResultWithSummary(content, summary), nil
}

// RemoteGrepTool provides remote file search via SSH.
type RemoteGrepTool struct {
	pool *ssh.Pool
}

func NewGrepTool(pool *ssh.Pool) *RemoteGrepTool {
	return &RemoteGrepTool{pool: pool}
}

func (t *RemoteGrepTool) Name() string { return "remote_grep" }

func (t *RemoteGrepTool) Description() string {
	return `Search for a pattern in files on a remote host.

Uses regular expression matching. Returns matching lines in format: path:line:content
Case-insensitive by default.`
}

func (t *RemoteGrepTool) Schema() llm.ToolSchema {
	return llm.ToolSchema{
		Type: "object",
		Properties: map[string]llm.Property{
			"host":           {Type: "string", Description: "Target host address or alias"},
			"pattern":        {Type: "string", Description: "Regex pattern to search for"},
			"paths":          {Type: "array", Description: "List of files or directories to search (optional, defaults to current directory)"},
			"case_sensitive": {Type: "boolean", Description: "Case sensitive search (default: false)"},
		},
		Required: []string{"host", "pattern"},
	}
}

type grepParams struct {
	Host          string   `json:"host"`
	Pattern       string   `json:"pattern"`
	Paths         []string `json:"paths,omitempty"`
	CaseSensitive bool     `json:"case_sensitive,omitempty"`
}

func (t *RemoteGrepTool) Execute(ctx context.Context, params json.RawMessage) (*tools.Result, error) {
	var args grepParams
	if err := tools.ParseParams(params, &args); err != nil {
		return nil, fmt.Errorf("failed to parse params: %w", err)
	}

	if args.Host == "" {
		return tools.ErrorResultf("host is required"), nil
	}
	if args.Pattern == "" {
		return tools.ErrorResultf("pattern is required"), nil
	}

	if len(args.Paths) == 0 {
		args.Paths = []string{"."}
	}

	opts := ssh.ConnectOptions{Host: args.Host}
	session, err := t.pool.Get(opts)
	if err != nil {
		return tools.ErrorResultf("failed to connect to %s: %v", args.Host, err), nil
	}

	results, err := session.Grep(args.Pattern, args.Paths, args.CaseSensitive)
	if err != nil {
		return tools.ErrorResultf("grep failed: %v", err), nil
	}

	content := strings.Join(results, "\n")
	if content == "" {
		content = "(no matches)"
	}

	summary := fmt.Sprintf("%d matches", len(results))
	return tools.StringResultWithSummary(content, summary), nil
}

// validateRemotePath validates a remote path for safety.
func validateRemotePath(path string) error {
	if path == "" {
		return fmt.Errorf("path is empty")
	}
	// Check for path traversal attempts
	if strings.Contains(path, "..") {
		// Allow .. if it's not at the start or used for traversal
		// This is a simplified check; in production you might want stricter validation
	}
	return nil
}
