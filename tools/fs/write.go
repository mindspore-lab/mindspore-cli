package fs

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/vigo999/ms-cli/integrations/llm"
	"github.com/vigo999/ms-cli/tools"
)

// WriteTool writes or creates file contents.
type WriteTool struct {
	workDir string
}

// NewWriteTool creates a new write tool.
func NewWriteTool(workDir string) *WriteTool {
	return &WriteTool{workDir: workDir}
}

// Name returns the tool name.
func (t *WriteTool) Name() string {
	return "write"
}

// Description returns the tool description.
func (t *WriteTool) Description() string {
	return "Create a new file or overwrite an existing file with new content."
}

// Schema returns the tool parameter schema.
func (t *WriteTool) Schema() llm.ToolSchema {
	return llm.ToolSchema{
		Type: "object",
		Properties: map[string]llm.Property{
			"path": {
				Type:        "string",
				Description: "Relative path to the file to write",
			},
			"content": {
				Type:        "string",
				Description: "Content to write to the file",
			},
		},
		Required: []string{"path", "content"},
	}
}

type writeParams struct {
	Path    string `json:"path"`
	Content string `json:"content"`
}

// Execute executes the write tool.
func (t *WriteTool) Execute(ctx context.Context, params json.RawMessage) (*tools.Result, error) {
	var p writeParams
	if err := tools.ParseParams(params, &p); err != nil {
		return tools.ErrorResult(err), nil
	}

	fullPath, err := resolveSafePath(t.workDir, p.Path)
	if err != nil {
		return tools.ErrorResult(err), nil
	}

	// Ensure parent directory exists
	dir := filepath.Dir(fullPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return tools.ErrorResultf("create directory: %w", err), nil
	}

	// Check if file already exists
	exists := false
	if _, err := os.Stat(fullPath); err == nil {
		exists = true
	}

	// Write file
	if err := os.WriteFile(fullPath, []byte(p.Content), 0644); err != nil {
		return tools.ErrorResultf("write file: %w", err), nil
	}

	// Build result
	lines := strings.Count(p.Content, "\n")
	if !strings.HasSuffix(p.Content, "\n") && p.Content != "" {
		lines++
	}

	action := "Created"
	if exists {
		action = "Updated"
	}

	content := fmt.Sprintf("%s: %s\n+ %s", action, p.Path, p.Content)
	summary := fmt.Sprintf("%s %d lines", action, lines)

	return tools.StringResultWithSummary(content, summary), nil
}
