package fs

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
	"github.com/mindspore-lab/mindspore-cli/tools"
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
	return "Create a new file or overwrite an existing file with new content. Arguments must be a JSON object containing required fields path and content."
}

// Schema returns the tool parameter schema.
func (t *WriteTool) Schema() llm.ToolSchema {
	return llm.ToolSchema{
		Type: "object",
		Properties: map[string]llm.Property{
			"path": {
				Type:        "string",
				Description: "Required. Relative path to the file to write. Use this exact field name; do not use file_path or filename.",
			},
			"content": {
				Type:        "string",
				Description: "Required. Full content to write to the file.",
			},
		},
		Required: []string{"path", "content"},
	}
}

type writeParams struct {
	Path     string `json:"path"`
	FilePath string `json:"file_path"`
	Filename string `json:"filename"`
	Content  string `json:"content"`
}

// Execute executes the write tool.
func (t *WriteTool) Execute(ctx context.Context, params json.RawMessage) (*tools.Result, error) {
	var p writeParams
	if err := tools.ParseParams(params, &p); err != nil {
		return tools.ErrorResult(err), nil
	}

	path := strings.TrimSpace(p.Path)
	if path == "" {
		path = strings.TrimSpace(p.FilePath)
	}
	if path == "" {
		path = strings.TrimSpace(p.Filename)
	}
	if path == "" {
		return tools.ErrorResultf(`invalid_write_args: missing path (required keys: "path","content"; aliases "file_path"/"filename" are fallback only)`), nil
	}

	fullPath, err := resolveSafePath(t.workDir, path)
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

	content := fmt.Sprintf("%s: %s\n+ %s", action, path, p.Content)
	summary := fmt.Sprintf("%s %d lines", action, lines)

	return tools.StringResultWithSummary(content, summary), nil
}
