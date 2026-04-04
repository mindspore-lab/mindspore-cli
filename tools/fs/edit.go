package fs

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
	"github.com/mindspore-lab/mindspore-cli/tools"
)

// EditTool edits file contents by replacing text.
type EditTool struct {
	workDir string
}

// NewEditTool creates a new edit tool.
func NewEditTool(workDir string) *EditTool {
	return &EditTool{workDir: workDir}
}

// Name returns the tool name.
func (t *EditTool) Name() string {
	return "edit"
}

// Description returns the tool description.
func (t *EditTool) Description() string {
	return "Edit a file by replacing specific text. Use this for making targeted changes. The old_string must match exactly including whitespace."
}

// Schema returns the tool parameter schema.
func (t *EditTool) Schema() llm.ToolSchema {
	return llm.ToolSchema{
		Type: "object",
		Properties: map[string]llm.Property{
			"path": {
				Type:        "string",
				Description: "Relative path to the file to edit",
			},
			"old_string": {
				Type:        "string",
				Description: "Exact text to replace (must match exactly including whitespace and newlines)",
			},
			"new_string": {
				Type:        "string",
				Description: "New text to replace the old_string with",
			},
		},
		Required: []string{"path", "old_string", "new_string"},
	}
}

type editParams struct {
	Path      string `json:"path"`
	OldString string `json:"old_string"`
	NewString string `json:"new_string"`
}

// Execute executes the edit tool.
func (t *EditTool) Execute(ctx context.Context, params json.RawMessage) (*tools.Result, error) {
	var p editParams
	if err := tools.ParseParams(params, &p); err != nil {
		return tools.ErrorResult(err), nil
	}

	fullPath, err := resolveSafePath(t.workDir, p.Path)
	if err != nil {
		return tools.ErrorResult(err), nil
	}

	// Read existing file
	content, err := os.ReadFile(fullPath)
	if err != nil {
		if os.IsNotExist(err) {
			return tools.ErrorResultf("file not found: %s", p.Path), nil
		}
		return tools.ErrorResultf("read file: %w", err), nil
	}

	contentStr := string(content)

	// Check if old_string exists
	if !strings.Contains(contentStr, p.OldString) {
		// Try to find similar content
		return tools.ErrorResultf("old_string not found in file. The text must match exactly (including whitespace and newlines)"), nil
	}

	// Count occurrences
	occurrences := strings.Count(contentStr, p.OldString)
	if occurrences > 1 {
		return tools.ErrorResultf("old_string appears %d times in the file. Please provide more context to make a unique match", occurrences), nil
	}

	// Replace
	newContent := strings.Replace(contentStr, p.OldString, p.NewString, 1)

	// Write back
	if err := os.WriteFile(fullPath, []byte(newContent), 0644); err != nil {
		return tools.ErrorResultf("write file: %w", err), nil
	}

	// Build diff-style result
	oldLines := strings.Count(p.OldString, "\n")
	newLines := strings.Count(p.NewString, "\n")
	if !strings.HasSuffix(p.OldString, "\n") && p.OldString != "" {
		oldLines++
	}
	if !strings.HasSuffix(p.NewString, "\n") && p.NewString != "" {
		newLines++
	}

	result := fmt.Sprintf("Edited: %s\n- %s\n+ %s", p.Path, p.OldString, p.NewString)
	summary := fmt.Sprintf("%d lines → %d lines", oldLines, newLines)

	return tools.StringResultWithSummary(result, summary), nil
}
