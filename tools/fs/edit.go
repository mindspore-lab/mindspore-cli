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
	return "Edit a file by replacing one exact text block. To edit files use edit instead of sed or awk. Use this for targeted file changes after reading the file first. The old_string must match exactly, including whitespace and newlines. Do not use shell redirection, tee, sed -i, or heredoc as a substitute for edit."
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
	oldPos := strings.Index(contentStr, p.OldString)
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

	out := tools.StringResultWithSummary(result, summary)
	out.Meta = buildEditDiffMeta(p.Path, contentStr, newContent, p.OldString, p.NewString, oldPos)
	return out, nil
}

func buildEditDiffMeta(path, before, after, oldString, newString string, oldPos int) map[string]any {
	if oldPos < 0 {
		return nil
	}
	const contextLines = 2

	beforeLines := splitDiffLines(before)
	afterLines := splitDiffLines(after)
	oldParts := splitDiffLines(oldString)
	newParts := splitDiffLines(newString)

	oldStart := strings.Count(before[:oldPos], "\n") + 1
	oldCount := max(1, len(oldParts))
	newCount := max(1, len(newParts))
	newStart := oldStart

	oldCtxStart := max(1, oldStart-contextLines)
	oldCtxEnd := min(len(beforeLines), oldStart+oldCount-1+contextLines)
	newCtxStart := max(1, newStart-contextLines)
	newCtxEnd := min(len(afterLines), newStart+newCount-1+contextLines)

	lines := make([]string, 0, (oldCtxEnd-oldCtxStart+1)+(newCtxEnd-newCtxStart+1)+8)
	for i := oldCtxStart; i < oldStart; i++ {
		lines = append(lines, " "+beforeLines[i-1])
	}
	lines = append(lines, lineDiff(oldParts, newParts)...)
	for i := newStart + newCount; i <= newCtxEnd; i++ {
		lines = append(lines, " "+afterLines[i-1])
	}

	header := fmt.Sprintf("@@ -%d,%d +%d,%d @@", oldCtxStart, oldCtxEnd-oldCtxStart+1, newCtxStart, newCtxEnd-newCtxStart+1)
	return map[string]any{
		"edit_diff": map[string]any{
			"path":   path,
			"header": header,
			"lines":  lines,
		},
	}
}

func splitDiffLines(s string) []string {
	normalized := strings.ReplaceAll(s, "\r\n", "\n")
	if normalized == "" {
		return []string{}
	}
	lines := strings.Split(normalized, "\n")
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	return lines
}

func lineDiff(oldLines, newLines []string) []string {
	oldN, newN := len(oldLines), len(newLines)
	lcs := make([][]int, oldN+1)
	for i := range lcs {
		lcs[i] = make([]int, newN+1)
	}
	for i := oldN - 1; i >= 0; i-- {
		for j := newN - 1; j >= 0; j-- {
			if oldLines[i] == newLines[j] {
				lcs[i][j] = lcs[i+1][j+1] + 1
				continue
			}
			if lcs[i+1][j] >= lcs[i][j+1] {
				lcs[i][j] = lcs[i+1][j]
			} else {
				lcs[i][j] = lcs[i][j+1]
			}
		}
	}

	out := make([]string, 0, oldN+newN)
	i, j := 0, 0
	for i < oldN && j < newN {
		switch {
		case oldLines[i] == newLines[j]:
			out = append(out, " "+oldLines[i])
			i++
			j++
		case lcs[i+1][j] >= lcs[i][j+1]:
			out = append(out, "-"+oldLines[i])
			i++
		default:
			out = append(out, "+"+newLines[j])
			j++
		}
	}
	for ; i < oldN; i++ {
		out = append(out, "-"+oldLines[i])
	}
	for ; j < newN; j++ {
		out = append(out, "+"+newLines[j])
	}
	return out
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
