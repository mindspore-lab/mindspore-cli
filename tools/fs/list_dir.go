package fs

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
	"github.com/mindspore-lab/mindspore-cli/tools"
)

// ListDirTool lists directory contents for structure exploration.
type ListDirTool struct {
	workDir string
}

// NewListDirTool creates a new list_dir tool.
func NewListDirTool(workDir string) *ListDirTool {
	return &ListDirTool{workDir: workDir}
}

// Name returns the tool name.
func (t *ListDirTool) Name() string {
	return "list_dir"
}

// Description returns the tool description.
func (t *ListDirTool) Description() string {
	return "List directory contents and shallow tree structure. Use this to answer questions like what is in the current directory or summarize the code structure. Prefer this over glob or shell ls/find/tree when exploring directory layout. Use glob to find files by path pattern, grep to search file contents, and read after you know the exact file path."
}

// Schema returns the tool parameter schema.
func (t *ListDirTool) Schema() llm.ToolSchema {
	return llm.ToolSchema{
		Type: "object",
		Properties: map[string]llm.Property{
			"path": {
				Type:        "string",
				Description: "Relative directory path to list. Use '.' for the current directory.",
			},
			"depth": {
				Type:        "integer",
				Description: "Maximum directory depth to include (default: 2, minimum: 1).",
			},
			"include_hidden": {
				Type:        "boolean",
				Description: "Whether to include dotfiles and dot-directories. Default: false.",
			},
			"offset": {
				Type:        "integer",
				Description: "Entry number to start returning from (1-indexed, 0 means from start).",
			},
			"limit": {
				Type:        "integer",
				Description: "Maximum number of entries to return (defaults to 100, maximum 100).",
			},
		},
	}
}

type listDirParams struct {
	Path          string `json:"path"`
	Depth         int    `json:"depth"`
	IncludeHidden bool   `json:"include_hidden"`
	Offset        int    `json:"offset"`
	Limit         int    `json:"limit"`
}

type listDirEntry struct {
	display string
	sortKey string
}

// Execute executes the list_dir tool.
func (t *ListDirTool) Execute(ctx context.Context, params json.RawMessage) (*tools.Result, error) {
	var p listDirParams
	if err := tools.ParseParams(params, &p); err != nil {
		return tools.ErrorResult(err), nil
	}

	basePath := "."
	if strings.TrimSpace(p.Path) != "" {
		basePath = strings.TrimSpace(p.Path)
	}
	fullPath, err := resolveSafePath(t.workDir, basePath)
	if err != nil {
		return tools.ErrorResult(err), nil
	}

	info, err := os.Stat(fullPath)
	if err != nil {
		if os.IsNotExist(err) {
			return tools.ErrorResultf("path not found: %s", basePath), nil
		}
		return tools.ErrorResult(err), nil
	}
	if !info.IsDir() {
		return tools.ErrorResultf("path is not a directory: %s", basePath), nil
	}

	depth := p.Depth
	if depth <= 0 {
		depth = 2
	}

	entries, err := t.collectEntries(fullPath, depth, p.IncludeHidden)
	if err != nil {
		return tools.ErrorResult(err), nil
	}
	if len(entries) == 0 {
		return tools.StringResultWithSummary("No entries found", "0 entries"), nil
	}

	sort.Slice(entries, func(i, j int) bool {
		return entries[i].sortKey < entries[j].sortKey
	})

	lines := make([]string, 0, len(entries))
	for _, entry := range entries {
		lines = append(lines, entry.display)
	}

	effectiveLimit := normalizeSearchResultLimit(p.Limit)
	totalEntries := len(lines)
	lines = sliceWithOffsetLimit(lines, p.Offset, effectiveLimit)

	summary := pagedSearchSummary(totalEntries, p.Offset, len(lines), "entries")
	result := buildSearchResultContent(summary, lines)
	return tools.StringResultWithSummary(result, summary), nil
}

func (t *ListDirTool) collectEntries(root string, maxDepth int, includeHidden bool) ([]listDirEntry, error) {
	var entries []listDirEntry

	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if path == root {
			return nil
		}

		name := d.Name()
		if isIgnoredGitName(name) {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if !includeHidden && strings.HasPrefix(name, ".") {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		relPath, relErr := filepath.Rel(root, path)
		if relErr != nil {
			return nil
		}
		depth := strings.Count(filepath.ToSlash(relPath), "/") + 1
		if depth > maxDepth {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		display := filepath.ToSlash(relPath)
		if d.IsDir() {
			display += "/"
		}
		entries = append(entries, listDirEntry{
			display: display,
			sortKey: display,
		})
		return nil
	})
	if err != nil {
		return nil, err
	}

	return entries, nil
}
