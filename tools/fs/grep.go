package fs

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"

	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
	"github.com/mindspore-lab/mindspore-cli/tools"
)

// GrepTool searches for patterns in files.
type GrepTool struct {
	workDir string
}

// NewGrepTool creates a new grep tool.
func NewGrepTool(workDir string) *GrepTool {
	return &GrepTool{workDir: workDir}
}

// Name returns the tool name.
func (t *GrepTool) Name() string {
	return "grep"
}

// Description returns the tool description.
func (t *GrepTool) Description() string {
	return "Search for patterns in files using regular expressions. Returns matching lines with file names and line numbers."
}

// Schema returns the tool parameter schema.
func (t *GrepTool) Schema() llm.ToolSchema {
	return llm.ToolSchema{
		Type: "object",
		Properties: map[string]llm.Property{
			"pattern": {
				Type:        "string",
				Description: "Regular expression pattern to search for (e.g., 'func.*main', 'TODO|FIXME')",
			},
			"path": {
				Type:        "string",
				Description: "Directory or file to search in (default: current directory)",
			},
			"include": {
				Type:        "string",
				Description: "File pattern to include using glob syntax (e.g., '*.go', '*.md')",
			},
			"case_sensitive": {
				Type:        "boolean",
				Description: "Whether the search is case sensitive (default: true)",
			},
			"offset": {
				Type:        "integer",
				Description: "Match number to start returning from (1-indexed, 0 means from start)",
			},
			"limit": {
				Type:        "integer",
				Description: "Maximum number of matches to return (defaults to 100, maximum 100)",
			},
		},
		Required: []string{"pattern"},
	}
}

type grepParams struct {
	Pattern       string `json:"pattern"`
	Path          string `json:"path"`
	Include       string `json:"include"`
	CaseSensitive bool   `json:"case_sensitive"`
	Offset        int    `json:"offset"`
	Limit         int    `json:"limit"`
}

// Match represents a single grep match.
type Match struct {
	File   string
	Line   int
	Column int
	Text   string
}

// Execute executes the grep tool.
func (t *GrepTool) Execute(ctx context.Context, params json.RawMessage) (*tools.Result, error) {
	var p grepParams
	if err := tools.ParseParams(params, &p); err != nil {
		return tools.ErrorResult(err), nil
	}

	// Default case sensitive
	if !p.CaseSensitive {
		// Pattern will be handled with case-insensitive flag
	}

	// Resolve search path
	searchPath := "."
	if p.Path != "" {
		searchPath = p.Path
	}
	fullPath, err := resolveSafePath(t.workDir, searchPath)
	if err != nil {
		return tools.ErrorResult(err), nil
	}

	// Compile regex
	pattern := p.Pattern
	if !p.CaseSensitive {
		pattern = "(?i)" + pattern
	}
	re, err := regexp.Compile(pattern)
	if err != nil {
		return tools.ErrorResultf("invalid pattern: %w", err), nil
	}

	// Find files and search
	matches, err := t.grep(ctx, fullPath, p.Include, re)
	if err != nil {
		return tools.ErrorResult(err), nil
	}

	// Format results
	if len(matches) == 0 {
		return tools.StringResultWithSummary("No matches found", "0 matches"), nil
	}
	effectiveLimit := normalizeSearchResultLimit(p.Limit)
	totalMatches := len(matches)
	matches = sliceWithOffsetLimit(matches, p.Offset, effectiveLimit)

	var lines []string
	for _, m := range matches {
		relPath, _ := filepath.Rel(t.workDir, m.File)
		lines = append(lines, fmt.Sprintf("%s:%d:%s", relPath, m.Line, m.Text))
	}

	summary := pagedSearchSummary(totalMatches, p.Offset, len(matches), "matches")
	result := buildSearchResultContent(summary, lines)

	return tools.StringResultWithSummary(result, summary), nil
}

func (t *GrepTool) grep(ctx context.Context, root, include string, re *regexp.Regexp) ([]Match, error) {
	var matches []Match

	info, err := os.Stat(root)
	if err != nil {
		return nil, err
	}

	if !info.IsDir() {
		// Single file
		return t.searchFile(root, re)
	}

	// Walk directory
	err = filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil // Skip errors
		}
		if isIgnoredGitName(d.Name()) {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		if d.IsDir() {
			return nil
		}

		// Check include pattern
		if include != "" {
			matched, _ := filepath.Match(include, filepath.Base(path))
			if !matched {
				return nil
			}
		}

		fileMatches, err := t.searchFile(path, re)
		if err != nil {
			return nil // Skip file errors
		}

		matches = append(matches, fileMatches...)
		return nil
	})

	if err != nil {
		return nil, err
	}

	return matches, nil
}

func (t *GrepTool) searchFile(path string, re *regexp.Regexp) ([]Match, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var matches []Match
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)
	lineNum := 0

	for scanner.Scan() {
		lineNum++
		line := scanner.Text()

		if loc := re.FindStringIndex(line); loc != nil {
			matches = append(matches, Match{
				File:   path,
				Line:   lineNum,
				Column: loc[0] + 1,
				Text:   line,
			})
		}
	}

	return matches, scanner.Err()
}
