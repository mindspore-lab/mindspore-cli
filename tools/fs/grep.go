package fs

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/vigo999/ms-cli/integrations/llm"
	"github.com/vigo999/ms-cli/tools"
)

// GrepTool searches for patterns in files.
type GrepTool struct {
	workDir string
}

const (
	defaultHeadLimit      = 20
	maxHeadLimit          = 100
	maxMatchLineRunes     = 300
	matchLineTruncation   = " [line truncated]"
	resultSummaryTemplate = "%d matches"
)

var errGrepLimitReached = errors.New("grep head limit reached")

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
			"head_limit": {
				Type:        "integer",
				Description: "Maximum number of matches to return (default: 20, max: 100)",
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
	HeadLimit     int    `json:"head_limit"`
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

	headLimit := normalizeHeadLimit(p.HeadLimit)

	// Find files and search
	matches, truncated, err := t.grep(ctx, fullPath, p.Include, re, headLimit)
	if err != nil {
		return tools.ErrorResult(err), nil
	}

	// Format results
	if len(matches) == 0 {
		return tools.StringResultWithSummary("No matches found", "0 matches"), nil
	}

	var lines []string
	for _, m := range matches {
		relPath, _ := filepath.Rel(t.workDir, m.File)
		lines = append(lines, fmt.Sprintf("%s:%d:%s", relPath, m.Line, truncateMatchLine(m.Text)))
	}

	result := strings.Join(lines, "\n")
	summary := fmt.Sprintf(resultSummaryTemplate, len(matches))
	if truncated {
		summary = fmt.Sprintf("%s (truncated at %d)", summary, headLimit)
		result += fmt.Sprintf("\n[grep truncated after %d matches]", headLimit)
	}

	return tools.StringResultWithSummary(result, summary), nil
}

func (t *GrepTool) grep(ctx context.Context, root, include string, re *regexp.Regexp, headLimit int) ([]Match, bool, error) {
	var matches []Match

	info, err := os.Stat(root)
	if err != nil {
		return nil, false, err
	}

	if !info.IsDir() {
		// Single file
		fileMatches, limitReached, err := t.searchFile(root, re, headLimit)
		if err != nil {
			return nil, false, err
		}
		return fileMatches, limitReached, nil
	}

	// Walk directory
	err = filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if err != nil {
			return nil // Skip errors
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

		if shouldSkipTrajectory(path, d) {
			return nil
		}

		remaining := headLimit - len(matches)
		if remaining <= 0 {
			return errGrepLimitReached
		}

		fileMatches, limitReached, err := t.searchFile(path, re, remaining)
		if err != nil {
			return nil // Skip file errors
		}

		matches = append(matches, fileMatches...)
		if limitReached || len(matches) >= headLimit {
			return errGrepLimitReached
		}
		return nil
	})

	if err != nil && !errors.Is(err, errGrepLimitReached) {
		return nil, false, err
	}

	return matches, errors.Is(err, errGrepLimitReached), nil
}

func (t *GrepTool) searchFile(path string, re *regexp.Regexp, headLimit int) ([]Match, bool, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, false, err
	}
	defer file.Close()

	var matches []Match
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)
	lineNum := 0
	limitReached := false

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
			if len(matches) >= headLimit {
				limitReached = true
				break
			}
		}
	}

	return matches, limitReached, scanner.Err()
}

func normalizeHeadLimit(limit int) int {
	switch {
	case limit <= 0:
		return defaultHeadLimit
	case limit > maxHeadLimit:
		return maxHeadLimit
	default:
		return limit
	}
}

func truncateMatchLine(line string) string {
	runes := []rune(line)
	if len(runes) <= maxMatchLineRunes {
		return line
	}
	return string(runes[:maxMatchLineRunes]) + matchLineTruncation
}

func shouldSkipTrajectory(path string, d os.DirEntry) bool {
	if d.IsDir() {
		return false
	}
	if !strings.HasSuffix(d.Name(), ".trajectory.jsonl") {
		return false
	}
	return filepath.Base(filepath.Dir(path)) == ".cache"
}
