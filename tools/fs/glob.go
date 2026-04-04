package fs

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/mindspore-lab/mindspore-cli/integrations/llm"
	"github.com/mindspore-lab/mindspore-cli/tools"
)

// GlobTool finds files matching a glob pattern.
type GlobTool struct {
	workDir string
}

// NewGlobTool creates a new glob tool.
func NewGlobTool(workDir string) *GlobTool {
	return &GlobTool{workDir: workDir}
}

// Name returns the tool name.
func (t *GlobTool) Name() string {
	return "glob"
}

// Description returns the tool description.
func (t *GlobTool) Description() string {
	return "Find files matching a glob pattern. Use this to explore project structure and find specific file types."
}

// Schema returns the tool parameter schema.
func (t *GlobTool) Schema() llm.ToolSchema {
	return llm.ToolSchema{
		Type: "object",
		Properties: map[string]llm.Property{
			"pattern": {
				Type:        "string",
				Description: "Glob pattern (e.g., '*.go', '**/*.yaml', 'cmd/*')",
			},
			"path": {
				Type:        "string",
				Description: "Base directory to search from (default: current directory)",
			},
			"offset": {
				Type:        "integer",
				Description: "File number to start returning from (1-indexed, 0 means from start)",
			},
			"limit": {
				Type:        "integer",
				Description: "Maximum number of files to return (defaults to 100, maximum 100)",
			},
		},
		Required: []string{"pattern"},
	}
}

type globParams struct {
	Pattern string `json:"pattern"`
	Path    string `json:"path"`
	Offset  int    `json:"offset"`
	Limit   int    `json:"limit"`
}

// Execute executes the glob tool.
func (t *GlobTool) Execute(ctx context.Context, params json.RawMessage) (*tools.Result, error) {
	var p globParams
	if err := tools.ParseParams(params, &p); err != nil {
		return tools.ErrorResult(err), nil
	}

	// Resolve base path
	basePath := "."
	if p.Path != "" {
		basePath = p.Path
	}
	fullBasePath, err := resolveSafePath(t.workDir, basePath)
	if err != nil {
		return tools.ErrorResult(err), nil
	}

	// Check if base path exists
	info, err := os.Stat(fullBasePath)
	if err != nil {
		if os.IsNotExist(err) {
			return tools.ErrorResultf("path not found: %s", p.Path), nil
		}
		return tools.ErrorResult(err), nil
	}

	pattern := filepath.ToSlash(strings.TrimSpace(p.Pattern))
	if pattern == "" {
		return tools.ErrorResultf("pattern is required"), nil
	}
	recursive := strings.Contains(pattern, "**") || strings.Contains(pattern, "/")

	// Find matches
	var matches []string
	if recursive {
		matches, err = t.globRecursive(fullBasePath, pattern)
	} else {
		matches, err = t.globSingle(fullBasePath, pattern)
	}
	if err != nil {
		return tools.ErrorResult(err), nil
	}

	// If base path is a file (not directory), check it directly
	if !info.IsDir() {
		matched, _ := filepath.Match(pattern, filepath.Base(fullBasePath))
		if matched {
			relPath, _ := filepath.Rel(t.workDir, fullBasePath)
			matches = append(matches, relPath)
		}
	}

	// Sort and deduplicate
	matches = uniqueStrings(matches)
	sort.Strings(matches)

	if len(matches) == 0 {
		return tools.StringResultWithSummary("No files found", "0 files"), nil
	}
	effectiveLimit := normalizeSearchResultLimit(p.Limit)
	totalMatches := len(matches)
	matches = sliceWithOffsetLimit(matches, p.Offset, effectiveLimit)

	summary := pagedSearchSummary(totalMatches, p.Offset, len(matches), "files")
	result := buildSearchResultContent(summary, matches)

	return tools.StringResultWithSummary(result, summary), nil
}

func (t *GlobTool) globSingle(root, pattern string) ([]string, error) {
	var matches []string

	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, err
	}

	for _, entry := range entries {
		name := entry.Name()
		if isIgnoredGitName(name) {
			continue
		}
		matched, _ := filepath.Match(pattern, name)
		if matched {
			relPath, _ := filepath.Rel(t.workDir, filepath.Join(root, name))
			matches = append(matches, relPath)
		}
	}

	return matches, nil
}

func (t *GlobTool) globRecursive(root, pattern string) ([]string, error) {
	var matches []string
	re, err := compileDoubleStarPattern(pattern)
	if err != nil {
		return nil, err
	}

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

		relFromRoot, relErr := filepath.Rel(root, path)
		if relErr != nil {
			return nil
		}
		matched := re.MatchString(filepath.ToSlash(relFromRoot))
		if matched {
			relPath, _ := filepath.Rel(t.workDir, path)
			matches = append(matches, relPath)
		}

		return nil
	})

	if err != nil {
		return nil, err
	}

	return matches, nil
}

func compileDoubleStarPattern(pattern string) (*regexp.Regexp, error) {
	var b strings.Builder
	b.WriteString("^")
	for i := 0; i < len(pattern); i++ {
		ch := pattern[i]
		switch ch {
		case '*':
			if i+1 < len(pattern) && pattern[i+1] == '*' {
				b.WriteString(".*")
				i++
				continue
			}
			b.WriteString(`[^/]*`)
		case '?':
			b.WriteString(`[^/]`)
		default:
			if strings.ContainsRune(".+()|[]{}^$\\", rune(ch)) {
				b.WriteByte('\\')
			}
			b.WriteByte(ch)
		}
	}
	b.WriteString("$")
	return regexp.Compile(b.String())
}

func uniqueStrings(s []string) []string {
	seen := make(map[string]bool)
	result := make([]string, 0, len(s))
	for _, v := range s {
		if !seen[v] {
			seen[v] = true
			result = append(result, v)
		}
	}
	return result
}
