package fs

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"
)

// Tool wraps fs read/write/patch operations.
type Tool struct {
	root string
}

func NewTool(root string) *Tool {
	return &Tool{root: root}
}

func (t *Tool) Read(path string) (string, error) {
	absPath, err := t.resolvePath(path)
	if err != nil {
		return "", err
	}
	data, err := os.ReadFile(absPath)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// Glob returns workspace-relative path matches for the provided pattern.
func (t *Tool) Glob(pathArg, pattern string, maxMatches int) ([]string, error) {
	if maxMatches <= 0 {
		maxMatches = 50
	}
	base := pathArg
	if strings.TrimSpace(base) == "" {
		base = "."
	}
	absBase, err := t.resolvePath(base)
	if err != nil {
		return nil, err
	}
	info, err := os.Stat(absBase)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("glob path %q is not a directory", pathArg)
	}

	pat := strings.TrimSpace(pattern)
	if pat == "" {
		pat = "*"
	}
	pat = filepath.ToSlash(pat)

	matches := make([]string, 0, maxMatches)
	err = filepath.WalkDir(absBase, func(curr string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if d.IsDir() {
			name := d.Name()
			if name == ".git" || name == ".cache" {
				return filepath.SkipDir
			}
			return nil
		}
		if len(matches) >= maxMatches {
			return nil
		}

		rel, relErr := filepath.Rel(t.root, curr)
		if relErr != nil {
			rel = curr
		}
		rel = filepath.ToSlash(rel)
		if !globMatch(pat, rel) {
			return nil
		}
		matches = append(matches, rel)
		return nil
	})
	return matches, err
}

// Grep returns "<path>:<line>: <text>" matches.
func (t *Tool) Grep(path, pattern string, maxMatches int) ([]string, error) {
	if strings.TrimSpace(pattern) == "" {
		return nil, errors.New("pattern is required")
	}
	if maxMatches <= 0 {
		maxMatches = 50
	}

	re, err := regexp.Compile(pattern)
	if err != nil {
		return nil, fmt.Errorf("invalid regexp pattern: %w", err)
	}

	target := path
	if strings.TrimSpace(target) == "" {
		target = "."
	}

	absTarget, err := t.resolvePath(target)
	if err != nil {
		return nil, err
	}

	info, err := os.Stat(absTarget)
	if err != nil {
		return nil, err
	}

	out := make([]string, 0, maxMatches)
	addMatches := func(filePath string) error {
		file, err := os.Open(filePath)
		if err != nil {
			return err
		}
		defer file.Close()

		relPath, relErr := filepath.Rel(t.root, filePath)
		if relErr != nil {
			relPath = filePath
		}

		reader := bufio.NewReader(file)
		ln := 0
		for {
			line, readErr := readLongLine(reader)
			if readErr == io.EOF {
				break
			}
			if readErr != nil {
				return readErr
			}
			ln++
			if re.MatchString(line) {
				out = append(out, fmt.Sprintf("%s:%d: %s", relPath, ln, line))
				if len(out) >= maxMatches {
					return nil
				}
			}
		}
		return nil
	}

	if info.IsDir() {
		err = filepath.WalkDir(absTarget, func(path string, d fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			if d.IsDir() {
				name := d.Name()
				if name == ".git" || name == ".cache" {
					return filepath.SkipDir
				}
				return nil
			}
			if len(out) >= maxMatches {
				return nil
			}
			return addMatches(path)
		})
		return out, err
	}

	if err := addMatches(absTarget); err != nil {
		return nil, err
	}
	return out, nil
}

func globMatch(pattern, relPath string) bool {
	pattern = strings.TrimSpace(pattern)
	if pattern == "" {
		return false
	}

	// filepath.Match does not support "**". Use a regex fallback for recursive patterns.
	if strings.Contains(pattern, "**") {
		re := globRegex(pattern)
		return re.MatchString(relPath)
	}
	ok, err := path.Match(pattern, relPath)
	if err != nil {
		return false
	}
	return ok
}

func globRegex(pattern string) *regexp.Regexp {
	var b strings.Builder
	b.WriteString("^")
	for i := 0; i < len(pattern); i++ {
		ch := pattern[i]
		switch ch {
		case '*':
			if i+1 < len(pattern) && pattern[i+1] == '*' {
				i++
				if i+1 < len(pattern) && pattern[i+1] == '/' {
					b.WriteString("(?:.*/)?")
					i++
				} else {
					b.WriteString(".*")
				}
			} else {
				b.WriteString("[^/]*")
			}
		case '?':
			b.WriteString("[^/]")
		case '.', '+', '(', ')', '[', ']', '{', '}', '^', '$', '|', '\\':
			b.WriteByte('\\')
			b.WriteByte(ch)
		default:
			b.WriteByte(ch)
		}
	}
	b.WriteString("$")
	re, err := regexp.Compile(b.String())
	if err != nil {
		// Fallback to a never-match regex.
		return regexp.MustCompile("^$")
	}
	return re
}

func readLongLine(reader *bufio.Reader) (string, error) {
	part, isPrefix, err := reader.ReadLine()
	if err != nil {
		return "", err
	}

	if !isPrefix {
		return string(part), nil
	}

	var b strings.Builder
	b.Write(part)
	for isPrefix {
		part, isPrefix, err = reader.ReadLine()
		if err != nil {
			return "", err
		}
		b.Write(part)
	}
	return b.String(), nil
}

// Edit performs a single replacement and persists the file.
func (t *Tool) Edit(path, oldText, newText string) (string, error) {
	if strings.TrimSpace(oldText) == "" {
		return "", errors.New("old_text is required")
	}
	absPath, err := t.resolvePath(path)
	if err != nil {
		return "", err
	}

	data, err := os.ReadFile(absPath)
	if err != nil {
		return "", err
	}
	before := string(data)
	after := strings.Replace(before, oldText, newText, 1)
	if before == after {
		return "", errors.New("target text not found")
	}

	if err := os.WriteFile(absPath, []byte(after), 0o644); err != nil {
		return "", err
	}

	return fmt.Sprintf("- %s\n+ %s", oldText, newText), nil
}

func (t *Tool) Write(path, content string) (int, error) {
	absPath, err := t.resolvePath(path)
	if err != nil {
		return 0, err
	}
	if err := os.MkdirAll(filepath.Dir(absPath), 0o755); err != nil {
		return 0, err
	}
	if err := os.WriteFile(absPath, []byte(content), 0o644); err != nil {
		return 0, err
	}
	return len(content), nil
}

func (t *Tool) resolvePath(path string) (string, error) {
	if strings.TrimSpace(path) == "" {
		return "", errors.New("path is required")
	}

	clean := filepath.Clean(path)
	abs := clean
	if !filepath.IsAbs(abs) {
		abs = filepath.Join(t.root, clean)
	}
	abs, err := filepath.Abs(abs)
	if err != nil {
		return "", err
	}
	rootAbs, err := filepath.Abs(t.root)
	if err != nil {
		return "", err
	}

	if abs != rootAbs && !strings.HasPrefix(abs, rootAbs+string(os.PathSeparator)) {
		return "", fmt.Errorf("path %q escapes workspace", path)
	}
	return abs, nil
}
