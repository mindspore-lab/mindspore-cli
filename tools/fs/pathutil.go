package fs

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

var allowedAbsoluteHomePaths = []string{
	"~/.mscli/skills",
	"~/.mscli/mindspore-skills",
}

func resolveSafePath(workDir, input string) (string, error) {
	if strings.TrimSpace(input) == "" {
		return "", fmt.Errorf("path is required")
	}

	baseAbs, err := filepath.Abs(workDir)
	if err != nil {
		return "", fmt.Errorf("resolve working directory: %w", err)
	}

	cleaned := filepath.Clean(input)
	normalized, err := normalizeAllowedAbsolutePath(cleaned)
	if err != nil {
		return "", err
	}
	cleaned = normalized

	if filepath.IsAbs(cleaned) {
		absPath, err := filepath.Abs(cleaned)
		if err != nil {
			return "", fmt.Errorf("resolve path: %w", err)
		}
		if pathWithinBase(baseAbs, absPath) {
			if isIgnoredGitPath(absPath) {
				return "", fmt.Errorf("path is ignored: %s", input)
			}
			return absPath, nil
		}

		allowed, err := isAllowedAbsolutePath(absPath)
		if err != nil {
			return "", err
		}
		if !allowed {
			return "", fmt.Errorf("absolute paths are not allowed: %s", input)
		}
		if isIgnoredGitPath(absPath) {
			return "", fmt.Errorf("path is ignored: %s", input)
		}
		return absPath, nil
	}

	fullAbs, err := filepath.Abs(filepath.Join(baseAbs, cleaned))
	if err != nil {
		return "", fmt.Errorf("resolve path: %w", err)
	}

	rel, err := filepath.Rel(baseAbs, fullAbs)
	if err != nil {
		return "", fmt.Errorf("check path: %w", err)
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
		return "", fmt.Errorf("path escapes working directory: %s", input)
	}
	if isIgnoredGitPath(rel) {
		return "", fmt.Errorf("path is ignored: %s", input)
	}

	return fullAbs, nil
}

func normalizeAllowedAbsolutePath(input string) (string, error) {
	cleanedSlash := filepath.ToSlash(filepath.Clean(input))
	if !matchesAllowedHomePath(cleanedSlash) {
		return input, nil
	}

	homeDir, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("resolve home directory: %w", err)
	}

	trimmed := strings.TrimPrefix(cleanedSlash, "~/")
	return filepath.Join(homeDir, filepath.FromSlash(trimmed)), nil
}

func isAllowedAbsolutePath(input string) (bool, error) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return false, fmt.Errorf("resolve home directory: %w", err)
	}

	for _, allowedRoot := range allowedAbsoluteHomePaths {
		trimmed := strings.TrimPrefix(allowedRoot, "~/")
		base := filepath.Join(homeDir, filepath.FromSlash(trimmed))
		if pathWithinBase(base, input) {
			return true, nil
		}
	}

	return false, nil
}

func matchesAllowedHomePath(input string) bool {
	for _, allowedRoot := range allowedAbsoluteHomePaths {
		if input == allowedRoot || strings.HasPrefix(input, allowedRoot+"/") {
			return true
		}
	}
	return false
}

func pathWithinBase(base, target string) bool {
	rel, err := filepath.Rel(base, target)
	if err != nil {
		return false
	}
	if rel == "." {
		return true
	}
	return rel != ".." && !strings.HasPrefix(rel, ".."+string(os.PathSeparator))
}

func isIgnoredGitPath(path string) bool {
	cleaned := filepath.ToSlash(filepath.Clean(path))
	for _, part := range strings.Split(cleaned, "/") {
		if part == ".git" {
			return true
		}
	}
	return false
}

func isIgnoredGitName(name string) bool {
	return name == ".git"
}
