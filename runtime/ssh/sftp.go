package ssh

import (
	"context"
	"fmt"
	"io"
	"path/filepath"
	"strings"
)

// ReadFile reads a remote file via SFTP with optional offset and limit (lines).
// Falls back to shell command if SFTP fails.
func (s *Session) ReadFile(path string, offset, limit int) (string, error) {
	// Try SFTP first
	content, err := s.readFileSFTP(path, offset, limit)
	if err == nil {
		return content, nil
	}

	// Fallback to shell command
	return s.ReadFileViaShell(path, offset, limit)
}

// readFileSFTP reads file via SFTP protocol.
func (s *Session) readFileSFTP(path string, offset, limit int) (string, error) {
	if err := s.ensureSFTP(); err != nil {
		return "", err
	}

	// Resolve relative path
	if !filepath.IsAbs(path) {
		path = filepath.Join(s.GetWorkDir(), path)
	}

	file, err := s.sftp.Open(path)
	if err != nil {
		return "", fmt.Errorf("failed to open remote file: %w", err)
	}
	defer file.Close()

	// Read content
	content, err := io.ReadAll(file)
	if err != nil {
		return "", fmt.Errorf("failed to read remote file: %w", err)
	}

	// Apply line offset and limit
	return applyLineLimits(string(content), offset, limit), nil
}

// WriteFile writes content to a remote file via SFTP.
// Falls back to shell command if SFTP fails.
func (s *Session) WriteFile(path string, content string) error {
	// Try SFTP first
	if err := s.writeFileSFTP(path, content); err == nil {
		return nil
	}

	// Fallback to shell command
	return s.WriteFileViaShell(path, content)
}

// writeFileSFTP writes file via SFTP protocol.
func (s *Session) writeFileSFTP(path string, content string) error {
	if err := s.ensureSFTP(); err != nil {
		return err
	}

	// Resolve relative path
	if !filepath.IsAbs(path) {
		path = filepath.Join(s.GetWorkDir(), path)
	}

	// Ensure parent directory exists
	dir := filepath.Dir(path)
	if err := s.sftp.MkdirAll(dir); err != nil {
		return fmt.Errorf("failed to create remote directory: %w", err)
	}

	file, err := s.sftp.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create remote file: %w", err)
	}
	defer file.Close()

	if _, err := file.Write([]byte(content)); err != nil {
		return fmt.Errorf("failed to write remote file: %w", err)
	}

	return nil
}

// Glob finds files matching a pattern on the remote host.
func (s *Session) Glob(pattern string) ([]string, error) {
	// Resolve relative path
	if !filepath.IsAbs(pattern) && s.GetWorkDir() != "" {
		pattern = filepath.Join(s.GetWorkDir(), pattern)
	}

	// Check if pattern contains ** (recursive)
	if strings.Contains(pattern, "**") {
		return s.globRecursive(pattern)
	}

	// Simple glob - use SFTP
	if err := s.ensureSFTP(); err != nil {
		return nil, err
	}

	matches, err := s.sftp.Glob(pattern)
	if err != nil {
		return nil, fmt.Errorf("glob failed: %w", err)
	}

	return matches, nil
}

// Grep searches for a pattern in files on the remote host.
func (s *Session) Grep(pattern string, paths []string, caseSensitive bool) ([]string, error) {
	var results []string

	// Build grep command
	flags := "-n" // line numbers
	if !caseSensitive {
		flags += " -i"
	}

	// Escape pattern for shell
	escapedPattern := strings.ReplaceAll(pattern, "'", "'\"'\"'")

	for _, path := range paths {
		// Resolve relative path
		if !filepath.IsAbs(path) && s.GetWorkDir() != "" {
			path = filepath.Join(s.GetWorkDir(), path)
		}

		cmd := fmt.Sprintf("grep %s '%s' '%s' 2>/dev/null || true", flags, escapedPattern, path)
		result, err := s.Run(context.Background(), cmd)
		if err != nil && result.ExitCode != 0 {
			continue
		}

		lines := strings.Split(result.Stdout, "\n")
		for _, line := range lines {
			line = strings.TrimSpace(line)
			if line != "" {
				results = append(results, fmt.Sprintf("%s:%s", path, line))
			}
		}
	}

	return results, nil
}

// ReadFileViaShell reads a file using shell command (fallback method).
func (s *Session) ReadFileViaShell(path string, offset, limit int) (string, error) {
	// Resolve relative path
	if !filepath.IsAbs(path) && s.GetWorkDir() != "" {
		path = filepath.Join(s.GetWorkDir(), path)
	}

	// Escape path for shell
	escapedPath := strings.ReplaceAll(path, "'", "'\"'\"'")

	var cmd string
	if offset == 0 && limit == 0 {
		cmd = fmt.Sprintf("cat '%s' 2>/dev/null || echo '__FILE_NOT_FOUND__'", escapedPath)
	} else {
		// Use tail and head for line range
		cmd = fmt.Sprintf("tail -n +%d '%s' 2>/dev/null | head -n %d", offset+1, escapedPath, limit)
	}

	result, err := s.Run(context.Background(), cmd)
	if err != nil && result.ExitCode != 0 {
		return "", fmt.Errorf("failed to read file: %w", err)
	}

	content := result.Stdout
	if strings.Contains(content, "__FILE_NOT_FOUND__") {
		return "", fmt.Errorf("file not found: %s", path)
	}

	return content, nil
}

// WriteFileViaShell writes a file using shell command (fallback method).
func (s *Session) WriteFileViaShell(path string, content string) error {
	// Resolve relative path
	if !filepath.IsAbs(path) && s.GetWorkDir() != "" {
		path = filepath.Join(s.GetWorkDir(), path)
	}

	// Escape path for shell
	escapedPath := strings.ReplaceAll(path, "'", "'\"'\"'")

	// Create parent directory
	dir := filepath.Dir(path)
	mkdirCmd := fmt.Sprintf("mkdir -p '%s'", strings.ReplaceAll(dir, "'", "'\"'\"'"))
	if _, err := s.Run(context.Background(), mkdirCmd); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	// Write content using tee
	// We need to escape the content for the shell
	escapedContent := strings.ReplaceAll(content, "'", "'\"'\"'")
	cmd := fmt.Sprintf("echo '%s' | tee '%s' > /dev/null", escapedContent, escapedPath)

	result, err := s.Run(context.Background(), cmd)
	if err != nil && result.ExitCode != 0 {
		return fmt.Errorf("failed to write file: %w", err)
	}

	return nil
}

// EditFile performs a text replacement in a remote file.
func (s *Session) EditFile(path string, oldString, newString string) error {
	// Read current content
	content, err := s.ReadFile(path, 0, 0)
	if err != nil {
		return err
	}

	// Check if oldString exists
	if !strings.Contains(content, oldString) {
		return fmt.Errorf("old_string not found in file")
	}

	// Check for multiple occurrences
	if strings.Count(content, oldString) > 1 {
		return fmt.Errorf("old_string appears multiple times in file, cannot uniquely identify")
	}

	// Replace
	newContent := strings.Replace(content, oldString, newString, 1)

	// Write back
	return s.WriteFile(path, newContent)
}

// globRecursive performs recursive glob matching.
func (s *Session) globRecursive(pattern string) ([]string, error) {
	// Convert ** pattern to find command
	// pattern like /path/**/file*.go -> find /path -name 'file*.go'
	baseDir, filePattern := splitPattern(pattern)

	cmd := fmt.Sprintf("find '%s' -type f -name '%s' 2>/dev/null",
		strings.ReplaceAll(baseDir, "'", "'\"'\"'"),
		strings.ReplaceAll(filePattern, "'", "'\"'\"'"))

	result, err := s.Run(context.Background(), cmd)
	if err != nil && result.ExitCode != 0 {
		return nil, fmt.Errorf("find command failed: %w", err)
	}

	lines := strings.Split(strings.TrimSpace(result.Stdout), "\n")
	var matches []string
	for _, line := range lines {
		if line != "" {
			matches = append(matches, line)
		}
	}

	return matches, nil
}

// splitPattern splits a ** glob pattern into base dir and file pattern.
func splitPattern(pattern string) (baseDir, filePattern string) {
	parts := strings.Split(pattern, "**")
	if len(parts) == 1 {
		// No ** in pattern
		dir := filepath.Dir(pattern)
		file := filepath.Base(pattern)
		return dir, file
	}

	baseDir = strings.TrimSuffix(parts[0], "/")
	if len(parts) > 1 {
		filePattern = strings.TrimPrefix(parts[1], "/")
	}
	if filePattern == "" {
		filePattern = "*"
	}

	return baseDir, filePattern
}

// applyLineLimits applies offset and limit to content lines.
func applyLineLimits(content string, offset, limit int) string {
	if offset == 0 && limit == 0 {
		return content
	}

	lines := strings.Split(content, "\n")

	// Apply offset
	if offset >= len(lines) {
		return ""
	}
	lines = lines[offset:]

	// Apply limit
	if limit > 0 && limit < len(lines) {
		lines = lines[:limit]
	}

	return strings.Join(lines, "\n")
}
