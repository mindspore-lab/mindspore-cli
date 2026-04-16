package fs

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGlobToolIgnoresGitPaths(t *testing.T) {
	workDir := t.TempDir()
	mustWriteTestFile(t, filepath.Join(workDir, "visible.txt"), "visible")
	mustWriteTestFile(t, filepath.Join(workDir, "nested", "found.go"), "package nested")
	mustWriteTestFile(t, filepath.Join(workDir, ".git", "config"), "hidden")

	params, err := json.Marshal(map[string]any{
		"pattern": "**",
		"path":    ".",
	})
	if err != nil {
		t.Fatalf("Marshal returned error: %v", err)
	}

	result, err := NewGlobTool(workDir).Execute(context.Background(), params)
	if err != nil {
		t.Fatalf("Execute returned error: %v", err)
	}
	if result.Error != nil {
		t.Fatalf("Execute result error = %v, want nil", result.Error)
	}
	if strings.Contains(result.Content, ".git") {
		t.Fatalf("glob content = %q, should not include .git paths", result.Content)
	}
	if !strings.Contains(result.Content, "visible.txt") {
		t.Fatalf("glob content = %q, want visible.txt", result.Content)
	}
	if !strings.Contains(result.Content, filepath.Join("nested", "found.go")) {
		t.Fatalf("glob content = %q, want nested/found.go", result.Content)
	}
}

func TestGrepToolIgnoresGitPaths(t *testing.T) {
	workDir := t.TempDir()
	mustWriteTestFile(t, filepath.Join(workDir, "visible.txt"), "needle\n")
	mustWriteTestFile(t, filepath.Join(workDir, ".git", "config"), "needle\n")

	params, err := json.Marshal(map[string]any{
		"pattern":        "needle",
		"path":           ".",
		"case_sensitive": true,
	})
	if err != nil {
		t.Fatalf("Marshal returned error: %v", err)
	}

	result, err := NewGrepTool(workDir).Execute(context.Background(), params)
	if err != nil {
		t.Fatalf("Execute returned error: %v", err)
	}
	if result.Error != nil {
		t.Fatalf("Execute result error = %v, want nil", result.Error)
	}
	if strings.Contains(result.Content, ".git") {
		t.Fatalf("grep content = %q, should not include .git paths", result.Content)
	}
	if !strings.Contains(result.Content, "visible.txt:1:needle") {
		t.Fatalf("grep content = %q, want visible.txt match", result.Content)
	}
}

func TestGlobToolSupportsOffsetAndLimit(t *testing.T) {
	workDir := t.TempDir()
	mustWriteTestFile(t, filepath.Join(workDir, "a.txt"), "a")
	mustWriteTestFile(t, filepath.Join(workDir, "b.txt"), "b")
	mustWriteTestFile(t, filepath.Join(workDir, "nested", "c.txt"), "c")

	params, err := json.Marshal(map[string]any{
		"pattern": "**",
		"path":    ".",
		"offset":  2,
		"limit":   1,
	})
	if err != nil {
		t.Fatalf("Marshal returned error: %v", err)
	}

	result, err := NewGlobTool(workDir).Execute(context.Background(), params)
	if err != nil {
		t.Fatalf("Execute returned error: %v", err)
	}
	if result.Error != nil {
		t.Fatalf("Execute result error = %v, want nil", result.Error)
	}
	if got, want := result.Content, "showing 2-2 of 3 files\nb.txt"; got != want {
		t.Fatalf("glob content = %q, want %q", got, want)
	}
	if got, want := result.Summary, "showing 2-2 of 3 files"; got != want {
		t.Fatalf("glob summary = %q, want %q", got, want)
	}
}

func TestGlobToolDoubleStarMatchesRootAndNestedFiles(t *testing.T) {
	workDir := t.TempDir()
	mustWriteTestFile(t, filepath.Join(workDir, "train.py"), "print('root')\n")
	mustWriteTestFile(t, filepath.Join(workDir, "nested", "train.py"), "print('nested')\n")

	params, err := json.Marshal(map[string]any{
		"pattern": "**/train.py",
		"path":    ".",
	})
	if err != nil {
		t.Fatalf("Marshal returned error: %v", err)
	}

	result, err := NewGlobTool(workDir).Execute(context.Background(), params)
	if err != nil {
		t.Fatalf("Execute returned error: %v", err)
	}
	if result.Error != nil {
		t.Fatalf("Execute result error = %v, want nil", result.Error)
	}
	if !strings.Contains(result.Content, "train.py") {
		t.Fatalf("glob content = %q, want root train.py match", result.Content)
	}
	if !strings.Contains(result.Content, filepath.Join("nested", "train.py")) {
		t.Fatalf("glob content = %q, want nested train.py match", result.Content)
	}
}

func TestGlobToolDefaultsLimitTo100(t *testing.T) {
	workDir := t.TempDir()
	for i := 0; i < 120; i++ {
		mustWriteTestFile(t, filepath.Join(workDir, fmt.Sprintf("file%03d.txt", i)), "x")
	}

	params, err := json.Marshal(map[string]any{
		"pattern": "*.txt",
		"path":    ".",
	})
	if err != nil {
		t.Fatalf("Marshal returned error: %v", err)
	}

	result, err := NewGlobTool(workDir).Execute(context.Background(), params)
	if err != nil {
		t.Fatalf("Execute returned error: %v", err)
	}
	if result.Error != nil {
		t.Fatalf("Execute result error = %v, want nil", result.Error)
	}
	if got, want := len(strings.Split(result.Content, "\n")), 101; got != want {
		t.Fatalf("glob returned %d files, want %d", got, want)
	}
	if got, want := result.Summary, "showing 1-100 of 120 files"; got != want {
		t.Fatalf("glob summary = %q, want %q", got, want)
	}
}

func TestGlobToolCapsLimitAt100(t *testing.T) {
	workDir := t.TempDir()
	for i := 0; i < 120; i++ {
		mustWriteTestFile(t, filepath.Join(workDir, fmt.Sprintf("file%03d.txt", i)), "x")
	}

	params, err := json.Marshal(map[string]any{
		"pattern": "*.txt",
		"path":    ".",
		"limit":   150,
	})
	if err != nil {
		t.Fatalf("Marshal returned error: %v", err)
	}

	result, err := NewGlobTool(workDir).Execute(context.Background(), params)
	if err != nil {
		t.Fatalf("Execute returned error: %v", err)
	}
	if result.Error != nil {
		t.Fatalf("Execute result error = %v, want nil", result.Error)
	}
	if got, want := len(strings.Split(result.Content, "\n")), 101; got != want {
		t.Fatalf("glob returned %d files, want %d", got, want)
	}
	if got, want := result.Summary, "showing 1-100 of 120 files"; got != want {
		t.Fatalf("glob summary = %q, want %q", got, want)
	}
}

func TestGrepToolSupportsOffsetAndLimit(t *testing.T) {
	workDir := t.TempDir()
	mustWriteTestFile(t, filepath.Join(workDir, "a.txt"), "needle one\nneedle two\n")
	mustWriteTestFile(t, filepath.Join(workDir, "b.txt"), "needle three\n")

	params, err := json.Marshal(map[string]any{
		"pattern":        "needle",
		"path":           ".",
		"case_sensitive": true,
		"offset":         2,
		"limit":          1,
	})
	if err != nil {
		t.Fatalf("Marshal returned error: %v", err)
	}

	result, err := NewGrepTool(workDir).Execute(context.Background(), params)
	if err != nil {
		t.Fatalf("Execute returned error: %v", err)
	}
	if result.Error != nil {
		t.Fatalf("Execute result error = %v, want nil", result.Error)
	}
	if got, want := result.Content, "showing 2-2 of 3 matches\na.txt:2:needle two"; got != want {
		t.Fatalf("grep content = %q, want %q", got, want)
	}
	if got, want := result.Summary, "showing 2-2 of 3 matches"; got != want {
		t.Fatalf("grep summary = %q, want %q", got, want)
	}
}

func TestGrepToolDefaultsLimitTo100(t *testing.T) {
	workDir := t.TempDir()
	var content strings.Builder
	for i := 0; i < 120; i++ {
		content.WriteString(fmt.Sprintf("needle %03d\n", i))
	}
	mustWriteTestFile(t, filepath.Join(workDir, "matches.txt"), content.String())

	params, err := json.Marshal(map[string]any{
		"pattern":        "needle",
		"path":           ".",
		"case_sensitive": true,
	})
	if err != nil {
		t.Fatalf("Marshal returned error: %v", err)
	}

	result, err := NewGrepTool(workDir).Execute(context.Background(), params)
	if err != nil {
		t.Fatalf("Execute returned error: %v", err)
	}
	if result.Error != nil {
		t.Fatalf("Execute result error = %v, want nil", result.Error)
	}
	if got, want := len(strings.Split(result.Content, "\n")), 101; got != want {
		t.Fatalf("grep returned %d matches, want %d", got, want)
	}
	if got, want := result.Summary, "showing 1-100 of 120 matches"; got != want {
		t.Fatalf("grep summary = %q, want %q", got, want)
	}
}

func TestGrepToolCapsLimitAt100(t *testing.T) {
	workDir := t.TempDir()
	var content strings.Builder
	for i := 0; i < 120; i++ {
		content.WriteString(fmt.Sprintf("needle %03d\n", i))
	}
	mustWriteTestFile(t, filepath.Join(workDir, "matches.txt"), content.String())

	params, err := json.Marshal(map[string]any{
		"pattern":        "needle",
		"path":           ".",
		"case_sensitive": true,
		"limit":          150,
	})
	if err != nil {
		t.Fatalf("Marshal returned error: %v", err)
	}

	result, err := NewGrepTool(workDir).Execute(context.Background(), params)
	if err != nil {
		t.Fatalf("Execute returned error: %v", err)
	}
	if result.Error != nil {
		t.Fatalf("Execute result error = %v, want nil", result.Error)
	}
	if got, want := len(strings.Split(result.Content, "\n")), 101; got != want {
		t.Fatalf("grep returned %d matches, want %d", got, want)
	}
	if got, want := result.Summary, "showing 1-100 of 120 matches"; got != want {
		t.Fatalf("grep summary = %q, want %q", got, want)
	}
}

func mustWriteTestFile(t *testing.T, path, content string) {
	t.Helper()

	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatalf("MkdirAll returned error: %v", err)
	}
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}
}
