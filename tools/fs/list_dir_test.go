package fs

import (
	"context"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
)

func TestListDirToolListsFilesAndDirectories(t *testing.T) {
	workDir := t.TempDir()
	mustWriteTestFile(t, filepath.Join(workDir, "README.md"), "# readme\n")
	mustWriteTestFile(t, filepath.Join(workDir, "cmd", "main.go"), "package main\n")

	params, err := json.Marshal(map[string]any{
		"path":  ".",
		"depth": 2,
	})
	if err != nil {
		t.Fatalf("Marshal returned error: %v", err)
	}

	result, err := NewListDirTool(workDir).Execute(context.Background(), params)
	if err != nil {
		t.Fatalf("Execute returned error: %v", err)
	}
	if result.Error != nil {
		t.Fatalf("Execute result error = %v, want nil", result.Error)
	}
	if !strings.Contains(result.Content, "README.md") {
		t.Fatalf("list_dir content = %q, want README.md", result.Content)
	}
	if !strings.Contains(result.Content, "cmd/") {
		t.Fatalf("list_dir content = %q, want cmd/", result.Content)
	}
	if !strings.Contains(result.Content, "main.go") {
		t.Fatalf("list_dir content = %q, want nested main.go", result.Content)
	}
}

func TestListDirToolRespectsDepth(t *testing.T) {
	workDir := t.TempDir()
	mustWriteTestFile(t, filepath.Join(workDir, "pkg", "inner", "leaf.txt"), "x")

	params, err := json.Marshal(map[string]any{
		"path":  ".",
		"depth": 1,
	})
	if err != nil {
		t.Fatalf("Marshal returned error: %v", err)
	}

	result, err := NewListDirTool(workDir).Execute(context.Background(), params)
	if err != nil {
		t.Fatalf("Execute returned error: %v", err)
	}
	if result.Error != nil {
		t.Fatalf("Execute result error = %v, want nil", result.Error)
	}
	if !strings.Contains(result.Content, "pkg/") {
		t.Fatalf("list_dir content = %q, want pkg/", result.Content)
	}
	if strings.Contains(result.Content, "leaf.txt") {
		t.Fatalf("list_dir content = %q, should not include depth-2 file", result.Content)
	}
}

func TestListDirToolHidesDotEntriesByDefault(t *testing.T) {
	workDir := t.TempDir()
	mustWriteTestFile(t, filepath.Join(workDir, ".env"), "token=x\n")
	mustWriteTestFile(t, filepath.Join(workDir, "visible.txt"), "x\n")

	params, err := json.Marshal(map[string]any{
		"path": ".",
	})
	if err != nil {
		t.Fatalf("Marshal returned error: %v", err)
	}

	result, err := NewListDirTool(workDir).Execute(context.Background(), params)
	if err != nil {
		t.Fatalf("Execute returned error: %v", err)
	}
	if result.Error != nil {
		t.Fatalf("Execute result error = %v, want nil", result.Error)
	}
	if strings.Contains(result.Content, ".env") {
		t.Fatalf("list_dir content = %q, should hide dotfiles by default", result.Content)
	}
	if !strings.Contains(result.Content, "visible.txt") {
		t.Fatalf("list_dir content = %q, want visible.txt", result.Content)
	}
}
