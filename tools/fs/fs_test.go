package fs

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestToolReadAndEdit(t *testing.T) {
	root := t.TempDir()
	tool := NewTool(root)

	path := filepath.Join(root, "a.txt")
	if err := os.WriteFile(path, []byte("hello\nworld\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	content, err := tool.Read("a.txt")
	if err != nil {
		t.Fatalf("Read failed: %v", err)
	}
	if !strings.Contains(content, "world") {
		t.Fatalf("unexpected read content: %q", content)
	}

	diff, err := tool.Edit("a.txt", "world", "ms-cli")
	if err != nil {
		t.Fatalf("Edit failed: %v", err)
	}
	if !strings.Contains(diff, "+ ms-cli") {
		t.Fatalf("unexpected diff: %q", diff)
	}
}

func TestToolRejectPathEscape(t *testing.T) {
	root := t.TempDir()
	tool := NewTool(root)
	if _, err := tool.Read("../outside.txt"); err == nil {
		t.Fatal("expected path escape error")
	}
}

func TestToolGrep_LongLine(t *testing.T) {
	root := t.TempDir()
	tool := NewTool(root)

	longLine := strings.Repeat("a", 200_000) + "NEEDLE"
	path := filepath.Join(root, "long.txt")
	if err := os.WriteFile(path, []byte(longLine+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	matches, err := tool.Grep("long.txt", "NEEDLE", 10)
	if err != nil {
		t.Fatalf("Grep failed on long line: %v", err)
	}
	if len(matches) != 1 {
		t.Fatalf("expected 1 match, got %d", len(matches))
	}
}

func TestToolGlobRecursivePattern(t *testing.T) {
	root := t.TempDir()
	tool := NewTool(root)

	files := []string{
		filepath.Join(root, "configs", "a.yaml"),
		filepath.Join(root, "configs", "nested", "b.yaml"),
		filepath.Join(root, "configs", "nested", "c.txt"),
	}
	for _, p := range files {
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	matches, err := tool.Glob(".", "configs/**/*.yaml", 10)
	if err != nil {
		t.Fatalf("Glob failed: %v", err)
	}
	if len(matches) != 2 {
		t.Fatalf("expected 2 yaml matches, got %d (%v)", len(matches), matches)
	}
}
