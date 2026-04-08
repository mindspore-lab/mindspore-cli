package fs

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestEditToolExecute_AddsDiffMetaAndKeepsTextResult(t *testing.T) {
	tmp := t.TempDir()
	path := filepath.Join(tmp, "sample.txt")
	original := strings.Join([]string{
		"line-1",
		"line-2",
		"line-3",
		"line-4",
		"line-5",
	}, "\n")
	if err := os.WriteFile(path, []byte(original), 0o644); err != nil {
		t.Fatalf("write seed file: %v", err)
	}

	tool := NewEditTool(tmp)
	params := []byte(`{"path":"sample.txt","old_string":"line-3","new_string":"line-3-updated"}`)
	result, err := tool.Execute(context.Background(), params)
	if err != nil {
		t.Fatalf("execute edit: %v", err)
	}
	if result.Error != nil {
		t.Fatalf("unexpected tool result error: %v", result.Error)
	}

	if !strings.Contains(result.Content, "Edited: sample.txt") {
		t.Fatalf("expected textual edit summary, got:\n%s", result.Content)
	}
	if !strings.Contains(result.Content, "- line-3") || !strings.Contains(result.Content, "+ line-3-updated") {
		t.Fatalf("expected textual +/- section preserved, got:\n%s", result.Content)
	}

	diff, ok := result.Meta["edit_diff"].(map[string]any)
	if !ok {
		t.Fatalf("expected edit_diff metadata, got: %#v", result.Meta)
	}
	if got, _ := diff["path"].(string); got != "sample.txt" {
		t.Fatalf("diff path = %q, want sample.txt", got)
	}
	lines, ok := diff["lines"].([]string)
	if !ok {
		t.Fatalf("expected lines []string in diff meta, got %#v", diff["lines"])
	}

	for _, want := range []string{
		" line-1",
		" line-2",
		"-line-3",
		"+line-3-updated",
		" line-4",
		" line-5",
	} {
		found := false
		for _, line := range lines {
			if line == want {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("expected diff lines to contain %q, got %#v", want, lines)
		}
	}
}
