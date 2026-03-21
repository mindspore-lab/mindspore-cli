package fs

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGrepSkipsTrajectoryDuringDirectoryWalk(t *testing.T) {
	workDir := t.TempDir()
	writeTestFile(t, workDir, "notes.txt", "skill in project\n")
	writeTestFile(t, workDir, ".cache/session.trajectory.jsonl", `{"message":"skill in trace"}`+"\n")

	tool := NewGrepTool(workDir)
	result := executeGrep(t, tool, grepParams{Pattern: "skill", Path: "."})

	if strings.Contains(result.Content, ".cache/session.trajectory.jsonl") {
		t.Fatalf("directory walk should skip trajectory files, got %q", result.Content)
	}
	if !strings.Contains(result.Content, "notes.txt:1:skill in project") {
		t.Fatalf("expected normal file match, got %q", result.Content)
	}
}

func TestGrepAllowsExplicitTrajectoryFilePath(t *testing.T) {
	workDir := t.TempDir()
	writeTestFile(t, workDir, ".cache/session.trajectory.jsonl", `{"message":"skill in trace"}`+"\n")

	tool := NewGrepTool(workDir)
	result := executeGrep(t, tool, grepParams{
		Pattern: "skill",
		Path:    ".cache/session.trajectory.jsonl",
	})

	if !strings.Contains(filepath.ToSlash(result.Content), `.cache/session.trajectory.jsonl:1:{"message":"skill in trace"}`) {
		t.Fatalf("explicit trajectory file path should be searchable, got %q", result.Content)
	}
}

func TestGrepHeadLimitTruncatesMatches(t *testing.T) {
	workDir := t.TempDir()
	writeTestFile(t, workDir, "many.txt", "match 1\nmatch 2\nmatch 3\n")

	tool := NewGrepTool(workDir)
	result := executeGrep(t, tool, grepParams{
		Pattern:   "match",
		Path:      ".",
		HeadLimit: 2,
	})

	if strings.Contains(result.Content, "many.txt:3:match 3") {
		t.Fatalf("expected head limit to stop after 2 matches, got %q", result.Content)
	}
	if !strings.Contains(result.Content, "[grep truncated after 2 matches]") {
		t.Fatalf("expected truncation marker, got %q", result.Content)
	}
	if result.Summary != "2 matches (truncated at 2)" {
		t.Fatalf("summary = %q, want %q", result.Summary, "2 matches (truncated at 2)")
	}
}

func TestGrepTruncatesLongMatchedLine(t *testing.T) {
	workDir := t.TempDir()
	longTail := strings.Repeat("x", maxMatchLineRunes+50)
	writeTestFile(t, workDir, "long.txt", "match "+longTail+"\n")

	tool := NewGrepTool(workDir)
	result := executeGrep(t, tool, grepParams{Pattern: "match", Path: "."})

	if !strings.Contains(result.Content, matchLineTruncation) {
		t.Fatalf("expected long line truncation marker, got %q", result.Content)
	}
	if strings.Contains(result.Content, longTail) {
		t.Fatalf("expected long line content to be truncated, got %q", result.Content)
	}
}

func TestGrepNoMatchesFound(t *testing.T) {
	workDir := t.TempDir()
	writeTestFile(t, workDir, "notes.txt", "no relevant content\n")

	tool := NewGrepTool(workDir)
	result := executeGrep(t, tool, grepParams{Pattern: "skill", Path: "."})

	if result.Content != "No matches found" {
		t.Fatalf("content = %q, want %q", result.Content, "No matches found")
	}
	if result.Summary != "0 matches" {
		t.Fatalf("summary = %q, want %q", result.Summary, "0 matches")
	}
}

func executeGrep(t *testing.T, tool *GrepTool, params grepParams) *struct {
	Content string
	Summary string
} {
	t.Helper()

	raw, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("Marshal params failed: %v", err)
	}

	result, err := tool.Execute(context.Background(), raw)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
	if result.Error != nil {
		t.Fatalf("tool returned error: %v", result.Error)
	}

	return &struct {
		Content string
		Summary string
	}{
		Content: result.Content,
		Summary: result.Summary,
	}
}

func writeTestFile(t *testing.T, workDir, relPath, content string) {
	t.Helper()

	fullPath, err := resolveSafePath(workDir, relPath)
	if err != nil {
		t.Fatalf("resolveSafePath failed: %v", err)
	}
	dir := filepath.Dir(fullPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatalf("MkdirAll failed: %v", err)
	}
	if err := os.WriteFile(fullPath, []byte(content), 0644); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}
}
