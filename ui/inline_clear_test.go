package ui

import (
	"strings"
	"testing"
)

func TestClearHeadingLinesIncludesBannerAndResumeHint(t *testing.T) {
	app := New(nil, nil, "test-version", "/tmp/work", "github.com/example/repo", "demo-model", 4096)
	app.bootActive = false

	lines := app.clearHeadingLines("Resume the previous conversation with: mscli resume sess_123")
	if got, want := len(lines), 2; got != want {
		t.Fatalf("clear heading line count = %d, want %d", got, want)
	}
	if !strings.Contains(lines[0], "MindSpore CLI") {
		t.Fatalf("banner line missing logo heading: %q", lines[0])
	}
	if got, want := lines[1], metaStyle.Render("Resume the previous conversation with: mscli resume sess_123"); got != want {
		t.Fatalf("resume hint = %q, want %q", got, want)
	}
}
