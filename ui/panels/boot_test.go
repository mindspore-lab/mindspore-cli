package panels

import (
	"regexp"
	"strings"
	"testing"
)

var ansiPattern = regexp.MustCompile(`\x1b\[[0-9;]*m`)

func TestRenderBootScreenReturnsEmptyBeforeWindowSizeIsKnown(t *testing.T) {
	if got := RenderBootScreen(0, 0, 0); got != "" {
		t.Fatalf("RenderBootScreen() = %q, want empty string", got)
	}
}

func TestRenderBootScreenHasSingleBorderlessTitle(t *testing.T) {
	got := RenderBootScreen(80, 24, 3)
	if strings.Contains(got, "╭") || strings.Contains(got, "│") || strings.Contains(got, "╯") {
		t.Fatalf("expected borderless boot screen, got:\n%s", got)
	}

	plain := ansiPattern.ReplaceAllString(got, "")
	if count := strings.Count(plain, "MindSpore"); count != 1 {
		t.Fatalf("expected one boot title, got %d in:\n%s", count, plain)
	}
}
