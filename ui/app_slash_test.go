package ui

import (
	"strings"
	"testing"

	"github.com/vigo999/ms-cli/ui/model"
)

func newSlashTestApp() App {
	ch := make(chan model.Event)
	a := New(ch, nil, "v0.0.0", ".", "", "openrouter", "deepseek", 24000)
	a.width = 120
	a.height = 40
	a.resizeViewport()
	return a
}

func TestSlashSuggestions_DedupAndGroupedUsage(t *testing.T) {
	a := newSlashTestApp()
	a.input = a.input.SetValue("/")

	a.refreshSlashSuggestions()

	if len(a.slashCandidates) == 0 {
		t.Fatal("expected slash candidates")
	}

	modelCount := 0
	for _, c := range a.slashCandidates {
		if c.command == "/model" {
			modelCount++
			if !strings.Contains(c.display, "use <provider>/<model>") {
				t.Fatalf("expected /model grouped usage in display, got: %q", c.display)
			}
		}
	}
	if modelCount != 1 {
		t.Fatalf("expected exactly one /model candidate, got %d", modelCount)
	}
}

func TestSlashSuggestions_TabCompletesCommand(t *testing.T) {
	a := newSlashTestApp()
	a.input = a.input.SetValue("/mo")

	a.refreshSlashSuggestions()
	if len(a.slashCandidates) == 0 {
		t.Fatal("expected slash candidates")
	}

	a.applySelectedSlash()
	if got := a.input.Value(); got != "/model " {
		t.Fatalf("expected tab completion to /model, got %q", got)
	}
}

func TestSlashSuggestions_TabCompletesSubcommand(t *testing.T) {
	a := newSlashTestApp()
	a.input = a.input.SetValue("/model l")

	a.refreshSlashSuggestions()
	if len(a.slashCandidates) == 0 {
		t.Fatal("expected slash candidates")
	}

	a.applySelectedSlash()
	if got := a.input.Value(); got != "/model list" {
		t.Fatalf("expected tab completion to /model list, got %q", got)
	}
}

func TestSlashSuggestions_CycleWraps(t *testing.T) {
	a := newSlashTestApp()
	a.input = a.input.SetValue("/")
	a.refreshSlashSuggestions()
	if len(a.slashCandidates) < 2 {
		t.Fatalf("expected multiple slash candidates, got %d", len(a.slashCandidates))
	}

	a.slashSelected = 0
	a.cycleSlash(-1)
	if a.slashSelected != len(a.slashCandidates)-1 {
		t.Fatalf("expected wrap to last, got %d", a.slashSelected)
	}

	a.cycleSlash(1)
	if a.slashSelected != 0 {
		t.Fatalf("expected wrap back to first, got %d", a.slashSelected)
	}
}
