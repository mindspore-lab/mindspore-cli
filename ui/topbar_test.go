package ui

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/vigo999/mindspore-cli/ui/model"
)

func TestViewOmitsPersistentTopBarAndViewportFill(t *testing.T) {
	app := New(nil, nil, "MindSpore CLI. test", ".", "", "demo-model", 4096)
	app.bootActive = false
	app.state = app.state.WithMessage(model.Message{Kind: model.MsgAgent, Content: "history line"})

	next, _ := app.Update(tea.WindowSizeMsg{Width: 100, Height: 24})
	app = next.(App)

	view := app.View()
	if strings.Contains(view, "MindSpore CLI. test") {
		t.Fatalf("expected inline view to omit persistent top bar, got:\n%s", view)
	}
	if strings.Contains(view, "history line") {
		t.Fatalf("expected history to stay out of the live frame, got:\n%s", view)
	}
	if got := strings.Count(view, "\n") + 1; got >= 24 {
		t.Fatalf("expected view to avoid filling terminal height, got %d lines", got)
	}
}

func TestRenderBannerIncludesMetadata(t *testing.T) {
	banner := RenderBanner("MindSpore CLI. test", "/tmp/project", "github.com/vigo999/mindspore-cli", "demo-model", 4096)
	for _, want := range []string{
		"MindSpore CLI",
		"demo-model",
		"/tmp/project",
	} {
		if !strings.Contains(banner, want) {
			t.Fatalf("expected banner to include %q, got:\n%s", want, banner)
		}
	}
	if !strings.Contains(banner, "model: demo-model") {
		t.Fatalf("expected banner rows to stay left aligned, got:\n%s", banner)
	}
}
