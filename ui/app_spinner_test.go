package ui

import (
	"testing"

	"github.com/vigo999/ms-cli/ui/model"
)

func TestSpinnerTickRerendersThinkingViewport(t *testing.T) {
	eventCh := make(chan model.Event)
	a := New(eventCh, nil, "v0.0.0", ".", "", "openrouter", "deepseek", 24000)
	a.width = 120
	a.height = 40
	a.resizeViewport()
	a.state = a.state.WithMessage(model.Message{Kind: model.MsgThinking})
	a.updateViewport()

	before := a.viewport.View()

	tickCmd := a.spinner.Model.Tick
	msg := tickCmd()
	updatedModel, _ := a.Update(msg)
	updated, ok := updatedModel.(App)
	if !ok {
		t.Fatalf("expected App model")
	}

	after := updated.viewport.View()
	if before == after {
		t.Fatalf("expected spinner tick to rerender viewport content")
	}
}
