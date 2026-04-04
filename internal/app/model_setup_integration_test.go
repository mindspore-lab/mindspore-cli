package app

import (
	"strings"
	"testing"

	"github.com/mindspore-lab/mindspore-cli/configs"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

func TestModelCommand_OpensSetupPopupWithState(t *testing.T) {
	eventCh := make(chan model.Event, 64)
	app := &Application{
		EventCh:             eventCh,
		Config:              configs.DefaultConfig(),
		activeModelPresetID: "kimi-k2.5-free",
		llmReady:            true,
	}

	app.cmdModel(nil)

	var popup *model.SetupPopup
	for len(eventCh) > 0 {
		ev := <-eventCh
		if ev.Type == model.ModelSetupOpen {
			popup = ev.SetupPopup
		}
	}
	if popup == nil {
		t.Fatal("expected setup popup to open")
	}
	if !popup.CanEscape {
		t.Error("expected CanEscape=true from /model")
	}
	if popup.CurrentMode != modelModeMSCLIProvided {
		t.Errorf("expected current mode %q, got %q", modelModeMSCLIProvided, popup.CurrentMode)
	}
	if popup.CurrentPreset != "kimi-k2.5-free" {
		t.Errorf("expected current preset 'kimi-k2.5-free', got %q", popup.CurrentPreset)
	}

	disabledCount := 0
	for _, opt := range popup.PresetOptions {
		if opt.Disabled {
			disabledCount++
			if !strings.Contains(opt.Label, "coming soon") {
				t.Errorf("disabled option %q should contain 'coming soon'", opt.Label)
			}
		}
	}
	if disabledCount != 2 {
		t.Errorf("expected 2 disabled (coming soon) presets, got %d", disabledCount)
	}
}
