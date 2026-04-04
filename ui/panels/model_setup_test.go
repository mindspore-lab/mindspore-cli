package panels

import (
	"strings"
	"testing"

	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

func TestRenderSetupPopupModeSelect(t *testing.T) {
	popup := &model.SetupPopup{
		Screen:       model.SetupScreenModeSelect,
		ModeSelected: 0,
		CanEscape:    true,
	}
	result := RenderSetupPopup(popup)
	if !strings.Contains(result, "mscli-provided") {
		t.Error("expected 'mscli-provided' in output")
	}
	if !strings.Contains(result, "your own model") {
		t.Error("expected 'your own model' in output")
	}
}

func TestRenderSetupPopupPresetPicker(t *testing.T) {
	popup := &model.SetupPopup{
		Screen: model.SetupScreenPresetPicker,
		PresetOptions: []model.SelectionOption{
			{ID: "kimi-k2.5-free", Label: "kimi-k2.5 [free]"},
			{ID: "deepseek-v3", Label: "deepseek-v3"},
			{ID: "glm-4.7", Label: "glm-4.7 (coming soon)", Disabled: true},
		},
		PresetSelected: 0,
		CanEscape:      true,
	}
	result := RenderSetupPopup(popup)
	if !strings.Contains(result, "kimi-k2.5 [free]") {
		t.Error("expected active preset label in output")
	}
	if !strings.Contains(result, "deepseek-v3") {
		t.Error("expected deepseek preset in output")
	}
	if !strings.Contains(result, "glm-4.7 (coming soon)") {
		t.Error("expected coming-soon preset in output")
	}
}

func TestRenderSetupPopupTokenInput(t *testing.T) {
	popup := &model.SetupPopup{
		Screen:     model.SetupScreenTokenInput,
		TokenValue: "sk-abc",
		CanEscape:  true,
		SelectedPreset: model.SelectionOption{
			ID:    "kimi-k2.5-free",
			Label: "kimi-k2.5 [free]",
		},
	}
	result := RenderSetupPopup(popup)
	if !strings.Contains(result, "Token") {
		t.Error("expected 'Token' label in output")
	}
}

func TestRenderSetupPopupEnvInfo(t *testing.T) {
	popup := &model.SetupPopup{
		Screen:    model.SetupScreenEnvInfo,
		CanEscape: true,
	}
	result := RenderSetupPopup(popup)
	if !strings.Contains(result, "MSCLI_PROVIDER") {
		t.Error("expected env var example in output")
	}
	if !strings.Contains(result, "MSCLI_API_KEY") {
		t.Error("expected MSCLI_API_KEY in output")
	}
}
