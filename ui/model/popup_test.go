package model

import "testing"

func TestSetupPopupPresetNavigatesAllItems(t *testing.T) {
	popup := &SetupPopup{
		Screen: SetupScreenPresetPicker,
		PresetOptions: []SelectionOption{
			{ID: "kimi-k2.5-free", Label: "kimi-k2.5 [free]"},
			{ID: "deepseek-v3", Label: "deepseek-v3"},
			{ID: "glm-4.7", Label: "glm-4.7 (coming soon)", Disabled: true},
			{ID: "minimax-m2.7", Label: "minimax-m2.7 (coming soon)", Disabled: true},
		},
		PresetSelected: 0,
	}

	// Cursor skips disabled items.
	popup.MovePresetSelection(1)
	if popup.PresetSelected != 1 {
		t.Errorf("expected 1, got %d", popup.PresetSelected)
	}
	popup.MovePresetSelection(1)
	if popup.PresetSelected != 0 {
		t.Errorf("expected wrap to 0, got %d", popup.PresetSelected)
	}
	// Wraps around
	popup.MovePresetSelection(1)
	popup.MovePresetSelection(1)
	if popup.PresetSelected != 0 {
		t.Errorf("expected wrap to 0, got %d", popup.PresetSelected)
	}
}

func TestSetupPopupMoveMode(t *testing.T) {
	popup := &SetupPopup{
		Screen:       SetupScreenModeSelect,
		ModeSelected: 0,
	}
	popup.MoveModeSelection(1)
	if popup.ModeSelected != 1 {
		t.Errorf("expected 1, got %d", popup.ModeSelected)
	}
	popup.MoveModeSelection(1)
	if popup.ModeSelected != 0 {
		t.Errorf("expected wrap to 0, got %d", popup.ModeSelected)
	}
}
