package panels

import (
	"testing"

	"github.com/mindspore-lab/mindspore-cli/ui/theme"
)

func TestInitStyles_ModelPickerDescUsesMutedColor(t *testing.T) {
	original := theme.Current
	t.Cleanup(func() { theme.Current = original })

	theme.Current = theme.Dark
	InitStyles()

	got := modelPickerDescStyle.GetForeground()
	want := theme.Current.TextMuted
	if got != want {
		t.Fatalf("modelPickerDescStyle.GetForeground() = %v, want %v", got, want)
	}
}
