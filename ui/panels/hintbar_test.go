package panels

import "testing"

func TestFormatCtxHintUsesPercentageByDefault(t *testing.T) {
	if got, want := formatCtxHint(250, 1000, false), "ctx: 75% left"; got != want {
		t.Fatalf("formatCtxHint() = %q, want %q", got, want)
	}
}

func TestFormatCtxHintUsesConcreteCountsInDebugMode(t *testing.T) {
	if got, want := formatCtxHint(2500, 128000, true), "ctx: 2500/128000"; got != want {
		t.Fatalf("formatCtxHint() = %q, want %q", got, want)
	}
}
