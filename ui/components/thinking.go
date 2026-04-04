package components

import (
	"fmt"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

// thinkingSpinnerFrames alternates ✻ visibility for a pulsing effect.
var thinkingSpinnerFrames = []string{
	"✻", "✻", "✻", "✻", "✹", "✧", "·", "✧", "✹", "✻",
}

// Style vars are populated by InitStyles() in styles.go.
var thinkingStyle lipgloss.Style
var thinkingSpinnerStyle lipgloss.Style

// ThinkingSpinner shows a "⣻ Thinking..." animated indicator
type ThinkingSpinner struct {
	frame      int
	text       string
	tipText    string
	TextStyle  lipgloss.Style
	DotStyle   lipgloss.Style
	customStyle bool
	Elapsed    time.Duration
}

// NewThinkingSpinner creates a new thinking spinner with default text.
func NewThinkingSpinner() ThinkingSpinner {
	return ThinkingSpinner{
		frame:   0,
		text:    "Thinking...",
		tipText: RandomTip(),
	}
}

// NewThinkingSpinnerWithText creates a spinner with custom text.
func NewThinkingSpinnerWithText(text string) ThinkingSpinner {
	return ThinkingSpinner{
		frame:   0,
		text:    text,
		tipText: RandomTip(),
	}
}

// SetText updates the spinner text.
func (t *ThinkingSpinner) SetText(text string) {
	t.text = text
}

// TickMsg is the message sent on each animation tick.
type TickMsg struct {
	Time time.Time
}

// Tick returns a command that ticks the spinner.
func (t ThinkingSpinner) Tick() tea.Cmd {
	return tea.Tick(80*time.Millisecond, func(t time.Time) tea.Msg {
		return TickMsg{Time: t}
	})
}

// Update advances the spinner animation.
func (t ThinkingSpinner) Update(msg tea.Msg) (ThinkingSpinner, tea.Cmd) {
	switch msg.(type) {
	case TickMsg:
		t.frame = (t.frame + 1) % len(thinkingSpinnerFrames)
		return t, t.Tick()
	default:
		return t, nil
	}
}

// SetStyle sets custom dot and text styles.
func (t *ThinkingSpinner) SetStyle(dot, text lipgloss.Style) {
	t.DotStyle = dot
	t.TextStyle = text
	t.customStyle = true
}

// View renders the thinking spinner.
func (t ThinkingSpinner) View() string {
	frame := thinkingSpinnerFrames[t.frame]
	text := t.text
	if t.Elapsed >= time.Second {
		text += " " + model.FormatWaitDuration(t.Elapsed)
	}
	dotStyle := thinkingSpinnerStyle
	textStyle := thinkingStyle
	if t.customStyle {
		dotStyle = t.DotStyle
		textStyle = t.TextStyle
	}
	return fmt.Sprintf("%s %s",
		dotStyle.Render(frame),
		textStyle.Render(text))
}

// FrameView renders only the animated spinner character (no text).
func (t ThinkingSpinner) FrameView() string {
	return thinkingSpinnerStyle.Render(thinkingSpinnerFrames[t.frame])
}

// IsThinking returns true if the spinner is active.
func (t ThinkingSpinner) IsThinking() bool {
	return true
}

// Frame returns the current frame index (for testing).
func (t ThinkingSpinner) Frame() int {
	return t.frame
}

// ViewWithTip renders the thinking spinner with a tip line below.
func (t ThinkingSpinner) ViewWithTip() string {
	base := t.View()
	tip := fmt.Sprintf("  %s  %s",
		tipPrefixStyle.Render("⎿"),
		tipStyle.Render("Tip: "+t.tipText))
	return base + "\n" + tip
}

// Reset resets the spinner to the first frame and re-rolls the tip.
func (t *ThinkingSpinner) Reset() {
	t.frame = 0
	t.tipText = RandomTip()
}
