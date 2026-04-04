package components

import (
	"github.com/charmbracelet/lipgloss"
	"github.com/vigo999/mindspore-cli/ui/theme"
)

var tipPrefixStyle lipgloss.Style
var tipStyle lipgloss.Style

// InitStyles rebuilds all package-level style vars from theme.Current.
func InitStyles() {
	t := theme.Current

	// spinner.go
	spinnerStyle = lipgloss.NewStyle().Foreground(t.Thinking)

	// thinking.go
	thinkingStyle = lipgloss.NewStyle().Foreground(t.Thinking).Italic(true)
	thinkingSpinnerStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("212"))
	tipPrefixStyle = lipgloss.NewStyle().Foreground(t.TextSecondary)
	tipStyle = lipgloss.NewStyle().Foreground(t.TextMuted).Italic(true)

	// textinput.go
	sugCmdStyle = lipgloss.NewStyle().Foreground(t.TextPrimary)
	sugDescStyle = lipgloss.NewStyle().Foreground(t.TextSecondary)
	sugSelCmdStyle = lipgloss.NewStyle().Foreground(t.AccentAlt).Bold(true)
	sugSelDescStyle = lipgloss.NewStyle().Foreground(t.AccentAlt)
	separatorStyle = lipgloss.NewStyle().Foreground(t.Border)
}
