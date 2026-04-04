package render

import (
	"github.com/charmbracelet/lipgloss"
	"github.com/vigo999/mindspore-cli/ui/theme"
)

// InitStyles rebuilds all package-level style vars from theme.Current.
func InitStyles() {
	t := theme.Current

	// box.go
	BoxBorderStyle = lipgloss.NewStyle().Foreground(t.TextSecondary)
	TitleStyle = lipgloss.NewStyle().Bold(true).Foreground(t.TextPrimary)
	LabelStyle = lipgloss.NewStyle().Foreground(t.TextSecondary)
	ValueStyle = lipgloss.NewStyle().Foreground(t.TextPrimary)
	StatusOpenStyle = lipgloss.NewStyle().Foreground(t.Success)
	StatusDoingStyle = lipgloss.NewStyle().Foreground(t.Warning)
	ActivityStyle = lipgloss.NewStyle().Foreground(t.TextSecondary)

	// bug_index.go
	statusClosedStyle = lipgloss.NewStyle().Foreground(t.TextSecondary)
	bugRowSelectedStyle = lipgloss.NewStyle().Background(t.SelectionBG)
	bugHeaderStyle = lipgloss.NewStyle().Foreground(t.TextPrimary)

	// issues.go
	issueRowSelectedStyle = lipgloss.NewStyle().Background(t.SelectionBG)
}
