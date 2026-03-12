package panels

import (
	"strings"

	"github.com/charmbracelet/lipgloss"
)

var (
	bootBoxStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("245")).
			BorderStyle(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("239")).
			Padding(1, 3)

	bootMessageBaseStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("241")).
				Bold(true)

	bootMessageGlowStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("250")).
				Bold(true)

	bootMessageHotStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("255")).
				Bold(true)
)

// RenderBootScreen renders a centered splash screen shown before the TUI opens.
func RenderBootScreen(width, height, highlight int) string {
	content := bootBoxStyle.Render(
		renderBootShimmer("MindSpore AI Infra Agent", highlight),
	)

	if width <= 0 || height <= 0 {
		return content
	}

	return lipgloss.Place(width, height, lipgloss.Center, lipgloss.Center, content)
}

func renderBootShimmer(text string, highlight int) string {
	runes := []rune(text)
	if len(runes) == 0 {
		return ""
	}

	parts := make([]string, len(runes))
	bandCenter := highlight % (len(runes) + 6)
	bandCenter -= 3

	for i, r := range runes {
		style := bootMessageBaseStyle
		if r != ' ' {
			dist := absInt(i - bandCenter)
			switch {
			case dist == 0:
				style = bootMessageHotStyle
			case dist <= 2:
				style = bootMessageGlowStyle
			}
		}
		parts[i] = style.Render(string(r))
	}

	return strings.Join(parts, "")
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
