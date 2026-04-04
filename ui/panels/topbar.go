package panels

import (
	"fmt"
	"os"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

// Style vars are populated by InitStyles() in styles.go.
var (
	topBarStyle      = lipgloss.NewStyle().Padding(0, 1) // layout-only, not themed
	brandStyle       lipgloss.Style
	infoStyle        lipgloss.Style
	sepStyle         lipgloss.Style
	dividerStyle     lipgloss.Style
	bannerLabelStyle lipgloss.Style
	bannerValueStyle lipgloss.Style
	bannerDimStyle   lipgloss.Style
)

// RenderTopBar renders the top status bar.
// When showBanner is true, a second line with workdir + repo is shown.
func RenderTopBar(s model.State, width int) string {
	content := brandStyle.Render(s.Version)
	lineWidth := lipgloss.Width(content)
	padding := 0
	if width > lineWidth {
		padding = (width - lineWidth) / 2
	}
	line1 := topBarStyle.Render(strings.Repeat(" ", padding) + content)

	return line1
}

func shortenPath(p string) string {
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return p
	}
	if strings.HasPrefix(p, home) {
		return "~" + p[len(home):]
	}
	return p
}

func formatTokens(n int) string {
	switch {
	case n >= 1_000_000:
		return fmt.Sprintf("%.1fM", float64(n)/1_000_000)
	case n >= 1_000:
		return fmt.Sprintf("%.1fk", float64(n)/1_000)
	default:
		return fmt.Sprintf("%d", n)
	}
}

func repeatChar(ch string, n int) string {
	return strings.Repeat(ch, n)
}
