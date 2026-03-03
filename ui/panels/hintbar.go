package panels

import (
	"strings"

	"github.com/charmbracelet/lipgloss"
)

var (
	hintDividerStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("238"))

	hintTextStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("240")).
			PaddingLeft(1)

	hintKeyStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("250"))

	hintDescStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("240"))

	hintSepStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("238"))

	slashSelectedStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("86")).
				Bold(true)

	slashOptionStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("241"))

	slashHeaderStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("244"))

	slashPrefixStyle = lipgloss.NewStyle().
				Foreground(lipgloss.Color("238"))
)

type hint struct {
	key  string
	desc string
}

var hints = []hint{
	{"/", "commands"},
	{"wheel", "scroll"},
	{"pgup/pgdn", "scroll"},
	{"ctrl+c", "interrupt"},
	{"ctrl+c x2", "quit"},
}

// RenderHintBar renders the bottom keybinding hint bar.
func RenderHintBar(width int, slashCandidates []string, slashSelected int) string {
	divider := hintDividerStyle.Render(repeatChar("━", width))

	if len(slashCandidates) > 0 {
		lines := make([]string, 0, len(slashCandidates)+1)
		lines = append(lines, hintTextStyle.Render("")+slashHeaderStyle.Render("slash commands (↑/↓ select, tab complete):"))
		for i, cmd := range slashCandidates {
			prefix := slashPrefixStyle.Render("  ")
			item := slashOptionStyle.Render(cmd)
			if i == slashSelected {
				prefix = slashSelectedStyle.Render("› ")
				item = slashSelectedStyle.Render(cmd)
			}
			lines = append(lines, hintTextStyle.Render("")+prefix+item)
		}
		return divider + "\n" + strings.Join(lines, "\n")
	}

	parts := make([]string, len(hints))
	for i, h := range hints {
		parts[i] = hintKeyStyle.Render(h.key) + " " + hintDescStyle.Render(h.desc)
	}

	sep := hintSepStyle.Render(" • ")
	line := hintTextStyle.Render("")
	for i, p := range parts {
		if i > 0 {
			line += sep
		}
		line += p
	}

	return divider + "\n" + line
}
