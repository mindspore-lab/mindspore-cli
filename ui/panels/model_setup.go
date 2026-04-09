package panels

import (
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

// Style vars are populated by InitStyles() in styles.go.
var (
	setupTitleStyle      lipgloss.Style
	setupNormalStyle     lipgloss.Style
	setupSelectedStyle   lipgloss.Style
	setupDisabledStyle   lipgloss.Style
	setupHintStyle       lipgloss.Style
	setupErrorStyle      lipgloss.Style
	setupLabelStyle      lipgloss.Style
	setupBadgeStyle      lipgloss.Style
	modelPickerDescStyle lipgloss.Style
	setupBorderStyle     lipgloss.Style
)

const popupVisibleOptions = 12

// RenderSetupPopup renders the multi-step model setup popup.
func RenderSetupPopup(popup *model.SetupPopup) string {
	switch popup.Screen {
	case model.SetupScreenModeSelect:
		return renderModeSelect(popup)
	case model.SetupScreenPresetPicker:
		return renderPresetPicker(popup)
	case model.SetupScreenTokenInput:
		return renderTokenInput(popup)
	case model.SetupScreenEnvInfo:
		return renderEnvInfo(popup)
	default:
		return ""
	}
}

const (
	modeMSCLIProvided = "mscli-provided"
	modeModeOwn       = "own"
	popupContentWidth = 56
)

func renderModeSelect(popup *model.SetupPopup) string {
	title := popup.Title
	if title == "" {
		title = "Model Setup"
	}
	modes := []struct {
		label string
		mode  string
	}{
		{"mscli-provided model", modeMSCLIProvided},
		{"your own model", modeModeOwn},
	}

	maxW := len(title)
	for _, m := range modes {
		if w := 2 + len(m.label) + 12; w > maxW {
			maxW = w
		}
	}

	var lines []string
	lines = append(lines, setupTitleStyle.Width(maxW).Render(title))
	lines = append(lines, "")
	for i, m := range modes {
		marker := "  "
		style := setupNormalStyle
		if i == popup.ModeSelected {
			marker = "❯ "
			style = setupSelectedStyle
		}
		label := m.label
		if popup.CurrentMode == m.mode {
			label += setupBadgeStyle.Render("  (current)")
		}
		lines = append(lines, marker+style.Render(label))
	}
	lines = append(lines, "")
	hint := "↑/↓ select · enter confirm"
	if popup.CanEscape {
		hint += " · esc cancel"
	}
	lines = append(lines, setupHintStyle.Render(hint))

	return setupBorderStyle.Render(strings.Join(lines, "\n"))
}

func renderPresetPicker(popup *model.SetupPopup) string {
	title := popup.Title
	if title == "" {
		title = "mscli-provided"
	}
	filtered := filteredPopupOptions(popup.PresetOptions, popup.SearchQuery)
	visible, start, end := popupOptionWindow(filtered, popup.PresetSelected, popupVisibleOptions)
	maxW := popupContentWidth

	var lines []string
	lines = append(lines, setupTitleStyle.Width(maxW).Render(title))
	lines = append(lines, "")
	lines = append(lines, setupLabelStyle.Render("Search"))
	lines = append(lines, renderSearchField(popup.SearchQuery, maxW))
	lines = append(lines, "")
	for _, row := range visible {
		if row.Separator {
			lines = append(lines, "")
			continue
		}
		opt := row.Option
		marker := "  "
		style := setupNormalStyle
		if opt.Disabled {
			style = setupDisabledStyle
		}
		if row.Index == popup.PresetSelected {
			marker = "❯ "
			if !opt.Disabled {
				style = setupSelectedStyle
			}
		}
		label := opt.Label
		if opt.Desc != "" {
			label += " " + opt.Desc
		}
		if opt.ID == popup.CurrentPreset {
			label += setupBadgeStyle.Render("  (current)")
		}
		lines = append(lines, renderWrappedPopupOption(label, marker, style, maxW))
	}
	lines = append(lines, "")
	scrollHint := "↑/↓ select · enter · esc back"
	if start > 0 || end < len(filtered) {
		scrollHint = "↑/↓ scroll · enter · esc back"
	}
	lines = append(lines, setupHintStyle.Render(scrollHint))

	return setupBorderStyle.Render(strings.Join(lines, "\n"))
}

func renderTokenInput(popup *model.SetupPopup) string {
	title := popup.SelectedPreset.Label
	if title == "" {
		title = "Enter Token"
	}
	inputLabel := popup.InputLabel
	if inputLabel == "" {
		inputLabel = "Token"
	}

	var lines []string
	lines = append(lines, setupTitleStyle.Width(40).Render(title))
	lines = append(lines, "")
	lines = append(lines, setupLabelStyle.Render(inputLabel+": ")+renderTokenField(popup.TokenValue))
	if popup.TokenError != "" {
		lines = append(lines, "")
		lines = append(lines, setupErrorStyle.Render(popup.TokenError))
	}
	lines = append(lines, "")
	lines = append(lines, setupHintStyle.Render("enter apply · esc back"))

	return setupBorderStyle.Render(strings.Join(lines, "\n"))
}

// Style vars populated by InitStyles() in styles.go.
var (
	tokenCursorStyle lipgloss.Style
	tokenTextStyle   lipgloss.Style
)

func renderTokenField(token string) string {
	if len(token) == 0 {
		return tokenCursorStyle.Render(" ")
	}
	return tokenTextStyle.Render(maskToken(token)) + tokenCursorStyle.Render(" ")
}

func maskToken(token string) string {
	runes := []rune(token)
	n := len(runes)
	if n <= 8 {
		return token
	}
	return string(runes[:4]) + strings.Repeat("·", n-8) + string(runes[n-4:])
}

func renderEnvInfo(popup *model.SetupPopup) string {
	var lines []string
	lines = append(lines, setupTitleStyle.Width(50).Render("Your Own Model"))
	lines = append(lines, "")
	lines = append(lines, setupLabelStyle.Render("Set environment variables:"))
	lines = append(lines, "")
	lines = append(lines, setupNormalStyle.Render("  export MSCLI_PROVIDER=openai-completion"))
	lines = append(lines, setupNormalStyle.Render("  export MSCLI_BASE_URL=https://api.openai.com/v1"))
	lines = append(lines, setupNormalStyle.Render("  export MSCLI_API_KEY=sk-..."))
	lines = append(lines, setupNormalStyle.Render("  export MSCLI_MODEL=gpt-5.4"))
	lines = append(lines, "")
	lines = append(lines, setupHintStyle.Render("Then restart mscli."))
	lines = append(lines, "")
	lines = append(lines, setupHintStyle.Render("esc back"))

	return setupBorderStyle.Render(strings.Join(lines, "\n"))
}

func popupOptionWindow[T any](options []T, selected, maxVisible int) ([]T, int, int) {
	total := len(options)
	if total == 0 || maxVisible <= 0 || total <= maxVisible {
		return options, 0, total
	}
	if selected < 0 {
		selected = 0
	}
	if selected >= total {
		selected = total - 1
	}
	start := selected - maxVisible/2
	if start < 0 {
		start = 0
	}
	end := start + maxVisible
	if end > total {
		end = total
		start = end - maxVisible
	}
	return options[start:end], start, end
}

type popupOptionRow struct {
	Index     int
	Option    model.SelectionOption
	Separator bool
}

func filteredPopupOptions(options []model.SelectionOption, query string) []popupOptionRow {
	rows := make([]popupOptionRow, 0)
	indices := model.FilteredOptionIndicesForRender(options, query)
	for _, idx := range indices {
		if idx < 0 {
			rows = append(rows, popupOptionRow{Separator: true})
			continue
		}
		rows = append(rows, popupOptionRow{
			Index:  idx,
			Option: options[idx],
		})
	}
	return rows
}

func renderWrappedPopupOption(text, marker string, style lipgloss.Style, width int) string {
	if width < 4 {
		width = 4
	}
	contentWidth := width - 2
	if contentWidth < 1 {
		contentWidth = 1
	}
	wrapped := lipgloss.NewStyle().Width(contentWidth).Render(text)
	lines := strings.Split(wrapped, "\n")
	for i, line := range lines {
		prefix := "  "
		if i == 0 {
			prefix = marker
		}
		lines[i] = prefix + style.Render(line)
	}
	return strings.Join(lines, "\n")
}

func renderSearchField(query string, width int) string {
	if width < 4 {
		width = 4
	}
	text := query
	style := setupNormalStyle
	if strings.TrimSpace(text) == "" {
		text = "Type to filter"
		style = setupHintStyle
	}
	return style.Width(width).Render(text)
}
