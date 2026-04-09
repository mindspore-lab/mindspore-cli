package panels

import (
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
	"github.com/mindspore-lab/mindspore-cli/ui/theme"
)

const (
	modelBrowserCardHeight  = 19
	modelBrowserRearWidth   = 4
	modelBrowserRearDrop    = 2
	modelBrowserFocusPad    = 2
	modelBrowserLeftBiasPad = 1
	modelBrowserFooterHint  = "↑/↓ select · enter choose · esc"
)

func RenderModelBrowserPopup(popup *model.ModelBrowserPopup) string {
	if popup == nil {
		return ""
	}
	if popup.Focus == model.ModelBrowserFocusProvider {
		return renderProviderBrowserCard(popup)
	}
	return renderModelBrowserCard(popup)
}

func renderProviderBrowserCard(popup *model.ModelBrowserPopup) string {
	if popup.ProviderInput != nil {
		lines := renderCardHeader(model.ModelBrowserFocusProvider)
		lines = append(lines, "")
		lines = append(lines, setupLabelStyle.Render(popup.ProviderInput.Option.Label))
		lines = append(lines, "")
		lines = append(lines, setupLabelStyle.Render(popup.ProviderInput.Label+": ")+renderTokenField(popup.ProviderInput.Value))
		if popup.ProviderInput.Error != "" {
			lines = append(lines, "")
			lines = append(lines, setupErrorStyle.Render(popup.ProviderInput.Error))
		}
		lines = append(lines, "")
		lines = append(lines, setupHintStyle.Render(modelBrowserFooterHint))
		return renderFixedBrowserCard(lines)
	}

	return renderBrowserSelectionCard(model.ModelBrowserFocusProvider, &popup.Providers)
}

func renderModelBrowserCard(popup *model.ModelBrowserPopup) string {
	if popup.HasModels() {
		return renderBrowserSelectionCard(model.ModelBrowserFocusModel, &popup.Models)
	}

	lines := renderCardHeader(model.ModelBrowserFocusModel)
	lines = append(lines, "")
	lines = append(lines, setupLabelStyle.Render("Search"))
	lines = append(lines, renderSearchField("", popupContentWidth))
	lines = append(lines, "")
	lines = append(lines, setupHintStyle.Render("No models yet. Add a provider first."))
	lines = append(lines, "")
	lines = append(lines, setupHintStyle.Render(modelBrowserFooterHint))
	return renderFixedBrowserCard(lines)
}

func renderBrowserSelectionCard(focus model.ModelBrowserFocus, popup *model.SelectionPopup) string {
	filtered := filteredPopupOptions(popup.Options, popup.SearchQuery)
	visible, _, _ := popupOptionWindow(filtered, popup.Selected, popupVisibleOptions)
	maxW := popupContentWidth

	t := theme.Current
	normalStyle := lipgloss.NewStyle().Foreground(t.TextSecondary)
	selectedStyle := lipgloss.NewStyle().Foreground(t.Accent).Bold(true)
	disabledStyle := lipgloss.NewStyle().Foreground(t.TextMuted)

	lines := renderCardHeader(focus)
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
		style := normalStyle
		if opt.Disabled {
			style = disabledStyle
		}
		if row.Index == popup.Selected {
			marker = "❯ "
			if !opt.Disabled {
				style = selectedStyle
			}
		}
		if opt.ProviderRow {
			style = setupLabelStyle
			if row.Index == popup.Selected {
				style = selectedStyle
			}
		}
		if opt.DetailRow {
			lines = append(lines, renderWrappedPopupOption(opt.Label, "  ", setupHintStyle, maxW))
			continue
		}
		label := opt.Label
		if opt.Desc != "" {
			label += " " + modelPickerDescStyle.Render(opt.Desc)
		}
		if opt.ProviderRow && row.Index == popup.Selected && strings.TrimSpace(opt.DeleteProviderID) != "" {
			lines = append(lines, renderBrowserProviderRow(label, "double-press d to delete", marker, style, maxW))
			continue
		}
		lines = append(lines, renderWrappedPopupOption(label, marker, style, maxW))
	}
	lines = append(lines, "")
	lines = append(lines, setupHintStyle.Render(modelBrowserFooterHint))
	return renderFixedBrowserCard(lines)
}

func renderBrowserProviderRow(label, hint, marker string, style lipgloss.Style, width int) string {
	if width < 4 {
		width = 4
	}
	contentWidth := width - len([]rune(marker))
	if contentWidth < 1 {
		contentWidth = 1
	}
	hintRendered := setupHintStyle.Render(hint)
	hintWidth := lipgloss.Width(hintRendered)
	labelWidth := contentWidth - hintWidth
	if labelWidth < 1 {
		labelWidth = 1
	}
	left := style.Width(labelWidth).MaxWidth(labelWidth).Render(label)
	return marker + lipgloss.JoinHorizontal(lipgloss.Top, left, hintRendered)
}

func renderCardHeader(focus model.ModelBrowserFocus) []string {
	headerWidth := popupContentWidth
	leftText := "Select Providers"
	leftStyle := setupTitleStyle.Copy().Align(lipgloss.Left)
	rightText := "Select Models"
	rightStyle := setupTitleStyle.Copy().Align(lipgloss.Right)
	if focus == model.ModelBrowserFocusProvider {
		rightText = "→ Select Models"
		rightStyle = setupHintStyle.Copy().Align(lipgloss.Right)
	} else {
		leftText = "Select Providers ←"
		leftStyle = setupHintStyle.Copy().Align(lipgloss.Left)
	}
	left := leftStyle.Render(leftText)
	right := rightStyle.Render(rightText)
	line := lipgloss.JoinHorizontal(
		lipgloss.Top,
		lipgloss.NewStyle().Width(headerWidth-lipgloss.Width(right)).Render(left),
		right,
	)
	return []string{line}
}

func renderFixedBrowserCard(lines []string) string {
	if len(lines) == 0 {
		lines = []string{""}
	}
	footer := lines[len(lines)-1]
	bodyLines := append([]string(nil), lines[:len(lines)-1]...)
	for len(bodyLines) < modelBrowserCardHeight-1 {
		bodyLines = append(bodyLines, "")
	}
	if len(bodyLines) > modelBrowserCardHeight-1 {
		bodyLines = bodyLines[:modelBrowserCardHeight-1]
	}
	bodyLines = append(bodyLines, footer)
	body := make([]string, 0, len(lines))
	for _, line := range bodyLines {
		body = append(body, lipgloss.NewStyle().Width(popupContentWidth).MaxWidth(popupContentWidth).Render(line))
	}
	return setupBorderStyle.Render(strings.Join(body, "\n"))
}

func renderRearBrowserCard(leftSide bool) string {
	fullCard := renderBlankBrowserCardLines()
	lines := make([]string, 0, len(fullCard)+modelBrowserRearDrop)
	for i := 0; i < modelBrowserRearDrop; i++ {
		lines = append(lines, strings.Repeat(" ", modelBrowserRearWidth))
	}
	for _, line := range fullCard {
		if leftSide {
			lines = append(lines, trimLinePrefix(line, modelBrowserRearWidth))
			continue
		}
		lines = append(lines, trimLineSuffix(line, modelBrowserRearWidth))
	}
	return strings.Join(lines, "\n")
}

func renderBlankBrowserCardLines() []string {
	border := lipgloss.RoundedBorder()
	innerWidth := popupContentWidth + 4
	lines := make([]string, 0, modelBrowserCardHeight+2)
	lines = append(lines, border.TopLeft+strings.Repeat(border.Top, innerWidth)+border.TopRight)
	for i := 0; i < modelBrowserCardHeight; i++ {
		lines = append(lines, border.Left+strings.Repeat(" ", innerWidth)+border.Right)
	}
	lines = append(lines, border.BottomLeft+strings.Repeat(border.Bottom, innerWidth)+border.BottomRight)
	return lines
}

func trimLinePrefix(line string, width int) string {
	runes := []rune(line)
	if width <= 0 {
		return ""
	}
	if len(runes) <= width {
		return line
	}
	return string(runes[:width])
}

func trimLineSuffix(line string, width int) string {
	runes := []rune(line)
	if width <= 0 {
		return ""
	}
	if len(runes) <= width {
		return line
	}
	return string(runes[len(runes)-width:])
}

func overlayRearCard(mainCard, rearCard string, leftSide bool) string {
	mainLines := strings.Split(mainCard, "\n")
	rearLines := strings.Split(rearCard, "\n")
	mainWidth := 0
	if len(mainLines) > 0 {
		mainWidth = lipgloss.Width(mainLines[0])
	}
	out := make([]string, 0, maxInt(len(mainLines), len(rearLines)))
	for i := 0; i < maxInt(len(mainLines), len(rearLines)); i++ {
		mainLine := ""
		if i < len(mainLines) {
			mainLine = mainLines[i]
		}
		rearLine := ""
		if i < len(rearLines) {
			rearLine = rearLines[i]
		}
		if leftSide {
			out = append(out, rearLine+mainLine)
			continue
		}
		out = append(out, mainLine+strings.Repeat(" ", maxInt(0, mainWidth-lipgloss.Width(mainLine)))+rearLine)
	}
	return strings.Join(out, "\n")
}
