package panels

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

var (
	sessionPickerTitleStyle        lipgloss.Style
	sessionPickerNormalStyle       lipgloss.Style
	sessionPickerSelectedStyle     lipgloss.Style
	sessionPickerMetaStyle         lipgloss.Style
	sessionPickerSelectedMetaStyle lipgloss.Style
	sessionPickerPreviewStyle      lipgloss.Style
	sessionPickerHintStyle         lipgloss.Style
	sessionPickerEmptyStyle        lipgloss.Style
	sessionPickerBorderStyle       lipgloss.Style
)

func RenderSessionPicker(picker *model.SessionPicker, width, height int) string {
	if picker == nil {
		return ""
	}

	boxWidth := width - 4
	if boxWidth < 56 {
		boxWidth = width
	}
	boxHeight := height - 2
	if boxHeight < 12 {
		boxHeight = height
	}
	contentWidth := maxInt(20, boxWidth-6)
	contentHeight := maxInt(8, boxHeight-4)

	title := "Resume Session"
	if picker.Mode == model.SessionPickerReplay {
		title = "Replay Session"
		if speed := replaySpeedLabel(picker.ReplaySpeed); speed != "" {
			title += "  " + speed
		}
	}

	lines := []string{
		sessionPickerTitleStyle.Width(contentWidth).Render(title),
		sessionPickerMetaStyle.Render("Current workdir session history"),
		"",
	}

	bodyHeight := contentHeight - len(lines) - 2
	if bodyHeight < 3 {
		bodyHeight = 3
	}
	lines = append(lines, renderSessionPickerBody(picker, contentWidth, bodyHeight)...)
	lines = append(lines, "")
	lines = append(lines, sessionPickerHintStyle.Render("↑/↓ select · enter confirm · esc cancel"))

	box := sessionPickerBorderStyle.
		Width(boxWidth).
		Height(boxHeight).
		Render(strings.Join(lines, "\n"))
	return lipgloss.Place(width, height, lipgloss.Center, lipgloss.Center, box)
}

func renderSessionPickerBody(picker *model.SessionPicker, width, height int) []string {
	if len(picker.Items) == 0 {
		msg := picker.EmptyMessage
		if strings.TrimSpace(msg) == "" {
			msg = "No saved sessions found for this workdir."
		}
		return []string{sessionPickerEmptyStyle.Width(width).Render(msg)}
	}

	const linesPerItem = 5
	visibleCount := height / linesPerItem
	if visibleCount < 1 {
		visibleCount = 1
	}
	if visibleCount > len(picker.Items) {
		visibleCount = len(picker.Items)
	}

	start := picker.Selected - visibleCount/2
	if start < 0 {
		start = 0
	}
	if maxStart := len(picker.Items) - visibleCount; start > maxStart {
		start = maxStart
	}
	end := start + visibleCount

	lines := make([]string, 0, visibleCount*linesPerItem)
	if start > 0 {
		lines = append(lines, sessionPickerMetaStyle.Render(fmt.Sprintf("%d older session(s) above", start)))
	} else {
		lines = append(lines, "")
	}

	blockWidth := width - 2
	if blockWidth < 12 {
		blockWidth = width
	}
	for idx := start; idx < end; idx++ {
		item := picker.Items[idx]
		selected := idx == picker.Selected
		lines = append(lines, renderSessionPickerItem(item, selected, blockWidth)...)
	}
	if end < len(picker.Items) {
		lines = append(lines, sessionPickerMetaStyle.Render(fmt.Sprintf("%d older session(s) below", len(picker.Items)-end)))
	}

	return lines
}

func renderSessionPickerItem(item model.SessionPickerItem, selected bool, width int) []string {
	marker := "  "
	idStyle := sessionPickerNormalStyle
	metaStyle := sessionPickerMetaStyle
	previewStyle := sessionPickerPreviewStyle
	if selected {
		marker = "> "
		idStyle = sessionPickerSelectedStyle
		metaStyle = sessionPickerSelectedMetaStyle
		previewStyle = sessionPickerSelectedStyle
	}

	previewWidth := width - 2
	if previewWidth < 8 {
		previewWidth = width
	}

	return []string{
		marker + idStyle.Render(item.ID),
		"  " + metaStyle.Render("Created: "+item.CreatedAt.Format("2006-01-02 15:04:05")),
		"  " + metaStyle.Render("Updated: "+item.UpdatedAt.Format("2006-01-02 15:04:05")),
		"  " + previewStyle.Render(truncateSessionPickerText(item.FirstUserInput, previewWidth)),
		"",
	}
}

func replaySpeedLabel(speed float64) string {
	speed = normalizeReplaySpeed(speed)
	if speed == 1 {
		return ""
	}
	return fmt.Sprintf("(%.2gx)", speed)
}

func normalizeReplaySpeed(speed float64) float64 {
	if speed <= 0 {
		return 1
	}
	return speed
}

func truncateSessionPickerText(text string, width int) string {
	text = strings.TrimSpace(strings.Join(strings.Fields(text), " "))
	if width <= 3 || lipgloss.Width(text) <= width {
		return text
	}

	runes := []rune(text)
	for len(runes) > 0 && lipgloss.Width(string(runes)+"...") > width {
		runes = runes[:len(runes)-1]
	}
	return string(runes) + "..."
}
