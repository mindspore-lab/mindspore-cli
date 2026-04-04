package panels

import (
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/vigo999/mindspore-cli/ui/model"
	"github.com/vigo999/mindspore-cli/ui/render"
)

func RenderBugIndex(width, height int, st model.BugIndexState) string {
	if st.Err != "" {
		lines := []string{
			renderCenteredBugTitle(width, st.Filter),
			"",
			render.ValueStyle.Render("failed to load bugs"),
			render.ValueStyle.Render(st.Err),
		}
		return padBugBody(strings.Join(lines, "\n"), height)
	}
	return padBugBody(renderBugIndexBody(width, height, st), height)
}

func renderBugIndexBody(width, height int, st model.BugIndexState) string {
	body := render.BugIndex(st.Items, st.Cursor, width, height)
	lines := append([]string{renderCenteredBugTitle(width, st.Filter)}, strings.Split(body, "\n")...)
	return strings.Join(lines, "\n")
}

func bugFilterLabel(filter string) string {
	filter = strings.TrimSpace(filter)
	if filter == "" {
		return "all"
	}
	return filter
}

func padBugBody(body string, height int) string {
	if height <= 0 {
		return body
	}
	bodyHeight := lipgloss.Height(body)
	if bodyHeight >= height {
		return body
	}
	return strings.Repeat("\n", height-bodyHeight) + body
}

func renderCenteredBugTitle(width int, filter string) string {
	title := render.TitleStyle.Render("BUGS") + render.LabelStyle.Render(" (filter:"+bugFilterLabel(filter)+")")
	return lipgloss.NewStyle().Width(width).PaddingLeft(2).Align(lipgloss.Left).Render(title)
}
