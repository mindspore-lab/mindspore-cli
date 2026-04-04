package panels

import (
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/vigo999/mindspore-cli/ui/model"
	"github.com/vigo999/mindspore-cli/ui/render"
)

func RenderIssueIndex(width, height int, st model.IssueIndexState) string {
	if st.Err != "" {
		lines := []string{
			renderCenteredIssueTitle(width, st.Filter),
			"",
			render.ValueStyle.Render("failed to load issues"),
			render.ValueStyle.Render(st.Err),
		}
		return padBugBody(strings.Join(lines, "\n"), height)
	}
	body := render.IssueIndex(st.Items, st.Cursor, width, height)
	lines := append([]string{renderCenteredIssueTitle(width, st.Filter)}, strings.Split(body, "\n")...)
	return padBugBody(strings.Join(lines, "\n"), height)
}

func renderCenteredIssueTitle(width int, filter string) string {
	title := render.TitleStyle.Render("ISSUES") + render.LabelStyle.Render(" (filter:"+bugFilterLabel(filter)+")")
	return lipgloss.NewStyle().Width(width).PaddingLeft(2).Align(lipgloss.Left).Render(title)
}
