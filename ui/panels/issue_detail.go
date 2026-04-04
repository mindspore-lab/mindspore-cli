package panels

import (
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/vigo999/mindspore-cli/ui/model"
	"github.com/vigo999/mindspore-cli/ui/render"
)

func RenderIssueDetail(width, height int, st model.IssueDetailState) string {
	bodyWidth := width - 2
	if bodyWidth < 1 {
		bodyWidth = 1
	}
	padLeft := lipgloss.NewStyle().PaddingLeft(2)

	if st.Err != "" {
		lines := []string{
			render.TitleStyle.Render("ISSUE"),
			"",
			render.ValueStyle.Render("failed to load issue"),
			render.ValueStyle.Render(st.Err),
		}
		return padBugBody(padLeft.Render(strings.Join(lines, "\n")), height)
	}
	if st.Issue == nil {
		return padBugBody(padLeft.Render(render.TitleStyle.Render("ISSUE")), height)
	}
	return padBugBody(padLeft.Render(render.IssueDetail(*st.Issue, st.Notes, st.Activity, bodyWidth, height)), height)
}
