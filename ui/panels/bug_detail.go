package panels

import (
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/mindspore-lab/mindspore-cli/ui/model"
	"github.com/mindspore-lab/mindspore-cli/ui/render"
)

func RenderBugDetail(width, height int, st model.BugDetailState) string {
	bodyWidth := width - 2
	if bodyWidth < 1 {
		bodyWidth = 1
	}
	padLeft := lipgloss.NewStyle().PaddingLeft(2)

	if st.Err != "" {
		lines := []string{
			render.TitleStyle.Render("BUG"),
			"",
			render.ValueStyle.Render("failed to load bug"),
			render.ValueStyle.Render(st.Err),
		}
		return padBugBody(padLeft.Render(strings.Join(lines, "\n")), height)
	}
	if st.Bug == nil {
		return padBugBody(padLeft.Render(render.TitleStyle.Render("BUG")), height)
	}
	return padBugBody(padLeft.Render(render.BugDetail(*st.Bug, st.Activity, bodyWidth, height)), height)
}
