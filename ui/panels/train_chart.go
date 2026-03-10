package panels

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/vigo999/ms-cli/ui/model"
)

// RenderLaneChart renders an ASCII loss curve for a single lane.
func RenderLaneChart(series []model.TrainPoint, title string, pointColor, lineColor string, width, height int) string {
	if width < 10 || height < 4 {
		return ""
	}

	pointStyle := lipgloss.NewStyle().Foreground(lipgloss.Color(pointColor)).Bold(true)
	connStyle := lipgloss.NewStyle().Foreground(lipgloss.Color(lineColor))
	axisStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("244"))
	emptyStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("240")).Italic(true)
	labelStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("244")).Italic(true)

	var sections []string

	// Title bar
	titleStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(lipgloss.Color("252")).
		Background(lipgloss.Color("236")).
		Padding(0, 1)

	titleText := title
	if titleText == "" {
		titleText = " Loss Curve"
	}
	if len(series) > 0 {
		latest := series[len(series)-1]
		titleText += fmt.Sprintf("  (%.4f @ %d)", latest.Value, latest.Step)
	}
	sections = append(sections, " "+titleStyle.Render(titleText))

	if len(series) < 2 {
		sections = append(sections, "")
		sections = append(sections, emptyStyle.Render("  Waiting for training data..."))
		for len(sections) < height {
			sections = append(sections, "")
		}
		return strings.Join(sections[:height], "\n")
	}

	// Chart area dimensions
	chartHeight := height - 4
	chartWidth := width - 10
	if chartHeight < 2 {
		chartHeight = 2
	}
	if chartWidth < 5 {
		chartWidth = 5
	}

	// Find min/max
	minVal, maxVal := series[0].Value, series[0].Value
	for _, p := range series {
		if p.Value < minVal {
			minVal = p.Value
		}
		if p.Value > maxVal {
			maxVal = p.Value
		}
	}

	valRange := maxVal - minVal
	if valRange < 0.001 {
		valRange = 0.001
		minVal -= 0.0005
		maxVal += 0.0005
	}
	padding := valRange * 0.05
	minVal -= padding
	maxVal += padding
	valRange = maxVal - minVal

	columns := resampleSeries(series, chartWidth)

	colRows := make([]int, chartWidth)
	for i, v := range columns {
		normalized := (v - minVal) / valRange
		rowIdx := chartHeight - 1 - int(normalized*float64(chartHeight-1))
		if rowIdx < 0 {
			rowIdx = 0
		}
		if rowIdx >= chartHeight {
			rowIdx = chartHeight - 1
		}
		colRows[i] = rowIdx
	}

	sections = append(sections, "")

	for row := 0; row < chartHeight; row++ {
		rowVal := maxVal - (float64(row)/float64(chartHeight-1))*valRange
		label := fmt.Sprintf(" %7.4f", rowVal)
		if rowVal >= 10.0 {
			label = fmt.Sprintf(" %7.2f", rowVal)
		} else if rowVal >= 1.0 {
			label = fmt.Sprintf(" %7.3f", rowVal)
		}

		var line strings.Builder
		line.WriteString(axisStyle.Render(label + " ┤"))

		for col := 0; col < chartWidth; col++ {
			targetRow := colRows[col]
			if row == targetRow {
				line.WriteString(pointStyle.Render("●"))
			} else if col > 0 {
				prevRow := colRows[col-1]
				minR, maxR := prevRow, targetRow
				if minR > maxR {
					minR, maxR = maxR, minR
				}
				if row > minR && row < maxR {
					line.WriteString(connStyle.Render("│"))
				} else {
					line.WriteString(" ")
				}
			} else {
				line.WriteString(" ")
			}
		}

		sections = append(sections, line.String())
	}

	axisLine := fmt.Sprintf("          └%s", strings.Repeat("─", chartWidth))
	sections = append(sections, axisStyle.Render(axisLine))

	if len(series) > 0 {
		first := series[0].Step
		last := series[len(series)-1].Step
		stepLabel := fmt.Sprintf("           %d", first)
		gap := chartWidth - len(fmt.Sprintf("%d", first)) - len(fmt.Sprintf("%d", last))
		if gap > 0 {
			stepLabel += strings.Repeat(" ", gap) + fmt.Sprintf("%d", last)
		}
		sections = append(sections, labelStyle.Render(stepLabel))
	}

	if len(sections) > height {
		sections = sections[:height]
	}
	for len(sections) < height {
		sections = append(sections, "")
	}

	return strings.Join(sections, "\n")
}

// resampleSeries downsamples or upsamples the series to n columns.
func resampleSeries(series []model.TrainPoint, n int) []float64 {
	if len(series) == 0 {
		return make([]float64, n)
	}
	if len(series) == 1 {
		result := make([]float64, n)
		for i := range result {
			result[i] = series[0].Value
		}
		return result
	}

	result := make([]float64, n)
	for i := 0; i < n; i++ {
		t := float64(i) / float64(n-1) * float64(len(series)-1)
		idx := int(t)
		frac := t - float64(idx)

		if idx >= len(series)-1 {
			result[i] = series[len(series)-1].Value
		} else {
			result[i] = series[idx].Value*(1-frac) + series[idx+1].Value*frac
		}
	}
	return result
}
