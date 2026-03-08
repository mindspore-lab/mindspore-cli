package panels

import (
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
	"github.com/vigo999/ms-cli/ui/model"
)

func TestRenderTrainDashboardFitsRequestedSize(t *testing.T) {
	d := model.NewTrainDashboard()
	d.Apply(model.TrainUpdate{
		Kind:  model.TrainUpdateOpen,
		RunID: "run-001",
		Hosts: []string{"gpuA"},
	})
	d.Apply(model.TrainUpdate{
		Kind:          model.TrainUpdateMetric,
		Host:          "gpuA",
		Stage:         "dashboard",
		Step:          12,
		TotalStep:     120,
		Loss:          1.2345,
		Throughput:    512.7,
		GradNorm:      0.82,
		Model:         "qwen2.5-7b",
		HasStep:       true,
		HasTotalStep:  true,
		HasLoss:       true,
		HasThroughput: true,
		HasGradNorm:   true,
		HasModel:      true,
	})
	d.Apply(model.TrainUpdate{
		Kind:    model.TrainUpdateLog,
		Host:    "gpuA",
		Message: "ssh user@gpuA 'mkdir -p /remote/runs/run-001 && nohup python -u train.py > /remote/runs/run-001/log.txt 2>&1 & echo $! > /remote/runs/run-001/train.pid'",
	})

	rendered := RenderTrainDashboard(d, 80, 20, false, nil)

	if got := lipgloss.Width(rendered); got != 80 {
		t.Fatalf("expected rendered width=80, got %d\n%s", got, rendered)
	}
	if got := lipgloss.Height(rendered); got != 20 {
		t.Fatalf("expected rendered height=20, got %d\n%s", got, rendered)
	}
}

func TestRenderRecentLogsWrapsCommandWithoutEllipsis(t *testing.T) {
	d := model.NewTrainDashboard()
	d.Apply(model.TrainUpdate{
		Kind:  model.TrainUpdateOpen,
		RunID: "run-001",
		Hosts: []string{"gpuA"},
	})
	command := "[gpuA] ssh user@gpuA 'mkdir -p /remote/runs/run-001 && nohup python -u train.py > /remote/runs/run-001/log.txt 2>&1 & echo $! > /remote/runs/run-001/train.pid'"
	d.Apply(model.TrainUpdate{
		Kind:    model.TrainUpdateNote,
		Message: command,
	})

	lines := renderRecentLogs(d, 36, 1)
	if len(lines) < 2 {
		t.Fatalf("expected wrapped output across multiple lines, got %#v", lines)
	}

	joined := strings.Join(lines, "\n")
	if strings.Contains(joined, "…") {
		t.Fatalf("expected full command without ellipsis, got %q", joined)
	}
	if !strings.Contains(joined, "ssh user@gpuA") {
		t.Fatalf("expected command prefix preserved, got %q", joined)
	}
	if !strings.Contains(joined, "train.pid") {
		t.Fatalf("expected command suffix preserved, got %q", joined)
	}
	for _, line := range lines {
		if got := lipgloss.Width(line); got > 36 {
			t.Fatalf("expected wrapped line width <= 36, got %d for %q", got, line)
		}
	}
}

func TestRenderTrainDashboardShowsFailedCommand(t *testing.T) {
	d := model.NewTrainDashboard()
	d.Apply(model.TrainUpdate{
		Kind:  model.TrainUpdateOpen,
		RunID: "run-001",
		Hosts: []string{"gpuA"},
	})
	command := "ssh user@gpuA 'cd /remote/code && nohup python -u train.py --model qwen3 > /remote/runs/run-001/log.txt 2>&1 &'"
	d.Apply(model.TrainUpdate{
		Kind:    model.TrainUpdateError,
		Stage:   "launch",
		Host:    "gpuA",
		Command: command,
		Message: "launch host gpuA: exit status 1\nlog.txt: No such file or directory",
	})

	rendered := RenderTrainDashboard(d, 110, 24, false, nil)

	if !strings.Contains(rendered, "Failure") {
		t.Fatalf("expected failure section, got %q", rendered)
	}
	if !strings.Contains(rendered, "command:") {
		t.Fatalf("expected failed command label, got %q", rendered)
	}
	if !strings.Contains(rendered, "nohup python -u") {
		t.Fatalf("expected failed command prefix, got %q", rendered)
	}
	if !strings.Contains(rendered, "train.py --model qwen3 >") {
		t.Fatalf("expected failed command content, got %q", rendered)
	}
	if !strings.Contains(rendered, "log.txt: No such file or directory") {
		t.Fatalf("expected failure reason, got %q", rendered)
	}
}

func TestResolveTrainRightPanelLayoutPrioritizesConnectionLogs(t *testing.T) {
	d := model.NewTrainDashboard()
	d.Apply(model.TrainUpdate{
		Kind:  model.TrainUpdateOpen,
		RunID: "run-001",
		Hosts: []string{"gpuA"},
	})
	d.Apply(model.TrainUpdate{
		Kind:   model.TrainUpdateStage,
		Stage:  "launch",
		Status: string(model.TrainStageRunning),
	})

	connecting := resolveTrainRightPanelLayout(d, 24, false, 0)
	if connecting.LogTitle != "Connection Logs" {
		t.Fatalf("expected connection log title, got %+v", connecting)
	}
	if connecting.ShowLegend {
		t.Fatalf("did not expect legend during connection layout, got %+v", connecting)
	}
	if connecting.LogCount <= 4 {
		t.Fatalf("expected more connection logs during connect, got %+v", connecting)
	}

	d.Apply(model.TrainUpdate{
		Kind:   model.TrainUpdateStage,
		Stage:  "dashboard",
		Status: string(model.TrainStageRunning),
	})
	connected := resolveTrainRightPanelLayout(d, 24, false, 0)
	if connected.LogTitle != "Recent Logs" {
		t.Fatalf("expected recent logs after connection, got %+v", connected)
	}
	if !connected.ShowLegend {
		t.Fatalf("expected legend after connection, got %+v", connected)
	}
	if connected.ChartHeight <= connecting.ChartHeight {
		t.Fatalf("expected expanded chart after connection, connecting=%+v connected=%+v", connecting, connected)
	}
}

func TestRenderTrainDashboardShowsConnectionLogsDuringConnect(t *testing.T) {
	d := model.NewTrainDashboard()
	d.Apply(model.TrainUpdate{
		Kind:  model.TrainUpdateOpen,
		RunID: "run-001",
		Hosts: []string{"gpuA"},
	})
	d.Apply(model.TrainUpdate{
		Kind:   model.TrainUpdateStage,
		Stage:  "sync",
		Status: string(model.TrainStageRunning),
	})
	d.Apply(model.TrainUpdate{
		Kind:    model.TrainUpdateHost,
		Host:    "gpuA",
		Stage:   "sync",
		Status:  string(model.TrainHostRunning),
		Message: "rsync started",
	})

	rendered := RenderTrainDashboard(d, 90, 24, false, nil)
	if !strings.Contains(rendered, "Connection Logs") {
		t.Fatalf("expected connection logs title during connect, got %q", rendered)
	}
	if strings.Contains(rendered, "Legend") {
		t.Fatalf("did not expect legend during connect, got %q", rendered)
	}
}

func TestRenderTrainDashboardSanitizesRecentLogs(t *testing.T) {
	d := model.NewTrainDashboard()
	d.Apply(model.TrainUpdate{
		Kind:  model.TrainUpdateOpen,
		RunID: "run-001",
		Hosts: []string{"gpuA"},
	})
	d.Apply(model.TrainUpdate{
		Kind:    model.TrainUpdateLog,
		Host:    "gpuA",
		Message: "\r\x1b[32m  5/3708 [00:13<2:41:19,  2.61s/it]\x1b[0m",
	})

	rendered := RenderTrainDashboard(d, 90, 24, false, nil)
	if strings.Contains(rendered, "\r") || strings.Contains(rendered, "\x1b") {
		t.Fatalf("expected rendered dashboard to strip terminal control characters, got %q", rendered)
	}
	if !strings.Contains(rendered, "5/3708 [00:13<2:41:19,  2.61s/it]") {
		t.Fatalf("expected sanitized log content to remain visible, got %q", rendered)
	}
}

func TestRenderTrainStageSectionCollapsesAfterSuccess(t *testing.T) {
	d := model.NewTrainDashboard()
	d.Apply(model.TrainUpdate{
		Kind:  model.TrainUpdateOpen,
		RunID: "run-001",
		Hosts: []string{"gpuA"},
	})
	for _, stage := range d.StageOrder {
		d.Apply(model.TrainUpdate{
			Kind:   model.TrainUpdateStage,
			Stage:  stage,
			Status: string(model.TrainStageSuccess),
		})
	}
	d.Apply(model.TrainUpdate{
		Kind:    model.TrainUpdateDone,
		Message: "all host streams finished",
	})

	lines := renderTrainStageSection(d)
	joined := strings.Join(lines, "\n")
	if !strings.Contains(joined, "summary:") {
		t.Fatalf("expected collapsed stage summary, got %q", joined)
	}
	if !strings.Contains(joined, "all 5 stages succeeded") {
		t.Fatalf("expected success summary, got %q", joined)
	}
	if strings.Contains(joined, "Push Code (rsync)") {
		t.Fatalf("did not expect expanded stage list after success, got %q", joined)
	}
}

func TestRenderTrainStageSectionCollapsesWhenDashboardStarts(t *testing.T) {
	d := model.NewTrainDashboard()
	d.Apply(model.TrainUpdate{
		Kind:  model.TrainUpdateOpen,
		RunID: "run-001",
		Hosts: []string{"gpuA"},
	})
	for _, stage := range []string{"sync", "launch", "master", "stream"} {
		d.Apply(model.TrainUpdate{
			Kind:   model.TrainUpdateStage,
			Stage:  stage,
			Status: string(model.TrainStageSuccess),
		})
	}
	d.Apply(model.TrainUpdate{
		Kind:   model.TrainUpdateStage,
		Stage:  "dashboard",
		Status: string(model.TrainStageRunning),
	})

	lines := renderTrainStageSection(d)
	joined := strings.Join(lines, "\n")
	if !strings.Contains(joined, "summary:") {
		t.Fatalf("expected collapsed stage summary when dashboard starts, got %q", joined)
	}
	if !strings.Contains(joined, "4/4 setup stages completed") {
		t.Fatalf("expected setup-stage summary when dashboard starts, got %q", joined)
	}
	if !strings.Contains(joined, "realtime dashboard active") {
		t.Fatalf("expected realtime dashboard state when dashboard starts, got %q", joined)
	}
	if strings.Contains(joined, "Push Code (rsync)") {
		t.Fatalf("did not expect expanded stage list after dashboard starts, got %q", joined)
	}
}

func TestRenderTrainPanelsMoveChatToLeftLowerSide(t *testing.T) {
	d := model.NewTrainDashboard()
	d.Apply(model.TrainUpdate{
		Kind:  model.TrainUpdateOpen,
		RunID: "run-001",
		Hosts: []string{"gpuA"},
	})
	d.Apply(model.TrainUpdate{
		Kind:   model.TrainUpdateStage,
		Stage:  "dashboard",
		Status: string(model.TrainStageRunning),
	})
	d.Apply(model.TrainUpdate{
		Kind:    model.TrainUpdateLog,
		Host:    "gpuA",
		Message: "tail connected",
	})

	lowerPanel := &TrainLowerPanel{
		Title: "Analysis Chat",
		Body:  "assistant\ninput",
	}
	left := renderTrainLeftPanel(d, 40, 24, false, lowerPanel)
	right := renderTrainRightPanel(d, 49, 24, false)

	if !strings.Contains(left, "Analysis Chat") {
		t.Fatalf("expected embedded chat on left panel, got %q", left)
	}
	if strings.Contains(right, "Analysis Chat") {
		t.Fatalf("did not expect embedded chat on right panel, got %q", right)
	}
	if !strings.Contains(right, "Recent Logs") {
		t.Fatalf("expected right panel to keep logs, got %q", right)
	}
}

func TestRenderLossChartUsesTotalStepAxisAndContinuousLine(t *testing.T) {
	d := model.NewTrainDashboard()
	d.Apply(model.TrainUpdate{
		Kind:  model.TrainUpdateOpen,
		RunID: "run-001",
		Hosts: []string{"gpuA"},
	})

	for _, update := range []model.TrainUpdate{
		{
			Kind:         model.TrainUpdateMetric,
			Host:         "gpuA",
			Stage:        "dashboard",
			Step:         12,
			TotalStep:    120,
			Loss:         3.60,
			HasStep:      true,
			HasTotalStep: true,
			HasLoss:      true,
		},
		{
			Kind:         model.TrainUpdateMetric,
			Host:         "gpuA",
			Stage:        "dashboard",
			Step:         60,
			TotalStep:    120,
			Loss:         1.95,
			HasStep:      true,
			HasTotalStep: true,
			HasLoss:      true,
		},
		{
			Kind:         model.TrainUpdateMetric,
			Host:         "gpuA",
			Stage:        "dashboard",
			Step:         120,
			TotalStep:    120,
			Loss:         0.48,
			HasStep:      true,
			HasTotalStep: true,
			HasLoss:      true,
		},
	} {
		d.Apply(update)
	}

	chart, xRange, yRange := renderLossChart(d, 48, 12)

	if xRange != "x(total step): 0 -> 120" {
		t.Fatalf("expected total-step axis label, got %q", xRange)
	}
	if yRange != "y(loss): 0 -> 4" {
		t.Fatalf("expected zero-based y-axis label, got %q", yRange)
	}
	if strings.Contains(chart, "◆") || strings.Contains(chart, "╲") || strings.Contains(chart, "╱") {
		t.Fatalf("expected braille curve rendering instead of ascii line fragments, got %q", chart)
	}
	if !containsBrailleRune(chart) {
		t.Fatalf("expected braille curve glyphs in chart, got %q", chart)
	}
	if !strings.Contains(chart, "●") {
		t.Fatalf("expected highlighted endpoint marker, got %q", chart)
	}
	if !strings.Contains(chart, "0") || !strings.Contains(chart, "120") || !strings.Contains(chart, "60") {
		t.Fatalf("expected x-axis tick labels in chart, got %q", chart)
	}
	if !strings.Contains(chart, "4") {
		t.Fatalf("expected y-axis tick labels in chart, got %q", chart)
	}
	if !strings.Contains(chart, "└") || !strings.Contains(chart, "┬") {
		t.Fatalf("expected explicit axis baseline and tick marks, got %q", chart)
	}
}

func TestRenderLossChartUsesDynamicObservedXRangeInsteadOfTotalStepHint(t *testing.T) {
	d := model.NewTrainDashboard()
	d.Apply(model.TrainUpdate{
		Kind:  model.TrainUpdateOpen,
		RunID: "run-001",
		Hosts: []string{"npuA"},
	})

	for _, update := range []model.TrainUpdate{
		{
			Kind:         model.TrainUpdateMetric,
			Host:         "npuA",
			Stage:        "dashboard",
			Step:         4,
			TotalStep:    3708,
			Loss:         3.52,
			HasStep:      true,
			HasTotalStep: true,
			HasLoss:      true,
		},
		{
			Kind:         model.TrainUpdateMetric,
			Host:         "npuA",
			Stage:        "dashboard",
			Step:         8,
			TotalStep:    3708,
			Loss:         3.24,
			HasStep:      true,
			HasTotalStep: true,
			HasLoss:      true,
		},
	} {
		d.Apply(update)
	}

	_, xRange, _ := renderLossChart(d, 48, 12)
	if xRange != "x(total step): 0 -> 8" {
		t.Fatalf("expected dynamic observed x-range instead of total-step hint, got %q", xRange)
	}
}

func containsBrailleRune(s string) bool {
	for _, r := range s {
		if r >= '\u2801' && r <= '\u28ff' {
			return true
		}
	}
	return false
}
