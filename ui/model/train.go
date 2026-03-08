package model

import (
	"fmt"
	"regexp"
	"strings"
	"time"
	"unicode"
)

// TrainUpdateKind indicates the kind of training-dashboard event.
type TrainUpdateKind string

const (
	TrainUpdateOpen    TrainUpdateKind = "open"
	TrainUpdateStage   TrainUpdateKind = "stage"
	TrainUpdateHost    TrainUpdateKind = "host"
	TrainUpdateMetric  TrainUpdateKind = "metric"
	TrainUpdateLog     TrainUpdateKind = "log"
	TrainUpdateNote    TrainUpdateKind = "note"
	TrainUpdateDone    TrainUpdateKind = "done"
	TrainUpdateError   TrainUpdateKind = "error"
	TrainUpdateStopped TrainUpdateKind = "stopped"
	TrainUpdateClose   TrainUpdateKind = "close"
)

var trainDashboardANSIEscapeRE = regexp.MustCompile(`\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))`)

// TrainStageStatus is the state of a workflow stage.
type TrainStageStatus string

const (
	TrainStagePending TrainStageStatus = "pending"
	TrainStageRunning TrainStageStatus = "running"
	TrainStageSuccess TrainStageStatus = "success"
	TrainStageFailed  TrainStageStatus = "failed"
)

// TrainHostStatus is the state of a host within the workflow.
type TrainHostStatus string

const (
	TrainHostIdle    TrainHostStatus = "idle"
	TrainHostRunning TrainHostStatus = "running"
	TrainHostSuccess TrainHostStatus = "success"
	TrainHostFailed  TrainHostStatus = "failed"
)

// TrainUpdate carries a dashboard update from workflow/runtime.
type TrainUpdate struct {
	Kind    TrainUpdateKind
	RunID   string
	Hosts   []string
	Host    string
	Stage   string
	Status  string
	Command string
	Message string
	LogPath string

	Step       int
	TotalStep  int
	Loss       float64
	Throughput float64
	GradNorm   float64
	Model      string

	HasStep       bool
	HasTotalStep  bool
	HasLoss       bool
	HasThroughput bool
	HasGradNorm   bool
	HasModel      bool

	At time.Time
}

// TrainPoint is one loss-series point.
type TrainPoint struct {
	Step int
	Loss float64
}

// TrainHostMetrics stores latest host metrics rendered in the dashboard.
type TrainHostMetrics struct {
	Name       string
	Status     TrainHostStatus
	Stage      string
	Model      string
	Step       int
	TotalStep  int
	Loss       float64
	Throughput float64
	GradNorm   float64
	LogPath    string

	LastCommand string
	LastMessage string
	UpdatedAt   time.Time
}

// TrainDashboard is the state for /train dashboard mode.
type TrainDashboard struct {
	Active        bool
	RunID         string
	Status        string // running/success/failed/stopped
	Error         string
	FailedStage   string
	FailedHost    string
	FailedCommand string
	StartedAt     time.Time
	UpdatedAt     time.Time
	FinishedAt    time.Time
	CurrentStage  string

	StageOrder  []string
	StageLabels map[string]string
	StageStatus map[string]TrainStageStatus

	HostOrder []string
	Hosts     map[string]*TrainHostMetrics
	Series    map[string][]TrainPoint

	RecentLogs []string
	MaxLogs    int
}

// NewTrainDashboard returns the default train-dashboard state.
func NewTrainDashboard() TrainDashboard {
	stageOrder := []string{"sync", "launch", "master", "stream", "dashboard"}
	stageLabels := map[string]string{
		"sync":      "Push Code (rsync)",
		"launch":    "Launch Train (nohup)",
		"master":    "Create SSH Master",
		"stream":    "Stream Logs (tail -F)",
		"dashboard": "Realtime Parse + TUI",
	}
	stageStatus := make(map[string]TrainStageStatus, len(stageOrder))
	for _, stage := range stageOrder {
		stageStatus[stage] = TrainStagePending
	}

	return TrainDashboard{
		StageOrder:  stageOrder,
		StageLabels: stageLabels,
		StageStatus: stageStatus,
		Hosts:       make(map[string]*TrainHostMetrics),
		Series:      make(map[string][]TrainPoint),
		RecentLogs:  []string{},
		MaxLogs:     120,
	}
}

// Apply mutates dashboard state from one incoming update.
func (d *TrainDashboard) Apply(u TrainUpdate) {
	ts := u.At
	if ts.IsZero() {
		ts = time.Now()
	}

	switch u.Kind {
	case TrainUpdateOpen:
		next := NewTrainDashboard()
		next.Active = true
		next.RunID = u.RunID
		next.Status = "running"
		next.StartedAt = ts
		next.UpdatedAt = ts
		*d = next
		for _, host := range u.Hosts {
			d.ensureHost(host)
		}
		d.appendLog("workflow started")
		return

	case TrainUpdateClose:
		d.Active = false
		d.UpdatedAt = ts
		return
	}

	if !d.Active {
		return
	}
	d.UpdatedAt = ts

	switch u.Kind {
	case TrainUpdateStage:
		if u.Stage != "" {
			d.CurrentStage = u.Stage
		}
		if u.Stage != "" && u.Status != "" {
			d.StageStatus[u.Stage] = toStageStatus(u.Status)
		} else if u.Stage != "" && d.StageStatus[u.Stage] == "" {
			d.StageStatus[u.Stage] = TrainStageRunning
		}
		if u.Message != "" {
			d.appendLog(u.Message)
		}
		if u.Command != "" {
			d.appendLog(u.Command)
		}
		if u.Status == string(TrainStageFailed) {
			d.Status = "failed"
			d.Error = u.Message
			d.FailedStage = u.Stage
			if u.Host != "" {
				d.FailedHost = u.Host
			}
			if u.Command != "" {
				d.FailedCommand = u.Command
			}
			d.FinishedAt = ts
		}

	case TrainUpdateHost:
		host := d.ensureHost(u.Host)
		if u.Status != "" {
			host.Status = toHostStatus(u.Status)
		}
		if u.Stage != "" {
			host.Stage = u.Stage
		}
		if u.Command != "" {
			host.LastCommand = u.Command
			d.appendLog(fmt.Sprintf("[%s] %s", host.Name, u.Command))
		}
		if u.Message != "" {
			host.LastMessage = u.Message
			d.appendLog(fmt.Sprintf("[%s] %s", host.Name, u.Message))
		}
		if u.LogPath != "" {
			host.LogPath = u.LogPath
		}
		host.UpdatedAt = ts

	case TrainUpdateMetric:
		host := d.ensureHost(u.Host)
		host.Status = TrainHostRunning
		if u.Stage != "" {
			host.Stage = u.Stage
		}
		if u.HasModel {
			host.Model = u.Model
		}
		if u.HasStep {
			host.Step = u.Step
		}
		if u.HasTotalStep {
			host.TotalStep = u.TotalStep
		}
		if u.HasLoss {
			host.Loss = u.Loss
		}
		if u.HasThroughput {
			host.Throughput = u.Throughput
		}
		if u.HasGradNorm {
			host.GradNorm = u.GradNorm
		}
		host.UpdatedAt = ts
		if u.HasLoss {
			d.appendSeriesPoint(host.Name, u, host)
		}

	case TrainUpdateLog:
		host := d.ensureHost(u.Host)
		host.LastMessage = u.Message
		host.UpdatedAt = ts
		d.appendLog(fmt.Sprintf("[%s] %s", host.Name, u.Message))

	case TrainUpdateNote:
		if u.Message != "" {
			d.appendLog(u.Message)
		}

	case TrainUpdateDone:
		d.Status = "success"
		d.FinishedAt = ts
		if u.Message != "" {
			d.appendLog(u.Message)
		}
		if d.CurrentStage != "" {
			d.StageStatus[d.CurrentStage] = TrainStageSuccess
		}

	case TrainUpdateStopped:
		d.Status = "stopped"
		d.FinishedAt = ts
		if u.Message != "" {
			d.appendLog(u.Message)
		}

	case TrainUpdateError:
		d.Status = "failed"
		d.Error = u.Message
		d.FailedStage = u.Stage
		d.FailedHost = u.Host
		d.FailedCommand = u.Command
		d.FinishedAt = ts
		if u.Stage != "" {
			d.StageStatus[u.Stage] = TrainStageFailed
		}
		if u.Host != "" {
			host := d.ensureHost(u.Host)
			host.Status = TrainHostFailed
			host.Stage = u.Stage
			if u.Command != "" {
				host.LastCommand = u.Command
			}
			host.LastMessage = u.Message
			host.UpdatedAt = ts
		}
		if u.Message != "" {
			d.appendLog(u.Message)
		}
	}
}

func (d *TrainDashboard) ensureHost(name string) *TrainHostMetrics {
	host := name
	if host == "" {
		host = "unknown"
	}
	if _, ok := d.Hosts[host]; !ok {
		d.HostOrder = append(d.HostOrder, host)
		d.Hosts[host] = &TrainHostMetrics{
			Name:   host,
			Status: TrainHostIdle,
		}
	}
	return d.Hosts[host]
}

func (d *TrainDashboard) appendSeriesPoint(host string, u TrainUpdate, metrics *TrainHostMetrics) {
	series := d.Series[host]
	step := metrics.Step
	if u.HasStep {
		step = u.Step
	}
	if step <= 0 {
		if len(series) == 0 {
			step = 1
		} else {
			step = series[len(series)-1].Step + 1
		}
	}
	loss := metrics.Loss
	if u.HasLoss {
		loss = u.Loss
	}

	if len(series) > 0 && series[len(series)-1].Step == step {
		series[len(series)-1].Loss = loss
	} else {
		series = append(series, TrainPoint{Step: step, Loss: loss})
	}

	if len(series) > 4000 {
		series = series[len(series)-4000:]
	}
	d.Series[host] = series
}

func (d *TrainDashboard) appendLog(line string) {
	line = sanitizeTrainDashboardLog(line)
	if line == "" {
		return
	}
	d.RecentLogs = append(d.RecentLogs, line)
	maxLogs := d.MaxLogs
	if maxLogs <= 0 {
		maxLogs = 120
	}
	if len(d.RecentLogs) > maxLogs {
		d.RecentLogs = d.RecentLogs[len(d.RecentLogs)-maxLogs:]
	}
}

func sanitizeTrainDashboardLog(line string) string {
	line = strings.ReplaceAll(line, "\r\n", "\n")
	if strings.Contains(line, "\r") {
		parts := strings.Split(line, "\r")
		for i := len(parts) - 1; i >= 0; i-- {
			part := strings.TrimSpace(parts[i])
			if part != "" {
				line = part
				break
			}
		}
	}

	line = trainDashboardANSIEscapeRE.ReplaceAllString(line, "")

	var b strings.Builder
	lastSpace := false
	for _, r := range line {
		switch {
		case r == '\n':
			b.WriteRune('\n')
			lastSpace = false
		case r == '\r' || r == '\t':
			if !lastSpace {
				b.WriteByte(' ')
				lastSpace = true
			}
		case unicode.IsControl(r):
			if !lastSpace {
				b.WriteByte(' ')
				lastSpace = true
			}
		default:
			b.WriteRune(r)
			lastSpace = unicode.IsSpace(r)
		}
	}

	return strings.TrimSpace(b.String())
}

func toStageStatus(status string) TrainStageStatus {
	switch status {
	case string(TrainStageRunning):
		return TrainStageRunning
	case string(TrainStageSuccess):
		return TrainStageSuccess
	case string(TrainStageFailed):
		return TrainStageFailed
	default:
		return TrainStagePending
	}
}

func toHostStatus(status string) TrainHostStatus {
	switch status {
	case string(TrainHostRunning):
		return TrainHostRunning
	case string(TrainHostSuccess):
		return TrainHostSuccess
	case string(TrainHostFailed):
		return TrainHostFailed
	default:
		return TrainHostIdle
	}
}
