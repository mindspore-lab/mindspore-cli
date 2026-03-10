package model

// Train-specific UI event types.
const (
	TrainModeOpen  EventType = "TrainModeOpen"  // enter train workspace
	TrainModeClose EventType = "TrainModeClose" // exit train workspace
	TrainSetup     EventType = "TrainSetup"     // checklist item update
	TrainConnect   EventType = "TrainConnect"   // host connection update
	TrainReady     EventType = "TrainReady"     // all checks passed
	TrainStarted   EventType = "TrainStarted"   // lane training began
	TrainLogLine   EventType = "TrainLogLine"   // log line for a lane
	TrainMetric    EventType = "TrainMetric"    // lane metrics update
	TrainDone      EventType = "TrainDone"      // lane training completed
	TrainError     EventType = "TrainError"     // training failed (unrecoverable)

	// Phase 2: eval/drift/analysis/fix/rerun
	TrainLaunchFailed  EventType = "TrainLaunchFailed"  // NPU lane crashed at launch
	TrainEvalStarted   EventType = "TrainEvalStarted"   // evaluation begins
	TrainEvalCompleted EventType = "TrainEvalCompleted" // eval finished with results
	TrainDriftDetected EventType = "TrainDriftDetected" // accuracy drift found
	TrainAnalyzing     EventType = "TrainAnalyzing"     // analysis in progress
	TrainAnalysisReady EventType = "TrainAnalysisReady" // diagnosis available
	TrainFixApplied    EventType = "TrainFixApplied"    // patch applied
	TrainRerunStarted  EventType = "TrainRerunStarted"  // NPU rerun begins
	TrainVerified      EventType = "TrainVerified"      // fix verified
)

// TrainEventData carries training-specific fields on Event.
type TrainEventData struct {
	Model      string
	Method     string
	Check      string // check name
	Status     string // "checking", "passed", "failed"
	Detail     string
	Host       string
	Address    string
	Lane       string // "gpu" or "npu"
	Step       int
	TotalSteps int
	Loss       float64
	LR         float64
	Throughput float64
	RunLabel   string

	// Compare
	BaselineAcc  float64
	CandidateAcc float64
	Drift        float64

	// Issue
	IssueType   string // "runtime", "accuracy"
	IssueTitle  string
	IssueDetail string
	Confidence  string
	FixSummary  string
	DiffText    string
}

// TrainPoint is a single data point in a metric time series.
type TrainPoint struct {
	Step  int
	Value float64
}

// ── Lane types ───────────────────────────────────────────────

// TrainLaneID identifies a training lane.
type TrainLaneID string

const (
	LaneGPU TrainLaneID = "gpu"
	LaneNPU TrainLaneID = "npu"
)

// TrainLaneView holds per-lane rendering state.
type TrainLaneView struct {
	ID        TrainLaneID
	Title     string // "Torch / GPU Baseline"
	Framework string // "PyTorch", "MindSpore"
	Device    string // "CUDA", "Ascend"
	Host      string // "gpu-a100-0", "npu-910b-0"
	Role      string // "baseline", "candidate"
	Status    string // "pending", "running", "failed", "completed", "rerunning"
	RunLabel  string // "run1", "run2"

	Metrics    TrainMetricsView
	LossSeries []TrainPoint
	Logs       []string
	MaxLogs    int
}

// AppendLog appends a log line, trimming to MaxLogs.
func (l *TrainLaneView) AppendLog(line string) {
	l.Logs = append(l.Logs, line)
	max := l.MaxLogs
	if max <= 0 {
		max = 200
	}
	if len(l.Logs) > max {
		l.Logs = l.Logs[len(l.Logs)-max:]
	}
}

// ── Compare / Issue types ────────────────────────────────────

// TrainCompareView holds baseline vs candidate comparison.
type TrainCompareView struct {
	BaselineAcc  float64 // Torch/GPU
	CandidateAcc float64 // MindSpore/NPU
	Drift        float64 // negative = worse
	Status       string  // "none", "mismatch", "verified"
}

// TrainIssueView holds the current problem being diagnosed/fixed.
type TrainIssueView struct {
	Type       string // "runtime", "accuracy"
	Title      string
	Detail     string
	Confidence string
	FixSummary string
	DiffText   string
}

// ── Action types ─────────────────────────────────────────────

// TrainAction identifies a button in the action row.
type TrainAction int

const (
	ActionStart TrainAction = iota
	ActionStop
	ActionRetry // label: "Retry NPU"
	ActionClose
	ActionAnalyze  // label varies: "Analyze NPU" / "Analyze Drift"
	ActionApplyFix // label varies: "Apply Runtime Fix" / "Apply Accuracy Fix"
	ActionViewDiff // label: "View Diff"
)

// TrainActionItem is one button in the action row.
type TrainActionItem struct {
	Action   TrainAction
	Label    string
	Disabled bool
}

// ── View state ───────────────────────────────────────────────

// TrainViewState holds the rendering state for the train workspace.
// Managed directly by ui.App (not inside the immutable State tree).
type TrainViewState struct {
	Active bool
	Phase  string // setup, ready, running, launch_failed, evaluating, drift_detected, analyzing, fix_ready, rerunning, verified, completed, failed
	Model  string
	Method string

	// Setup checklist
	Checks []TrainCheckView
	Hosts  []TrainHostView

	// Dual lanes
	GPULane TrainLaneView
	NPULane TrainLaneView

	// Compare state (non-nil after first eval)
	Compare *TrainCompareView

	// Current issue (non-nil during analysis/fix cycle)
	Issue *TrainIssueView

	// Action row
	FocusedAction    int
	ActionRowFocused bool
	Actions          []TrainActionItem
}

// TrainCheckView is one item in the setup checklist.
type TrainCheckView struct {
	Name   string
	Label  string
	Status string // pending, checking, passed, failed
	Detail string
}

// TrainHostView is the SSH status for one host.
type TrainHostView struct {
	Name    string
	Address string
	Status  string // connecting, connected, failed
}

// TrainMetricsView holds live training metrics.
type TrainMetricsView struct {
	Step       int
	TotalSteps int
	Loss       float64
	LR         float64
	Throughput float64
}

// NewTrainViewState returns a TrainViewState with sensible defaults.
func NewTrainViewState(mdl, method string) TrainViewState {
	return TrainViewState{
		Active: true,
		Phase:  "setup",
		Model:  mdl,
		Method: method,
		Checks: []TrainCheckView{
			{Name: "local_repo", Label: "Local repository", Status: "pending"},
			{Name: "train_script", Label: "Training script", Status: "pending"},
			{Name: "base_model", Label: "Base model weights", Status: "pending"},
			{Name: "ssh_gpu", Label: "SSH (GPU host)", Status: "pending"},
			{Name: "ssh_npu", Label: "SSH (NPU host)", Status: "pending"},
			{Name: "remote_workdir", Label: "Remote workspace", Status: "pending"},
			{Name: "runtime_env", Label: "Runtime environment", Status: "pending"},
		},
		GPULane: TrainLaneView{
			ID:        LaneGPU,
			Title:     "Torch / GPU Baseline",
			Framework: "PyTorch",
			Device:    "CUDA",
			Host:      "gpu-a100-0",
			Role:      "baseline",
			Status:    "pending",
			RunLabel:  "run1",
			MaxLogs:   200,
		},
		NPULane: TrainLaneView{
			ID:        LaneNPU,
			Title:     "MindSpore / NPU Candidate",
			Framework: "MindSpore",
			Device:    "Ascend",
			Host:      "npu-910b-0",
			Role:      "candidate",
			Status:    "pending",
			RunLabel:  "run1",
			MaxLogs:   200,
		},
		Actions: actionsForPhase("setup", nil),
	}
}

// UpdatePhase sets the phase and refreshes the available actions.
func (tv *TrainViewState) UpdatePhase(phase string) {
	tv.Phase = phase
	tv.Actions = actionsForPhase(phase, tv.Issue)
	tv.FocusedAction = 0
}

// LaneByID returns a pointer to the lane matching the given ID.
func (tv *TrainViewState) LaneByID(lane string) *TrainLaneView {
	switch TrainLaneID(lane) {
	case LaneGPU:
		return &tv.GPULane
	case LaneNPU:
		return &tv.NPULane
	default:
		return nil
	}
}

// actionsForPhase returns the available actions for a given training phase.
func actionsForPhase(phase string, issue *TrainIssueView) []TrainActionItem {
	switch phase {
	case "setup":
		return []TrainActionItem{
			{Action: ActionClose, Label: "Close"},
		}
	case "ready":
		return []TrainActionItem{
			{Action: ActionStart, Label: "Start Training"},
			{Action: ActionClose, Label: "Close"},
		}
	case "running", "evaluating":
		return []TrainActionItem{
			{Action: ActionStop, Label: "Stop"},
		}
	case "launch_failed":
		return []TrainActionItem{
			{Action: ActionAnalyze, Label: "Analyze NPU"},
			{Action: ActionRetry, Label: "Retry NPU"},
			{Action: ActionClose, Label: "Close"},
		}
	case "analyzing":
		return []TrainActionItem{
			{Action: ActionStop, Label: "Stop"},
		}
	case "fix_ready":
		label := "Apply Fix"
		if issue != nil {
			switch issue.Type {
			case "runtime":
				label = "Apply Runtime Fix"
			case "accuracy":
				label = "Apply Accuracy Fix"
			}
		}
		return []TrainActionItem{
			{Action: ActionApplyFix, Label: label},
			{Action: ActionClose, Label: "Close"},
		}
	case "drift_detected":
		return []TrainActionItem{
			{Action: ActionAnalyze, Label: "Analyze Drift"},
			{Action: ActionClose, Label: "Close"},
		}
	case "rerunning":
		return []TrainActionItem{
			{Action: ActionStop, Label: "Stop"},
		}
	case "verified":
		return []TrainActionItem{
			{Action: ActionViewDiff, Label: "View Diff"},
			{Action: ActionRetry, Label: "Retry NPU"},
			{Action: ActionClose, Label: "Close"},
		}
	case "completed":
		return []TrainActionItem{
			{Action: ActionClose, Label: "Close"},
		}
	case "failed":
		return []TrainActionItem{
			{Action: ActionRetry, Label: "Retry"},
			{Action: ActionClose, Label: "Close"},
		}
	default:
		return []TrainActionItem{
			{Action: ActionClose, Label: "Close"},
		}
	}
}
