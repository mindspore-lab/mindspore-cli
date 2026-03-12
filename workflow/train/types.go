// Package train provides training workflow execution.
// This package must NOT import ui/model — event conversion
// happens in internal/app/train.go.
package train

// EventKind identifies a training workflow event.
type EventKind string

const (
	// Train lane lifecycle events
	EventTrainModeOpened   EventKind = "TrainModeOpened"
	EventTrainSetupStarted EventKind = "TrainSetupStarted"
	EventCheckPassed       EventKind = "CheckPassed"
	EventCheckFailed       EventKind = "CheckFailed"
	EventConnectionStatus  EventKind = "ConnectionStatus"
	EventPlanReady         EventKind = "PlanReady"
	EventReadyToStart      EventKind = "ReadyToStart"
	EventTrainStarted      EventKind = "TrainStarted"
	EventIssueDetected     EventKind = "IssueDetected"
	EventLogLine           EventKind = "LogLine"
	EventMetricUpdate      EventKind = "MetricUpdate"
	EventTrainStopped      EventKind = "TrainStopped"
	EventTrainCompleted    EventKind = "TrainCompleted"
	EventTrainFailed       EventKind = "TrainFailed"

	// Setup check lifecycle
	EventMessage        EventKind = "Message"
	EventCheckStarted   EventKind = "CheckStarted"
	EventHostConnecting EventKind = "HostConnecting"
	EventHostConnected  EventKind = "HostConnected"
	EventHostFailed     EventKind = "HostFailed"

	// Phase 2: evaluation, drift, analysis, fix, rerun
	EventEvalStarted        EventKind = "EvalStarted"
	EventEvalCompleted      EventKind = "EvalCompleted"
	EventDriftDetected      EventKind = "DriftDetected"
	EventAnalysisStarted    EventKind = "AnalysisStarted"
	EventAnalysisReady      EventKind = "AnalysisReady"
	EventActionSuggested    EventKind = "ActionSuggested"
	EventFixApplied         EventKind = "FixApplied"
	EventActionApplied      EventKind = "ActionApplied"
	EventRerunStarted       EventKind = "RerunStarted"
	EventVerificationPassed EventKind = "VerificationPassed"
)

// Event is a single output from the training workflow.
type Event struct {
	Kind       EventKind
	RunID      string
	Message    string
	Check      string // which check item
	Host       string // host name
	Address    string // host address
	Lane       string // "gpu" or "npu" (empty = global)
	Step       int
	TotalSteps int
	Loss       float64
	LR         float64
	Throughput float64
	DelayMs    int // hint: pause before emitting this event

	// Compare fields
	BaselineAcc  float64 // GPU baseline accuracy
	CandidateAcc float64 // NPU candidate accuracy
	Drift        float64 // accuracy drift (negative = worse)
	RunLabel     string  // "run1" or "run2"

	// Issue fields
	IssueType    string // "runtime", "accuracy"
	IssueID      string
	IssueTitle   string // diagnosis issue headline
	IssueDetail  string // diagnosis explanation
	FixSummary   string // short fix description
	DiffText     string // code/config diff
	Confidence   string // diagnosis confidence level
	ActionID     string
	ActionKind   string
	ActionLabel  string
	ActionSource string
	PlanID       string
	RepoPath     string
	RepoSource   string
	ScriptPath   string
	BaseModelRef string
	ConfigPath   string
	EnvKind      string
	Workdir      string

	// Probe-derived fields (Phase 1)
	Scope    string         // "local" or "target"
	Critical bool           // whether this check is a readiness blocker
	Details  map[string]any // extra probe data
}

// Session holds state for a training lane session.
type Session struct {
	ID     string
	Model  string
	Method string
	Phase  string // setup, ready, running, completed, failed, stopped

	// FailAtStep causes the demo backend to crash training at this step.
	// 0 means no failure injection.
	FailAtStep int
}
