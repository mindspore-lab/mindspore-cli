// Package train provides training workflow execution.
// This package must NOT import ui/model — event conversion
// happens in internal/app/train.go.
package train

// EventKind identifies a training workflow event.
type EventKind string

const (
	EventMessage        EventKind = "Message"        // agent-like chat message
	EventCheckStarted   EventKind = "CheckStarted"   // a setup check begins
	EventCheckPassed    EventKind = "CheckPassed"    // a setup check passed
	EventCheckFailed    EventKind = "CheckFailed"    // a setup check failed
	EventHostConnecting EventKind = "HostConnecting" // SSH connecting
	EventHostConnected  EventKind = "HostConnected"  // SSH connected
	EventHostFailed     EventKind = "HostFailed"     // SSH failed
	EventReadyToStart   EventKind = "ReadyToStart"   // all checks passed
	EventTrainStarted   EventKind = "TrainStarted"   // training kicked off
	EventLogLine        EventKind = "LogLine"        // one log line from remote
	EventMetricUpdate   EventKind = "MetricUpdate"   // step/loss/throughput snapshot
	EventTrainCompleted EventKind = "TrainCompleted" // training finished
	EventTrainFailed    EventKind = "TrainFailed"    // training crashed (generic)
	EventLaunchFailed   EventKind = "LaunchFailed"   // training failed at launch (analyzable runtime issue)

	// Phase 2: evaluation, drift, analysis, fix, rerun
	EventEvalStarted        EventKind = "EvalStarted"        // evaluation begins
	EventEvalCompleted      EventKind = "EvalCompleted"      // evaluation finished
	EventDriftDetected      EventKind = "DriftDetected"      // accuracy drift found
	EventAnalysisStarted    EventKind = "AnalysisStarted"    // root cause analysis begins
	EventAnalysisReady      EventKind = "AnalysisReady"      // diagnosis available
	EventFixApplied         EventKind = "FixApplied"         // patch applied
	EventRerunStarted       EventKind = "RerunStarted"       // second training run begins
	EventVerificationPassed EventKind = "VerificationPassed" // rerun improved result
)

// Event is a single output from the training workflow.
type Event struct {
	Kind       EventKind
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
	IssueType   string // "runtime", "accuracy"
	IssueTitle  string // diagnosis issue headline
	IssueDetail string // diagnosis explanation
	FixSummary  string // short fix description
	DiffText    string // code/config diff
	Confidence  string // diagnosis confidence level
}
