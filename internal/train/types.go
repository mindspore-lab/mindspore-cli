// Package train defines domain types for training sessions.
package train

// Phase represents the current training lifecycle phase.
type Phase string

const (
	PhaseSetup         Phase = "setup"
	PhaseReady         Phase = "ready"
	PhaseRunning       Phase = "running"
	PhaseEvaluating    Phase = "evaluating"
	PhaseDriftDetected Phase = "drift_detected"
	PhaseAnalyzing     Phase = "analyzing"
	PhaseFixReady      Phase = "fix_ready"
	PhaseRerunning     Phase = "rerunning"
	PhaseVerified      Phase = "verified"
	PhaseStopped       Phase = "stopped"
	PhaseFailed        Phase = "failed"
	PhaseCompleted     Phase = "completed"
)

// CheckStatus is the state of a single setup check.
type CheckStatus string

const (
	CheckPending CheckStatus = "pending"
	CheckRunning CheckStatus = "checking"
	CheckPassed  CheckStatus = "passed"
	CheckFailed  CheckStatus = "failed"
)

// Request captures what the user asked for with /train.
type Request struct {
	Model  string
	Method string // e.g. "lora", "full", "qlora"
}

// CheckItem is one entry in the setup checklist.
type CheckItem struct {
	Name   string      // e.g. "local_repo", "train_script", "ssh"
	Label  string      // human-readable label
	Status CheckStatus
	Detail string // extra info shown after check completes
}

// HostStatus tracks SSH connection state for one host.
type HostStatus struct {
	Name    string
	Address string
	Status  CheckStatus
}

// EvalResult holds evaluation accuracy results.
type EvalResult struct {
	BaselineAcc float64 // Torch/GPU baseline
	NPUAcc      float64 // MindSpore/NPU result
	Drift       float64 // difference (negative = worse)
}

// Diagnosis describes the suspected root cause.
type Diagnosis struct {
	IssueTitle  string // headline
	IssueDetail string // explanation
	Confidence  string // "high", "medium", "low"
}

// FixSummary describes the applied fix.
type FixSummary struct {
	Description string
	DiffText    string // code/config delta
}
