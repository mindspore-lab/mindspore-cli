package orchestrator

import "time"

// RunRequest is the orchestrator's input — what the caller wants executed.
type RunRequest struct {
	ID          string
	Description string
}

// RunEvent is an event produced during orchestrated execution.
type RunEvent struct {
	Type       string
	Message    string
	ToolName   string
	Summary    string
	CtxUsed    int
	CtxMax     int
	TokensUsed int
	Timestamp  time.Time
	DelayMs    int // optional: delay before emitting this event (used by demo/scenario playback)
}

// NewRunEvent creates a new run event.
func NewRunEvent(eventType, message string) RunEvent {
	return RunEvent{
		Type:      eventType,
		Message:   message,
		Timestamp: time.Now(),
	}
}

// Event type constants used by the orchestrator.
// These are orchestrator-level semantics — the engine adapter maps
// engine-specific events into these.
const (
	EventTaskStarted   = "TaskStarted"
	EventTaskCompleted = "TaskCompleted"
	EventTaskFailed    = "TaskFailed"
	EventAgentThinking = "AgentThinking"
	EventAgentReply    = "AgentReply"
	EventLLMResponse   = "LLMResponse"
	EventToolError     = "ToolError"
)
