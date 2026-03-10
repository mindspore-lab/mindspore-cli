package executor

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/vigo999/ms-cli/agent/orchestrator"
	"github.com/vigo999/ms-cli/agent/planner"
)

// Scenario defines a demo playback scenario loaded from JSON.
type Scenario struct {
	Name     string  `json:"name"`
	Summary  string  `json:"summary"`
	Timeline []Stage `json:"timeline"`
}

// Stage groups related events in the scenario timeline.
type Stage struct {
	Name   string          `json:"stage"`
	Events []ScenarioEvent `json:"events"`
}

// ScenarioEvent is a single event in a scenario timeline.
type ScenarioEvent struct {
	Type       string `json:"type"`
	Message    string `json:"message,omitempty"`
	ToolName   string `json:"tool_name,omitempty"`
	Summary    string `json:"summary,omitempty"`
	CtxUsed    int    `json:"ctx_used,omitempty"`
	TokensUsed int    `json:"tokens_used,omitempty"`
	DelayMs    int    `json:"delay_ms,omitempty"`
}

// DemoExecutor plays back a scenario file as workflow execution events.
// It satisfies orchestrator.WorkflowExecutor.
type DemoExecutor struct {
	scenarioDir string
}

// NewDemo creates a demo executor that loads scenarios from the given directory.
func NewDemo(scenarioDir string) *DemoExecutor {
	return &DemoExecutor{scenarioDir: scenarioDir}
}

// Execute loads the scenario identified by plan.Workflow (or defaults to
// "perf_opt") and returns the timeline as RunEvents with DelayMs hints.
func (d *DemoExecutor) Execute(ctx context.Context, _ orchestrator.RunRequest, plan planner.Plan) ([]orchestrator.RunEvent, error) {
	name := plan.Workflow
	if name == "" {
		name = "perf_opt"
	}

	scenario, err := d.loadScenario(name)
	if err != nil {
		return nil, fmt.Errorf("load scenario %q: %w", name, err)
	}

	var events []orchestrator.RunEvent
	for _, stage := range scenario.Timeline {
		// Emit a stage marker so the UI can show progress.
		events = append(events, orchestrator.RunEvent{
			Type:    orchestrator.EventAgentReply,
			Message: fmt.Sprintf("── %s ──", stage.Name),
			DelayMs: 300,
		})

		for _, se := range stage.Events {
			events = append(events, toRunEvent(se))
		}
	}

	return events, nil
}

func (d *DemoExecutor) loadScenario(name string) (*Scenario, error) {
	// Try exact name, then with .json suffix.
	candidates := []string{
		filepath.Join(d.scenarioDir, name+".json"),
		filepath.Join(d.scenarioDir, name),
	}

	var data []byte
	var lastErr error
	for _, path := range candidates {
		var err error
		data, err = os.ReadFile(path)
		if err == nil {
			break
		}
		lastErr = err
	}
	if data == nil {
		return nil, fmt.Errorf("scenario not found: %w", lastErr)
	}

	var s Scenario
	if err := json.Unmarshal(data, &s); err != nil {
		return nil, fmt.Errorf("parse scenario: %w", err)
	}
	return &s, nil
}

func toRunEvent(se ScenarioEvent) orchestrator.RunEvent {
	now := time.Now()
	return orchestrator.RunEvent{
		Type:       mapEventType(se.Type),
		Message:    se.Message,
		ToolName:   se.ToolName,
		Summary:    se.Summary,
		CtxUsed:    se.CtxUsed,
		TokensUsed: se.TokensUsed,
		Timestamp:  now,
		DelayMs:    se.DelayMs,
	}
}

// mapEventType normalizes scenario event type strings to orchestrator constants
// or passes through as-is for types the orchestrator already knows (e.g. tool events).
func mapEventType(t string) string {
	aliases := map[string]string{
		"thinking":  orchestrator.EventAgentThinking,
		"reply":     orchestrator.EventAgentReply,
		"error":     orchestrator.EventToolError,
		"completed": orchestrator.EventTaskCompleted,
	}
	if mapped, ok := aliases[strings.ToLower(t)]; ok {
		return mapped
	}
	return t // pass through: "ToolRead", "CmdStarted", "CmdOutput", etc.
}
