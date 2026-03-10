// Package orchestrator dispatches tasks to the appropriate executor
// based on the planner's decision.
package orchestrator

import (
	"context"
	"errors"
	"fmt"

	"github.com/vigo999/ms-cli/agent/planner"
)

// ErrWorkflowNotImplemented is a sentinel error that workflow executor
// stubs return to signal the orchestrator should fall back to agent mode.
var ErrWorkflowNotImplemented = errors.New("workflow executor not yet implemented")

// AgentExecutor handles exploratory, open-ended tasks via the ReAct loop.
type AgentExecutor interface {
	Execute(ctx context.Context, req RunRequest) ([]RunEvent, error)
}

// WorkflowExecutor handles structured, multi-step tasks.
type WorkflowExecutor interface {
	Execute(ctx context.Context, req RunRequest, plan planner.Plan) ([]RunEvent, error)
}

// Config holds orchestrator settings.
type Config struct {
	AvailableTools []string
}

// Orchestrator receives a request, calls the planner, and dispatches
// to the appropriate executor based on the plan's mode.
type Orchestrator struct {
	config       Config
	planner      *planner.Planner
	agentExec    AgentExecutor
	workflowExec WorkflowExecutor
	callback     PlanCallback
}

// New creates an Orchestrator.
func New(cfg Config, agentExec AgentExecutor, p *planner.Planner, wfExec WorkflowExecutor) *Orchestrator {
	return &Orchestrator{
		config:       cfg,
		planner:      p,
		agentExec:    agentExec,
		workflowExec: wfExec,
		callback:     NoOpCallback{},
	}
}

// SetCallback sets the plan lifecycle callback.
func (o *Orchestrator) SetCallback(cb PlanCallback) {
	if cb == nil {
		o.callback = NoOpCallback{}
		return
	}
	o.callback = cb
}

// Run analyzes the request via planner and dispatches to the right executor.
// If planner is nil or fails, falls back to agent mode.
func (o *Orchestrator) Run(ctx context.Context, req RunRequest) ([]RunEvent, error) {
	// No planner → agent mode directly
	if o.planner == nil {
		events := []RunEvent{
			NewRunEvent(EventAgentReply, "Planner unavailable, running in agent mode"),
		}
		agentEvents, agentErr := o.agentExec.Execute(ctx, req)
		events = append(events, agentEvents...)
		return events, agentErr
	}

	plan, err := o.planner.Plan(ctx, req.Description, o.config.AvailableTools)
	if err != nil {
		events := []RunEvent{
			NewRunEvent(EventAgentReply,
				fmt.Sprintf("Planner failed, falling back to agent mode: %v", err)),
		}
		agentEvents, agentErr := o.agentExec.Execute(ctx, req)
		events = append(events, agentEvents...)
		return events, agentErr
	}

	return o.dispatch(ctx, req, plan)
}

// dispatch routes to the appropriate executor based on plan mode.
func (o *Orchestrator) dispatch(ctx context.Context, req RunRequest, plan planner.Plan) ([]RunEvent, error) {
	switch plan.Mode {
	case planner.ModeWorkflow:
		return o.runWorkflow(ctx, req, plan)
	default:
		return o.agentExec.Execute(ctx, req)
	}
}

// RunWorkflow executes a pre-built plan directly, bypassing the planner.
// Useful when the caller already knows the plan (e.g. demo mode, retry).
func (o *Orchestrator) RunWorkflow(req RunRequest, plan planner.Plan) ([]RunEvent, error) {
	return o.runWorkflow(context.Background(), req, plan)
}

// runWorkflow executes a workflow plan with callback notifications.
func (o *Orchestrator) runWorkflow(ctx context.Context, req RunRequest, plan planner.Plan) ([]RunEvent, error) {
	var events []RunEvent

	summary := fmt.Sprintf("Plan: %s", plan.Goal)
	if plan.Workflow != "" {
		summary = fmt.Sprintf("Plan: %s (workflow: %s)", plan.Goal, plan.Workflow)
	} else if len(plan.Steps) > 0 {
		summary = fmt.Sprintf("Plan: %s (%d steps)", plan.Goal, len(plan.Steps))
	}
	events = append(events, NewRunEvent(EventAgentReply, summary))

	// Notify callback
	if err := o.callback.OnPlanCreated(plan); err != nil {
		events = append(events, NewRunEvent(EventTaskFailed, fmt.Sprintf("Plan rejected: %v", err)))
		return events, err
	}

	if err := o.callback.OnPlanApproved(plan); err != nil {
		return events, err
	}

	// Workflow executor is required for workflow mode
	if o.workflowExec == nil {
		return events, fmt.Errorf("workflow mode requires a WorkflowExecutor")
	}

	wfEvents, err := o.workflowExec.Execute(ctx, req, plan)
	if errors.Is(err, ErrWorkflowNotImplemented) {
		// Executor stub → fall back to agent for the whole task,
		// preserving plan events so the UI knows planning happened.
		agentEvents, agentErr := o.agentExec.Execute(ctx, req)
		events = append(events, agentEvents...)
		return events, agentErr
	}
	events = append(events, wfEvents...)
	return events, err
}
