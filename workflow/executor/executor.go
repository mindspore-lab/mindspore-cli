// Package executor provides workflow execution backends.
// The orchestrator dispatches to a WorkflowExecutor; this package
// supplies concrete implementations: Stub (real, not yet built) and
// DemoExecutor (scenario playback).
package executor

import (
	"context"

	"github.com/vigo999/ms-cli/agent/orchestrator"
	"github.com/vigo999/ms-cli/agent/planner"
)

// Stub satisfies orchestrator.WorkflowExecutor but always returns
// orchestrator.ErrWorkflowNotImplemented. Replace with a real
// implementation when the workflow engine is built.
type Stub struct{}

// NewStub creates a workflow executor stub.
func NewStub() *Stub {
	return &Stub{}
}

// Execute always returns ErrWorkflowNotImplemented.
func (s *Stub) Execute(_ context.Context, _ orchestrator.RunRequest, _ planner.Plan) ([]orchestrator.RunEvent, error) {
	return nil, orchestrator.ErrWorkflowNotImplemented
}
