package train

import (
	"context"

	itrain "github.com/mindspore-lab/mindspore-cli/internal/train"
)

// Backend provides setup and run behavior for a training lane.
// Phase 1 uses the demo backend; future phases will add real SSH execution.
type Backend interface {
	Setup(ctx context.Context, req itrain.Request, sink func(Event)) error
	Run(ctx context.Context, session *Session, sink func(Event)) error
}

// Controller is the train lane facade / entrypoint for the app layer.
type Controller struct {
	backend Backend
}

// NewController creates a controller with the given backend.
func NewController(backend Backend) *Controller {
	return &Controller{backend: backend}
}

// NewDemoController creates a controller wired to the demo backend.
func NewDemoController() *Controller {
	return NewController(&DemoBackend{})
}

// Open initializes a training session from the request.
func (c *Controller) Open(_ context.Context, req itrain.Request) *Session {
	id := req.RunID
	if id == "" {
		id = "primary"
	}
	s := &Session{
		ID:     id,
		Model:  req.Model,
		Method: req.Method,
		Phase:  "setup",
	}
	// Read demo failure injection config.
	if failStep, ok := req.Target.Config["demo_fail_at_step"].(int); ok && failStep > 0 {
		s.FailAtStep = failStep
	}
	if fixed, ok := req.Target.Config["demo_drift_fixed"].(bool); ok && fixed {
		s.DriftFixed = true
	}
	if fixed, ok := req.Target.Config["demo_perf_fixed"].(bool); ok && fixed {
		s.PerfFixed = true
	}
	if applied, ok := req.Target.Config["demo_trick_applied"].(bool); ok && applied {
		s.TrickApplied = true
	}
	return s
}

// Setup runs the setup phase: local probes, then target probes.
func (c *Controller) Setup(ctx context.Context, req itrain.Request, sink func(Event)) error {
	return RunSetupSequence(ctx, req, c.backend, sink)
}

// Start begins the training run phase.
func (c *Controller) Start(ctx context.Context, session *Session, sink func(Event)) error {
	session.Phase = "running"
	return RunTraining(ctx, session, c.backend, sink)
}

// Stop cancels the current training run.
// The caller is expected to cancel the context passed to Start.
func (c *Controller) Stop(_ context.Context, session *Session) {
	session.Phase = "stopped"
}

// Retry re-runs training after a failure or fix.
func (c *Controller) Retry(ctx context.Context, session *Session, sink func(Event)) error {
	session.Phase = "running"
	return RunTraining(ctx, session, c.backend, sink)
}

// Backend returns the controller's backend for direct phase 2 operations.
func (c *Controller) Backend() Backend {
	return c.backend
}
