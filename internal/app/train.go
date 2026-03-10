package app

import (
	"context"
	"fmt"
	"strings"

	"github.com/vigo999/ms-cli/internal/train"
	"github.com/vigo999/ms-cli/ui/model"
	wtrain "github.com/vigo999/ms-cli/workflow/train"
)

type trainSnapshot struct {
	mode      bool
	phase     string
	req       *train.Request
	issueType string
}

// cmdTrain handles the /train command.
func (a *Application) cmdTrain(args []string) {
	if len(args) < 2 {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "Usage: /train <model> <method>\nExample: /train qwen3 lora",
		}
		return
	}

	req := train.Request{
		Model:  args[0],
		Method: args[1],
	}

	ctx, runID := a.beginTrainMode(req)
	a.EventCh <- model.Event{
		Type: model.TrainModeOpen,
		Train: &model.TrainEventData{
			Model:  req.Model,
			Method: req.Method,
		},
	}

	go a.runTrainSetup(ctx, runID, req)
}

// runTrainSetup runs the training setup workflow and converts events to UI events.
func (a *Application) runTrainSetup(ctx context.Context, runID uint64, req train.Request) {
	sink := func(ev wtrain.Event) {
		a.convertAndEmitTrainEvent(runID, ev)
	}

	err := wtrain.RunSetup(ctx, req.Model, req.Method, sink)
	if err != nil {
		a.EventCh <- model.Event{
			Type:    model.TrainError,
			Message: fmt.Sprintf("Setup failed: %v", err),
		}
	}
}

// runConcurrentTraining starts both lanes (GPU healthy, NPU crashes).
func (a *Application) runConcurrentTraining(ctx context.Context, runID uint64, req train.Request) {
	sink := func(ev wtrain.Event) {
		a.convertAndEmitTrainEvent(runID, ev)
	}

	err := wtrain.RunConcurrentTraining(ctx, req.Model, req.Method, sink)
	if err != nil && ctx.Err() == nil {
		a.EventCh <- model.Event{
			Type:    model.TrainError,
			Message: fmt.Sprintf("Training failed: %v", err),
		}
	}
}

// runAnalysis dispatches to the correct analysis function based on trainIssueType.
func (a *Application) runAnalysis(ctx context.Context, runID uint64, req train.Request, issueType string) {
	sink := func(ev wtrain.Event) {
		a.convertAndEmitTrainEvent(runID, ev)
	}

	var err error
	switch issueType {
	case "runtime":
		err = wtrain.RunNPUAnalysis(ctx, req.Model, req.Method, sink)
	case "accuracy":
		err = wtrain.RunDriftAnalysis(ctx, req.Model, req.Method, sink)
	default:
		err = wtrain.RunNPUAnalysis(ctx, req.Model, req.Method, sink)
	}

	if err != nil && ctx.Err() == nil {
		a.EventCh <- model.Event{
			Type:    model.TrainError,
			Message: fmt.Sprintf("Analysis failed: %v", err),
		}
	}
}

// runApplyFix dispatches to the correct fix+run function based on trainIssueType.
func (a *Application) runApplyFix(ctx context.Context, runID uint64, req train.Request, issueType string) {
	sink := func(ev wtrain.Event) {
		a.convertAndEmitTrainEvent(runID, ev)
	}

	var err error
	switch issueType {
	case "runtime":
		err = wtrain.RunNPUFixAndResume(ctx, req.Model, req.Method, sink)
	case "accuracy":
		err = wtrain.RunDriftFixAndRerun(ctx, req.Model, req.Method, sink)
	default:
		err = wtrain.RunNPUFixAndResume(ctx, req.Model, req.Method, sink)
	}

	if err != nil && ctx.Err() == nil {
		a.EventCh <- model.Event{
			Type:    model.TrainError,
			Message: fmt.Sprintf("Fix failed: %v", err),
		}
	}
}

// handleTrainInput routes user input during train mode.
// Commands are gated by trainPhase to enforce the correct state machine.
func (a *Application) handleTrainInput(input string) {
	lower := strings.ToLower(strings.TrimSpace(input))
	snapshot := a.getTrainSnapshot()

	// Stop and exit are always allowed.
	switch {
	case lower == "stop":
		a.stopTraining()
		return
	case lower == "exit" || lower == "back":
		a.exitTrainMode()
		return
	}

	// Gate all other commands on the current phase.
	switch {
	case lower == "start" || lower == "start training":
		if snapshot.phase != "ready" {
			a.rejectCommand("start", "setup must complete first")
			return
		}
		a.startTraining()

	case lower == "retry" || lower == "retry npu":
		if snapshot.phase != "launch_failed" && snapshot.phase != "verified" {
			a.rejectCommand("retry", "nothing to retry")
			return
		}
		a.retryNPU()

	case lower == "analyze" || lower == "analyze npu" || lower == "analyze drift":
		if snapshot.phase != "launch_failed" && snapshot.phase != "drift_detected" {
			a.rejectCommand("analyze", "no failure or drift to investigate")
			return
		}
		a.analyzeTraining()

	case lower == "apply fix" || lower == "apply runtime fix" || lower == "apply accuracy fix":
		if snapshot.phase != "fix_ready" {
			a.rejectCommand("apply fix", "run analysis first")
			return
		}
		a.applyFix()

	case lower == "view diff":
		a.viewDiff()

	default:
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: fmt.Sprintf("Train mode commands: start, stop, analyze, apply fix, retry, view diff, exit (got: %s)", input),
		}
	}
}

// rejectCommand sends a user-visible message explaining why a command was blocked.
func (a *Application) rejectCommand(cmd, reason string) {
	phase := a.getTrainPhase()
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: fmt.Sprintf("Cannot %s right now: %s. (phase: %s)", cmd, reason, phase),
	}
}

func (a *Application) startTraining() {
	ctx, runID, req, _, ok := a.beginTrainTask("running")
	if !ok {
		return
	}
	go a.runConcurrentTraining(ctx, runID, req)
}

func (a *Application) stopTraining() {
	a.stopTrainTask("stopped")
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: "Training stopped.",
	}
}

func (a *Application) retryNPU() {
	ctx, runID, req, issueType, ok := a.beginTrainTask("running")
	if !ok {
		return
	}
	go a.runApplyFix(ctx, runID, req, issueType)
}

func (a *Application) viewDiff() {
	snapshot := a.getTrainSnapshot()
	if snapshot.req == nil {
		return
	}
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: "Diff is shown in the Issue section of the left panel.",
	}
}

func (a *Application) analyzeTraining() {
	ctx, runID, req, issueType, ok := a.beginTrainTask("analyzing")
	if !ok {
		return
	}
	go a.runAnalysis(ctx, runID, req, issueType)
}

func (a *Application) applyFix() {
	ctx, runID, req, issueType, ok := a.beginTrainTask("applying")
	if !ok {
		return
	}
	go a.runApplyFix(ctx, runID, req, issueType)
}

func (a *Application) exitTrainMode() {
	a.resetTrainState()
	a.EventCh <- model.Event{Type: model.TrainModeClose}
}

// convertAndEmitTrainEvent maps workflow train events to UI events.
// It also updates trainPhase to keep Application-layer gating in sync.
func (a *Application) convertAndEmitTrainEvent(runID uint64, ev wtrain.Event) {
	if !a.isCurrentTrainRun(runID) {
		return
	}

	switch ev.Kind {
	case wtrain.EventMessage:
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: ev.Message,
		}

	case wtrain.EventCheckStarted:
		a.EventCh <- model.Event{
			Type: model.TrainSetup,
			Train: &model.TrainEventData{
				Check:  ev.Check,
				Status: "checking",
			},
		}

	case wtrain.EventCheckPassed:
		a.EventCh <- model.Event{
			Type: model.TrainSetup,
			Train: &model.TrainEventData{
				Check:  ev.Check,
				Status: "passed",
				Detail: ev.Message,
			},
		}

	case wtrain.EventCheckFailed:
		a.EventCh <- model.Event{
			Type: model.TrainSetup,
			Train: &model.TrainEventData{
				Check:  ev.Check,
				Status: "failed",
				Detail: ev.Message,
			},
		}

	case wtrain.EventHostConnecting:
		a.EventCh <- model.Event{
			Type: model.TrainConnect,
			Train: &model.TrainEventData{
				Host:    ev.Host,
				Address: ev.Address,
				Status:  "connecting",
			},
		}

	case wtrain.EventHostConnected:
		a.EventCh <- model.Event{
			Type: model.TrainConnect,
			Train: &model.TrainEventData{
				Host:    ev.Host,
				Address: ev.Address,
				Status:  "connected",
			},
		}

	case wtrain.EventHostFailed:
		a.EventCh <- model.Event{
			Type: model.TrainConnect,
			Train: &model.TrainEventData{
				Host:   ev.Host,
				Status: "failed",
			},
		}

	case wtrain.EventReadyToStart:
		a.setTrainPhase("ready")
		a.EventCh <- model.Event{
			Type:    model.TrainReady,
			Message: ev.Message,
		}

	case wtrain.EventTrainStarted:
		a.setTrainPhase("running")
		a.EventCh <- model.Event{
			Type:    model.TrainStarted,
			Message: ev.Message,
			Train: &model.TrainEventData{
				Lane:     ev.Lane,
				RunLabel: ev.RunLabel,
			},
		}

	case wtrain.EventLogLine:
		a.EventCh <- model.Event{
			Type:    model.TrainLogLine,
			Message: ev.Message,
			Train: &model.TrainEventData{
				Lane: ev.Lane,
			},
		}

	case wtrain.EventMetricUpdate:
		a.EventCh <- model.Event{
			Type: model.TrainMetric,
			Train: &model.TrainEventData{
				Lane:       ev.Lane,
				Step:       ev.Step,
				TotalSteps: ev.TotalSteps,
				Loss:       ev.Loss,
				LR:         ev.LR,
				Throughput: ev.Throughput,
				RunLabel:   ev.RunLabel,
			},
		}

	case wtrain.EventTrainCompleted:
		// When GPU completes and NPU already failed, transition to launch_failed.
		snapshot := a.getTrainSnapshot()
		if ev.Lane == "gpu" && snapshot.issueType == "runtime" && snapshot.phase == "running" {
			a.setTrainPhase("launch_failed")
		}
		a.EventCh <- model.Event{
			Type:    model.TrainDone,
			Message: ev.Message,
			Train: &model.TrainEventData{
				Lane:     ev.Lane,
				RunLabel: ev.RunLabel,
			},
		}

	case wtrain.EventTrainFailed:
		a.setTrainPhase("failed")
		a.EventCh <- model.Event{
			Type:    model.TrainError,
			Message: ev.Message,
		}

	case wtrain.EventLaunchFailed:
		a.setTrainIssueType(ev.IssueType)
		// Don't set trainPhase to launch_failed yet — GPU may still be running.
		// Phase transitions to launch_failed in EventTrainCompleted when GPU finishes.
		a.EventCh <- model.Event{
			Type:    model.TrainLaunchFailed,
			Message: ev.Message,
			Train: &model.TrainEventData{
				Lane:        ev.Lane,
				IssueType:   ev.IssueType,
				IssueTitle:  ev.IssueTitle,
				IssueDetail: ev.IssueDetail,
			},
		}

	// ── Phase 2 events ──

	case wtrain.EventEvalStarted:
		a.setTrainPhase("evaluating")
		a.EventCh <- model.Event{
			Type:    model.TrainEvalStarted,
			Message: ev.Message,
		}

	case wtrain.EventEvalCompleted:
		a.EventCh <- model.Event{
			Type:    model.TrainEvalCompleted,
			Message: ev.Message,
			Train: &model.TrainEventData{
				BaselineAcc:  ev.BaselineAcc,
				CandidateAcc: ev.CandidateAcc,
				Drift:        ev.Drift,
				RunLabel:     ev.RunLabel,
			},
		}

	case wtrain.EventDriftDetected:
		a.setTrainIssueType("accuracy")
		a.setTrainPhase("drift_detected")
		a.EventCh <- model.Event{
			Type:    model.TrainDriftDetected,
			Message: ev.Message,
			Train: &model.TrainEventData{
				BaselineAcc:  ev.BaselineAcc,
				CandidateAcc: ev.CandidateAcc,
				Drift:        ev.Drift,
			},
		}

	case wtrain.EventAnalysisStarted:
		a.setTrainPhase("analyzing")
		a.EventCh <- model.Event{
			Type:    model.TrainAnalyzing,
			Message: ev.Message,
		}

	case wtrain.EventAnalysisReady:
		a.setTrainPhase("fix_ready")
		a.EventCh <- model.Event{
			Type:    model.TrainAnalysisReady,
			Message: ev.Message,
			Train: &model.TrainEventData{
				IssueType:   ev.IssueType,
				IssueTitle:  ev.IssueTitle,
				IssueDetail: ev.IssueDetail,
				Confidence:  ev.Confidence,
				FixSummary:  ev.FixSummary,
				DiffText:    ev.DiffText,
			},
		}

	case wtrain.EventFixApplied:
		a.EventCh <- model.Event{
			Type:    model.TrainFixApplied,
			Message: ev.Message,
			Train: &model.TrainEventData{
				Lane:       ev.Lane,
				FixSummary: ev.FixSummary,
				DiffText:   ev.DiffText,
			},
		}

	case wtrain.EventRerunStarted:
		a.setTrainPhase("rerunning")
		a.EventCh <- model.Event{
			Type:    model.TrainRerunStarted,
			Message: ev.Message,
			Train: &model.TrainEventData{
				Lane:     ev.Lane,
				RunLabel: ev.RunLabel,
			},
		}

	case wtrain.EventVerificationPassed:
		a.setTrainPhase("verified")
		a.EventCh <- model.Event{
			Type:    model.TrainVerified,
			Message: ev.Message,
			Train: &model.TrainEventData{
				BaselineAcc:  ev.BaselineAcc,
				CandidateAcc: ev.CandidateAcc,
				Drift:        ev.Drift,
			},
		}
	}
}

func (a *Application) isTrainMode() bool {
	a.trainMu.RLock()
	defer a.trainMu.RUnlock()
	return a.trainMode
}

func (a *Application) getTrainPhase() string {
	a.trainMu.RLock()
	defer a.trainMu.RUnlock()
	return a.trainPhase
}

func (a *Application) getTrainSnapshot() trainSnapshot {
	a.trainMu.RLock()
	defer a.trainMu.RUnlock()

	var reqCopy *train.Request
	if a.trainReq != nil {
		req := *a.trainReq
		reqCopy = &req
	}

	return trainSnapshot{
		mode:      a.trainMode,
		phase:     a.trainPhase,
		req:       reqCopy,
		issueType: a.trainIssueType,
	}
}

func (a *Application) setTrainPhase(phase string) {
	a.trainMu.Lock()
	defer a.trainMu.Unlock()
	a.trainPhase = phase
}

func (a *Application) setTrainIssueType(issueType string) {
	a.trainMu.Lock()
	defer a.trainMu.Unlock()
	a.trainIssueType = issueType
}

func (a *Application) beginTrainMode(req train.Request) (context.Context, uint64) {
	ctx, cancel := context.WithCancel(context.Background())

	a.trainMu.Lock()
	oldCancel := a.trainCancel
	a.trainRunID++
	runID := a.trainRunID
	a.trainMode = true
	a.trainPhase = "setup"
	a.trainReq = &req
	a.trainIssueType = ""
	a.trainCancel = cancel
	a.trainMu.Unlock()

	if oldCancel != nil {
		oldCancel()
	}

	return ctx, runID
}

func (a *Application) beginTrainTask(phase string) (context.Context, uint64, train.Request, string, bool) {
	ctx, cancel := context.WithCancel(context.Background())

	a.trainMu.Lock()
	if a.trainReq == nil {
		a.trainMu.Unlock()
		cancel()
		return nil, 0, train.Request{}, "", false
	}

	oldCancel := a.trainCancel
	a.trainRunID++
	runID := a.trainRunID
	req := *a.trainReq
	issueType := a.trainIssueType
	a.trainPhase = phase
	a.trainCancel = cancel
	a.trainMu.Unlock()

	if oldCancel != nil {
		oldCancel()
	}

	return ctx, runID, req, issueType, true
}

func (a *Application) stopTrainTask(phase string) {
	a.trainMu.Lock()
	oldCancel := a.trainCancel
	a.trainRunID++
	a.trainPhase = phase
	a.trainCancel = nil
	a.trainMu.Unlock()

	if oldCancel != nil {
		oldCancel()
	}
}

func (a *Application) resetTrainState() {
	a.trainMu.Lock()
	oldCancel := a.trainCancel
	a.trainRunID++
	a.trainMode = false
	a.trainPhase = ""
	a.trainReq = nil
	a.trainIssueType = ""
	a.trainCancel = nil
	a.trainMu.Unlock()

	if oldCancel != nil {
		oldCancel()
	}
}

func (a *Application) isCurrentTrainRun(runID uint64) bool {
	a.trainMu.RLock()
	defer a.trainMu.RUnlock()
	return a.trainMode && a.trainRunID == runID
}
