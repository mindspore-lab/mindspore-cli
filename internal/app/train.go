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

// trainTextAliases maps free-form sentences to canonical commands.
// Used for demo: the user can type natural phrases instead of exact commands.
var trainTextAliases = map[string]string{
	// start / rerun
	"run it":              "start",
	"run the training":    "start",
	"rerun":               "start",
	"rerun training":      "start",
	"rerun experiment":    "start",
	"go":                  "start",
	"launch":              "start",
	"start it":            "start",
	"start it up":         "start",
	"run again":           "start",
	"run it again":        "start",
	"begin":               "start",
	"begin training":      "start",
	"let's go":            "start",
	"let's run it":        "start",
	"kick it off":         "start",
	"proceed":             "start",
	"execute":             "start",
	"run experiment":      "start",
	"start the run":       "start",
	// analyze / diagnose
	"analysis":            "analyze",
	"what went wrong":     "analyze",
	"check the error":     "analyze",
	"investigate":         "analyze",
	"why did it fail":     "analyze",
	"check failure":       "analyze",
	"what happened":       "analyze",
	"debug":               "analyze",
	"debug it":            "analyze",
	"analyze it":          "analyze",
	"look into it":        "analyze",
	"check it":            "analyze",
	"find the problem":    "analyze",
	"what's the issue":    "analyze",
	"what's wrong":        "analyze",
	"show me the error":   "analyze",
	"explain the failure": "analyze",
	"figure it out":       "analyze",
	// diagnose (explicit)
	"diagnose it":         "diagnose",
	"find the issue":      "diagnose",
	"root cause":          "diagnose",
	"diagnose the issue":  "diagnose",
	// retry
	"try again":           "retry",
	"one more time":       "retry",
	"retry it":            "retry",
	// apply fix (confirmation words like "yes"/"ok"/"do it" are
	// handled in the UI layer — they fire the current focused button)
	"fix it":              "apply fix",
	"apply the fix":       "apply fix",
	"patch it":            "apply fix",
	"apply":               "apply fix",
	"apply patch":         "apply fix",
	"apply the change":    "apply fix",
	"make the change":     "apply fix",
	// analyze perf
	"check performance":   "analyze perf",
	"profile it":          "analyze perf",
	"why is it slow":      "analyze perf",
	"check perf":          "analyze perf",
	"perf analysis":       "analyze perf",
	"optimize":            "analyze perf",
	"optimize it":         "analyze perf",
	"make it faster":      "analyze perf",
	"speed it up":         "analyze perf",
	"check throughput":    "analyze perf",
	"check speed":         "analyze perf",
	"profile":             "analyze perf",
	"tune performance":    "analyze perf",
	"bottleneck":          "analyze perf",
	"why slow":            "analyze perf",
	// algo-feature
	"add mhc":             "add algo-feature mhc",
	"try mhc":             "add algo-feature mhc",
	"enable mhc":          "add algo-feature mhc",
	"apply mhc":           "add algo-feature mhc",
	"use mhc":             "add algo-feature mhc",
	"mhc":                 "add algo-feature mhc",
	"add algo-feature":    "add algo-feature mhc",
	"add feature":         "add algo-feature mhc",
	"add a technique":     "add algo-feature mhc",
	"try a new technique": "add algo-feature mhc",
	"improve accuracy":    "add algo-feature mhc",
	"boost accuracy":      "add algo-feature mhc",
	// perf-feature
	"add fa2":             "add perf-feature fa2",
	"flash attention":     "add perf-feature fa2",
	"fused adam":          "add perf-feature fused-adam",
	"gradient checkpoint": "add perf-feature gradient-ckpt",
	"bf16":                "add perf-feature bf16-mixed",
	"graph mode":          "add perf-feature graph-mod",
	"graph mod":           "add perf-feature graph-mod",
	"comm overlap":        "add perf-feature comm-overlap",
	"zero offload":        "add perf-feature zero-offload",
	"sequence parallel":   "add perf-feature sequence-parallel",
	"selective recompute": "add perf-feature selective-recompute",
	"add perf-feature":    "add perf-feature fa2",
	"boost perf":          "add perf-feature fa2",
	"optimize perf":       "add perf-feature fa2",
	// stop
	"cancel":              "stop",
	"abort":               "stop",
	"stop it":             "stop",
	"halt":                "stop",
	"kill it":             "stop",
	"stop training":       "stop",
}

type bootstrapRunState struct {
	Applied         map[string]bool
	PendingActionID string
}

// cmdTrain handles the /train command.
func (a *Application) cmdTrain(args []string) {
	rawInput := strings.Join(args, " ")
	workspaceRunID := a.nextWorkspaceRunID()

	modelName := ""
	method := ""
	if len(args) > 0 {
		modelName = args[0]
	}
	if len(args) > 1 {
		method = args[1]
	}

	req := train.Request{
		RunID:  workspaceRunID,
		Model:  modelName,
		Method: method,
		Target: train.TrainTarget{
			Provider: train.ProviderOnPrem,
			Backend:  train.BackendSSHHost,
			Name:     "torch-npu-910b-0",
			Config: map[string]any{
				"address":            "8.9.72.194:22",
				"env_manager":        "docker",
				"demo_ssh_flaky":     true,
				"demo_libs_missing":  true,
				"demo_fail_at_step":  50,
			},
		},
	}

	var ctx context.Context
	var runID uint64
	if a.isTrainMode() {
		ctx, runID = a.appendTrainRun(req)
	} else {
		ctx, runID = a.beginTrainMode(req)
	}

	// Initialize controller
	a.trainController = wtrain.NewDemoController()

	a.EventCh <- model.Event{
		Type: model.TrainModeOpen,
		Train: &model.TrainEventData{
			RunID:    workspaceRunID,
			RawInput: rawInput,
			Model:    req.Model,
			Method:   req.Method,
		},
	}

	go a.runTrainSetup(ctx, runID, req)
}

// runTrainSetup runs the training setup workflow via the controller.
func (a *Application) runTrainSetup(ctx context.Context, runID uint64, req train.Request) {
	sink := func(ev wtrain.Event) {
		a.convertAndEmitTrainEvent(runID, ev)
	}

	err := a.trainController.Setup(ctx, req, sink)
	if err != nil && ctx.Err() == nil {
		a.EventCh <- model.Event{
			Type:    model.TrainError,
			Message: fmt.Sprintf("setup failed: %v", err),
		}
	}
}

// runTrainRun runs the training phase via the controller.
func (a *Application) runTrainRun(ctx context.Context, runID uint64, req train.Request) {
	sink := func(ev wtrain.Event) {
		a.convertAndEmitTrainEvent(runID, ev)
	}

	session := a.trainController.Open(ctx, req)
	err := a.trainController.Start(ctx, session, sink)
	if err != nil && ctx.Err() == nil {
		a.EventCh <- model.Event{
			Type:    model.TrainError,
			Message: fmt.Sprintf("training failed: %v", err),
		}
	}
}

// runLegacySetup runs the legacy Phase 2 setup flow (for backward compatibility).
func (a *Application) runLegacySetup(ctx context.Context, runID uint64, req train.Request) {
	sink := func(ev wtrain.Event) {
		a.convertAndEmitTrainEvent(runID, ev)
	}
	err := wtrain.RunSetup(ctx, req.Model, req.Method, sink)
	if err != nil && ctx.Err() == nil {
		a.EventCh <- model.Event{
			Type:    model.TrainError,
			Message: fmt.Sprintf("setup failed: %v", err),
		}
	}
}

// runConcurrentTraining starts both lanes (GPU healthy, NPU crashes).
// Legacy Phase 2 flow.
func (a *Application) runConcurrentTraining(ctx context.Context, runID uint64, req train.Request) {
	sink := func(ev wtrain.Event) {
		a.convertAndEmitTrainEvent(runID, ev)
	}

	err := wtrain.RunConcurrentTraining(ctx, req.Model, req.Method, sink)
	if err != nil && ctx.Err() == nil {
		a.EventCh <- model.Event{
			Type:    model.TrainError,
			Message: fmt.Sprintf("training failed: %v", err),
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
	case "failure":
		err = wtrain.AnalyzeFailure(ctx, req.Model, req.Method, sink)
	case "runtime":
		err = wtrain.RunNPUAnalysis(ctx, req.Model, req.Method, sink)
	case "accuracy":
		if a.trainController != nil {
			err = wtrain.AnalyzeSingleLaneDrift(ctx, req.Model, req.Method, sink)
		} else {
			err = wtrain.RunDriftAnalysis(ctx, req.Model, req.Method, sink)
		}
	case "performance":
		if a.trainController != nil {
			err = wtrain.AnalyzeSingleLanePerf(ctx, req.Model, req.Method, sink)
		} else {
			err = wtrain.RunPerformanceAnalysis(ctx, req.Model, req.Method, sink)
		}
	default:
		err = wtrain.AnalyzeFailure(ctx, req.Model, req.Method, sink)
	}

	if err != nil && ctx.Err() == nil {
		a.EventCh <- model.Event{
			Type:    model.TrainError,
			Message: fmt.Sprintf("analysis failed: %v", err),
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
	case "failure":
		err = wtrain.ApplyFailureFix(ctx, req.Model, req.Method, sink)
	case "runtime":
		err = wtrain.RunNPUFixAndResume(ctx, req.Model, req.Method, sink)
	case "accuracy":
		if a.trainController != nil {
			err = wtrain.ApplySingleLaneDriftFix(ctx, req.Model, req.Method, sink)
		} else {
			err = wtrain.RunDriftFixAndRerun(ctx, req.Model, req.Method, sink)
		}
	case "performance":
		if a.trainController != nil {
			err = wtrain.ApplySingleLanePerfFix(ctx, req.Model, req.Method, sink)
		} else {
			err = wtrain.RunPerformanceFixAndRerun(ctx, req.Model, req.Method, sink)
		}
	default:
		err = wtrain.ApplyFailureFix(ctx, req.Model, req.Method, sink)
	}

	if err != nil && ctx.Err() == nil {
		a.EventCh <- model.Event{
			Type:    model.TrainError,
			Message: fmt.Sprintf("fix failed: %v", err),
		}
		return
	}

	// Fix succeeded — clear failure injection so rerun won't fail again.
	a.trainMu.Lock()
	if a.trainReq != nil {
		if a.trainReq.Target.Config != nil {
			delete(a.trainReq.Target.Config, "demo_fail_at_step")
		}
		if issueType == "accuracy" || issueType == "performance" {
			if a.trainReq.Target.Config == nil {
				a.trainReq.Target.Config = map[string]any{}
			}
			if issueType == "accuracy" {
				a.trainReq.Target.Config["demo_drift_fixed"] = true
			} else {
				a.trainReq.Target.Config["demo_perf_fixed"] = true
			}
		}
	}
	if r, ok := a.trainReqs[a.trainCurrentRun]; ok {
		if r.Target.Config != nil {
			delete(r.Target.Config, "demo_fail_at_step")
		}
		if issueType == "accuracy" || issueType == "performance" {
			if r.Target.Config == nil {
				r.Target.Config = map[string]any{}
			}
			if issueType == "accuracy" {
				r.Target.Config["demo_drift_fixed"] = true
			} else {
				r.Target.Config["demo_perf_fixed"] = true
			}
		}
		a.trainReqs[a.trainCurrentRun] = r
	}
	a.trainMu.Unlock()
}

// handleTrainInput routes user input during train mode.
// Commands are gated by trainPhase to enforce the correct state machine.
func (a *Application) handleTrainInput(input string) {
	lower := strings.ToLower(strings.TrimSpace(input))

	// Resolve free-form aliases to canonical commands.
	if canonical, ok := trainTextAliases[lower]; ok {
		lower = canonical
	}

	snapshot := a.getTrainSnapshot()
	_, pendingBootstrapAction, hasBootstrapAction := a.currentBootstrapAction()

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
		if snapshot.phase != "ready" && snapshot.phase != "completed" {
			a.rejectCommand("start", "setup must complete first")
			return
		}
		a.startTraining()

	case lower == "retry" || lower == "retry npu":
		if snapshot.phase != "failed" {
			a.rejectCommand("retry", "nothing to retry")
			return
		}
		a.startTraining()

	case lower == "analyze" || lower == "analyze npu" || lower == "analyze drift" || lower == "diagnose":
		if snapshot.phase != "failed" && snapshot.phase != "drift_detected" {
			a.rejectCommand("analyze", "no failure or drift to investigate")
			return
		}
		a.analyzeTraining()

	case lower == "analyze perf" || lower == "analyze performance":
		if snapshot.phase != "running" && snapshot.phase != "completed" {
			a.rejectCommand("analyze performance", "performance analysis needs a finished or active run")
			return
		}
		a.setTrainIssueType("performance")
		a.analyzeTraining()

	case lower == "apply fix" || lower == "apply runtime fix" || lower == "apply accuracy fix":
		if hasBootstrapAction {
			a.applyBootstrapAction(pendingBootstrapAction)
			return
		}
		if snapshot.phase != "ready" {
			a.rejectCommand("apply fix", "run analysis first")
			return
		}
		a.applyFix()

	case hasBootstrapAction && lower == pendingBootstrapAction:
		a.applyBootstrapAction(pendingBootstrapAction)

	case strings.HasPrefix(lower, "add algo-feature"):
		if snapshot.phase != "ready" && snapshot.phase != "completed" {
			a.rejectCommand("add algo-feature", "workspace must be stable before algo-feature iteration")
			return
		}
		a.addAlgoFeature(strings.TrimSpace(strings.TrimPrefix(lower, "add algo-feature")))

	case strings.HasPrefix(lower, "add perf-feature"):
		if snapshot.phase != "ready" && snapshot.phase != "completed" {
			a.rejectCommand("add perf-feature", "workspace must be stable before perf-feature iteration")
			return
		}
		a.addPerfFeature(strings.TrimSpace(strings.TrimPrefix(lower, "add perf-feature")))

	case lower == "view diff":
		a.viewDiff()

	default:
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: fmt.Sprintf("train mode commands: start, stop, analyze, apply fix, retry, view diff, exit (got: %s)", input),
		}
	}
}

// rejectCommand sends a user-visible message explaining why a command was blocked.
func (a *Application) rejectCommand(cmd, reason string) {
	phase := a.getTrainPhase()
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: fmt.Sprintf("cannot %s right now: %s. (phase: %s)", cmd, reason, phase),
	}
}

func (a *Application) startTraining() {
	ctx, runID, req, _, ok := a.beginTrainTask("running")
	if !ok {
		return
	}
	// Use controller for Phase 1, fall back to legacy dual-lane for Phase 2
	if a.trainController != nil {
		go a.runTrainRun(ctx, runID, req)
	} else {
		go a.runConcurrentTraining(ctx, runID, req)
	}
}

func (a *Application) stopTraining() {
	a.stopTrainTask("stopped")
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: "training stopped.",
	}
}

func (a *Application) viewDiff() {
	snapshot := a.getTrainSnapshot()
	if snapshot.req == nil {
		return
	}
	a.EventCh <- model.Event{
		Type:    model.AgentReply,
		Message: "diff is shown in the issue section of the left panel.",
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
	ctx, runID, req, issueType, ok := a.beginTrainTask("fixing")
	if !ok {
		return
	}
	go a.runApplyFix(ctx, runID, req, issueType)
}

func (a *Application) applyBootstrapAction(actionID string) {
	ctx, runID, req, _, ok := a.beginTrainTask("fixing")
	if !ok {
		return
	}
	go func() {
		err := wtrain.RunBootstrapApply(ctx, req, actionID, func(ev wtrain.Event) {
			a.convertAndEmitTrainEvent(runID, ev)
		})
		if err != nil && ctx.Err() == nil {
			a.EventCh <- model.Event{
				Type:    model.TrainError,
				Message: fmt.Sprintf("bootstrap apply failed: %v", err),
				Train:   &model.TrainEventData{RunID: req.RunID},
			}
			return
		}
		a.markBootstrapActionApplied(req.RunID, actionID)
		recheckCtx, recheckRunID := a.appendTrainRun(req)
		a.setTrainPhase("setup")
		if a.isBootstrapRequest(req) {
			applied := a.bootstrapApplied(req.RunID)
			err = wtrain.RunBootstrapRecheck(recheckCtx, req, applied, func(ev wtrain.Event) {
				a.convertAndEmitTrainEvent(recheckRunID, ev)
			})
			if err != nil && recheckCtx.Err() == nil {
				a.EventCh <- model.Event{
					Type:    model.TrainError,
					Message: fmt.Sprintf("bootstrap setup failed: %v", err),
					Train:   &model.TrainEventData{RunID: req.RunID},
				}
			}
			return
		}
		a.runTrainSetup(recheckCtx, recheckRunID, req)
	}()
}

func (a *Application) addAlgoFeature(feature string) {
	ctx, runID, req, _, ok := a.beginTrainTask("fixing")
	if !ok {
		return
	}
	go func() {
		sink := func(ev wtrain.Event) {
			a.convertAndEmitTrainEvent(runID, ev)
		}
		var err error
		if a.trainController != nil {
			err = wtrain.RunSingleLaneAlgoFeature(ctx, req.Model, req.Method, feature, sink)
		} else {
			err = wtrain.RunTrickIteration(ctx, req.Model, req.Method, feature, sink)
		}
		if err != nil && ctx.Err() == nil {
			a.EventCh <- model.Event{
				Type:    model.TrainError,
				Message: fmt.Sprintf("algo-feature iteration failed: %v", err),
			}
			return
		}
		// Set flag so rerun uses improved training data.
		a.trainMu.Lock()
		if a.trainReq != nil {
			if a.trainReq.Target.Config == nil {
				a.trainReq.Target.Config = map[string]any{}
			}
			a.trainReq.Target.Config["demo_trick_applied"] = true
		}
		if r, ok := a.trainReqs[a.trainCurrentRun]; ok {
			if r.Target.Config == nil {
				r.Target.Config = map[string]any{}
			}
			r.Target.Config["demo_trick_applied"] = true
			a.trainReqs[a.trainCurrentRun] = r
		}
		a.trainMu.Unlock()
	}()
}

func (a *Application) addPerfFeature(feature string) {
	ctx, runID, req, _, ok := a.beginTrainTask("fixing")
	if !ok {
		return
	}
	go func() {
		sink := func(ev wtrain.Event) {
			a.convertAndEmitTrainEvent(runID, ev)
		}
		err := wtrain.RunSingleLanePerfFeature(ctx, req.Model, req.Method, feature, sink)
		if err != nil && ctx.Err() == nil {
			a.EventCh <- model.Event{
				Type:    model.TrainError,
				Message: fmt.Sprintf("perf-feature iteration failed: %v", err),
			}
			return
		}
		a.trainMu.Lock()
		if a.trainReq != nil {
			if a.trainReq.Target.Config == nil {
				a.trainReq.Target.Config = map[string]any{}
			}
			a.trainReq.Target.Config["demo_perf_applied"] = true
		}
		if r, ok := a.trainReqs[a.trainCurrentRun]; ok {
			if r.Target.Config == nil {
				r.Target.Config = map[string]any{}
			}
			r.Target.Config["demo_perf_applied"] = true
			a.trainReqs[a.trainCurrentRun] = r
		}
		a.trainMu.Unlock()
	}()
}

func (a *Application) exitTrainMode() {
	a.resetTrainState()
	a.EventCh <- model.Event{Type: model.TrainModeClose}
}

// convertAndEmitTrainEvent maps workflow train events to UI events.
// The app-layer trainPhase remains the authoritative state for command gating.
// The UI keeps its own phase copy in TrainViewState for rendering only.
func (a *Application) convertAndEmitTrainEvent(runID uint64, ev wtrain.Event) {
	if !a.isCurrentTrainRun(runID) {
		return
	}

	switch ev.Kind {
	case wtrain.EventMessage:
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: ev.Message,
			Train: &model.TrainEventData{
				ActionSource: ev.ActionSource,
			},
		}

	case wtrain.EventDiffLine:
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: ev.Message,
			Train: &model.TrainEventData{
				IsDiff: true,
			},
		}

	case wtrain.EventTrainSetupStarted:
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: ev.Message,
			Train: &model.TrainEventData{
				ActionSource: ev.ActionSource,
			},
		}

	case wtrain.EventCheckStarted:
		a.EventCh <- model.Event{
			Type: model.TrainSetup,
			Train: &model.TrainEventData{
				RunID:  ev.RunID,
				Check:  ev.Check,
				Status: "checking",
				Scope:  ev.Scope,
			},
		}

	case wtrain.EventCheckPassed:
		a.EventCh <- model.Event{
			Type: model.TrainSetup,
			Train: &model.TrainEventData{
				RunID:    ev.RunID,
				Check:    ev.Check,
				Status:   "passed",
				Detail:   ev.Message,
				Scope:    ev.Scope,
				Critical: ev.Critical,
			},
		}

	case wtrain.EventCheckFailed:
		a.EventCh <- model.Event{
			Type: model.TrainSetup,
			Train: &model.TrainEventData{
				RunID:    ev.RunID,
				Check:    ev.Check,
				Status:   "failed",
				Detail:   ev.Message,
				Scope:    ev.Scope,
				Critical: ev.Critical,
			},
		}

	case wtrain.EventHostConnecting:
		a.EventCh <- model.Event{
			Type: model.TrainConnect,
			Train: &model.TrainEventData{
				RunID:   ev.RunID,
				Host:    ev.Host,
				Address: ev.Address,
				Status:  "connecting",
			},
		}

	case wtrain.EventHostConnected:
		a.EventCh <- model.Event{
			Type: model.TrainConnect,
			Train: &model.TrainEventData{
				RunID:   ev.RunID,
				Host:    ev.Host,
				Address: ev.Address,
				Status:  "connected",
			},
		}

	case wtrain.EventHostFailed:
		a.EventCh <- model.Event{
			Type: model.TrainConnect,
			Train: &model.TrainEventData{
				RunID:  ev.RunID,
				Host:   ev.Host,
				Status: "failed",
			},
		}

	case wtrain.EventConnectionStatus:
		a.EventCh <- model.Event{
			Type: model.TrainConnect,
			Train: &model.TrainEventData{
				RunID:   ev.RunID,
				Host:    ev.Host,
				Address: ev.Address,
				Status:  ev.Message,
			},
		}

	case wtrain.EventIssueDetected:
		a.EventCh <- model.Event{
			Type:    model.TrainIssueDetected,
			Message: ev.Message,
			Train: &model.TrainEventData{
				RunID:       ev.RunID,
				IssueID:     valueOrString(ev.IssueID, "issue-"+ev.RunID),
				IssueType:   ev.IssueType,
				IssueTitle:  ev.IssueTitle,
				IssueDetail: ev.IssueDetail,
			},
		}

	case wtrain.EventPlanReady:
		a.EventCh <- model.Event{
			Type:    model.TrainPlanReady,
			Message: ev.Message,
			Train: &model.TrainEventData{
				RunID:        ev.RunID,
				PlanID:       ev.PlanID,
				RepoPath:     ev.RepoPath,
				RepoSource:   ev.RepoSource,
				ScriptPath:   ev.ScriptPath,
				BaseModelRef: ev.BaseModelRef,
				ConfigPath:   ev.ConfigPath,
				EnvKind:      ev.EnvKind,
				Workdir:      ev.Workdir,
			},
		}

	case wtrain.EventReadyToStart:
		a.setTrainPhase("ready")
		a.clearBootstrapState()
		a.EventCh <- model.Event{
			Type:    model.TrainReady,
			Message: ev.Message,
			Train: &model.TrainEventData{
				RunID:        ev.RunID,
				ActionSource: ev.ActionSource,
			},
		}

	case wtrain.EventTrainStarted:
		a.setTrainPhase("running")
		a.EventCh <- model.Event{
			Type:    model.TrainStarted,
			Message: ev.Message,
			Train: &model.TrainEventData{
				RunID:    ev.RunID,
				Lane:     ev.Lane,
				RunLabel: ev.RunLabel,
			},
		}

	case wtrain.EventLogLine:
		a.EventCh <- model.Event{
			Type:    model.TrainLogLine,
			Message: ev.Message,
			Train: &model.TrainEventData{
				RunID: ev.RunID,
				Lane:  ev.Lane,
			},
		}

	case wtrain.EventMetricUpdate:
		a.EventCh <- model.Event{
			Type: model.TrainMetric,
			Train: &model.TrainEventData{
				RunID:      ev.RunID,
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
		// When GPU completes and NPU already failed, transition to failed.
		snapshot := a.getTrainSnapshot()
		if ev.Lane == "gpu" && snapshot.issueType == "runtime" && snapshot.phase == "running" {
			a.setTrainPhase("failed")
		} else if ev.Lane == "" {
			// Phase 1 single-lane completion
			a.setTrainPhase("completed")
		}
		a.EventCh <- model.Event{
			Type:    model.TrainDone,
			Message: ev.Message,
			Train: &model.TrainEventData{
				RunID:    ev.RunID,
				Lane:     ev.Lane,
				RunLabel: ev.RunLabel,
			},
		}

	case wtrain.EventTrainStopped:
		a.setTrainPhase("stopped")
		a.EventCh <- model.Event{
			Type:    model.TrainStopped,
			Message: ev.Message,
			Train: &model.TrainEventData{
				RunID: ev.RunID,
				Lane:  ev.Lane,
			},
		}

	case wtrain.EventTrainFailed:
		a.setTrainPhase("failed")
		if ev.IssueType != "" {
			a.setTrainIssueType(ev.IssueType)
		}
		a.clearBootstrapState()
		a.EventCh <- model.Event{
			Type:    model.TrainIssueDetected,
			Message: ev.Message,
			Train: &model.TrainEventData{
				RunID:       ev.RunID,
				IssueID:     valueOrString(ev.IssueID, "failure-"+ev.RunID),
				IssueType:   valueOrString(ev.IssueType, "failure"),
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
			Train: &model.TrainEventData{
				RunID: ev.RunID,
			},
		}

	case wtrain.EventEvalCompleted:
		a.EventCh <- model.Event{
			Type:    model.TrainEvalCompleted,
			Message: ev.Message,
			Train: &model.TrainEventData{
				RunID:        ev.RunID,
				BaselineAcc:  ev.BaselineAcc,
				CandidateAcc: ev.CandidateAcc,
				Drift:        ev.Drift,
				RunLabel:     ev.RunLabel,
			},
		}

	case wtrain.EventDriftDetected:
		a.setTrainIssueType("accuracy")
		a.setTrainPhase("drift_detected")
		// Emit TrainIssueDetected without Message to avoid duplicate chat line —
		// TrainDriftDetected below will show the message.
		a.EventCh <- model.Event{
			Type: model.TrainIssueDetected,
			Train: &model.TrainEventData{
				RunID:       ev.RunID,
				IssueID:     valueOrString(ev.IssueID, "accuracy-issue"),
				IssueType:   "accuracy",
				IssueTitle:  "Accuracy drift detected",
				IssueDetail: ev.Message,
			},
		}
		a.EventCh <- model.Event{
			Type:    model.TrainDriftDetected,
			Message: ev.Message,
			Train: &model.TrainEventData{
				RunID:        ev.RunID,
				BaselineAcc:  ev.BaselineAcc,
				CandidateAcc: ev.CandidateAcc,
				Drift:        ev.Drift,
			},
		}

	case wtrain.EventAnalysisStarted:
		a.setTrainPhase("analyzing")
		a.EventCh <- model.Event{
			Type:    model.TrainAnalysisStarted,
			Message: ev.Message,
			Train: &model.TrainEventData{
				RunID:        ev.RunID,
				ActionSource: ev.ActionSource,
			},
		}

	case wtrain.EventActionSuggested:
		if ev.IssueType == "bootstrap" {
			a.setBootstrapPendingAction(ev.RunID, ev.ActionID)
		}
		a.EventCh <- model.Event{
			Type:    model.TrainActionSuggested,
			Message: ev.Message,
			Train: &model.TrainEventData{
				RunID:        ev.RunID,
				ActionID:     ev.ActionID,
				ActionKind:   ev.ActionKind,
				ActionLabel:  ev.ActionLabel,
				ActionSource: ev.ActionSource,
				FixSummary:   ev.FixSummary,
				IssueType:    ev.IssueType,
			},
		}

	case wtrain.EventActionApplied:
		a.EventCh <- model.Event{
			Type:    model.TrainActionApplied,
			Message: ev.Message,
			Train: &model.TrainEventData{
				RunID:        ev.RunID,
				ActionID:     ev.ActionID,
				ActionKind:   ev.ActionKind,
				ActionLabel:  ev.ActionLabel,
				ActionSource: ev.ActionSource,
				IssueType:    ev.IssueType,
			},
		}

	case wtrain.EventAnalysisReady:
		a.setTrainPhase("ready")
		a.EventCh <- model.Event{
			Type:    model.TrainAnalysisReady,
			Message: ev.Message,
			Train: &model.TrainEventData{
				RunID:        ev.RunID,
				IssueType:    ev.IssueType,
				IssueTitle:   ev.IssueTitle,
				IssueDetail:  ev.IssueDetail,
				Confidence:   ev.Confidence,
				FixSummary:   ev.FixSummary,
				DiffText:     ev.DiffText,
				ActionID:     ev.ActionID,
				ActionKind:   ev.ActionKind,
				ActionLabel:  ev.ActionLabel,
				ActionSource: ev.ActionSource,
			},
		}

	case wtrain.EventFixApplied:
		a.setTrainPhase("ready")
		a.EventCh <- model.Event{
			Type:    model.TrainFixApplied,
			Message: ev.Message,
			Train: &model.TrainEventData{
				RunID:        ev.RunID,
				Lane:         ev.Lane,
				FixSummary:   ev.FixSummary,
				DiffText:     ev.DiffText,
				ActionSource: ev.ActionSource,
			},
		}

	case wtrain.EventRerunStarted:
		a.setTrainPhase("running")
		a.EventCh <- model.Event{
			Type:    model.TrainRerunStarted,
			Message: ev.Message,
			Train: &model.TrainEventData{
				RunID:    ev.RunID,
				Lane:     ev.Lane,
				RunLabel: ev.RunLabel,
			},
		}

	case wtrain.EventVerificationPassed:
		a.setTrainPhase("completed")
		a.EventCh <- model.Event{
			Type:    model.TrainVerified,
			Message: ev.Message,
			Train: &model.TrainEventData{
				RunID:        ev.RunID,
				ActionSource: ev.ActionSource,
				BaselineAcc:  ev.BaselineAcc,
				CandidateAcc: ev.CandidateAcc,
				Drift:        ev.Drift,
			},
		}
	}
}

func valueOrString(value, fallback string) string {
	if strings.TrimSpace(value) != "" {
		return value
	}
	return fallback
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
	a.trainReqs = map[string]train.Request{
		req.RunID: req,
	}
	a.trainBootstrap = map[string]*bootstrapRunState{}
	if a.isBootstrapRequest(req) {
		a.trainBootstrap[req.RunID] = &bootstrapRunState{Applied: map[string]bool{}}
	}
	a.trainCurrentRun = req.RunID
	a.trainIssueType = ""
	a.trainCancel = cancel
	a.trainTasks = map[uint64]struct{}{
		runID: {},
	}
	a.trainMu.Unlock()

	if oldCancel != nil {
		oldCancel()
	}

	return ctx, runID
}

func (a *Application) appendTrainRun(req train.Request) (context.Context, uint64) {
	ctx, cancel := context.WithCancel(context.Background())

	a.trainMu.Lock()
	a.trainRunID++
	runID := a.trainRunID
	if a.trainReqs == nil {
		a.trainReqs = map[string]train.Request{}
	}
	a.trainReqs[req.RunID] = req
	if a.trainBootstrap == nil {
		a.trainBootstrap = map[string]*bootstrapRunState{}
	}
	if a.isBootstrapRequest(req) && a.trainBootstrap[req.RunID] == nil {
		a.trainBootstrap[req.RunID] = &bootstrapRunState{Applied: map[string]bool{}}
	}
	a.trainReq = &req
	a.trainCurrentRun = req.RunID
	a.trainPhase = "setup"
	if a.trainTasks == nil {
		a.trainTasks = map[uint64]struct{}{}
	}
	a.trainTasks[runID] = struct{}{}
	a.trainCancel = cancel
	a.trainMu.Unlock()

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
	if current, ok := a.trainReqs[a.trainCurrentRun]; ok {
		req = current
	}
	issueType := a.trainIssueType
	a.trainPhase = phase
	a.trainCancel = cancel
	if a.trainTasks == nil {
		a.trainTasks = map[uint64]struct{}{}
	}
	a.trainTasks[runID] = struct{}{}
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
	a.trainReqs = nil
	a.trainCurrentRun = ""
	a.trainIssueType = ""
	a.trainCancel = nil
	a.trainTasks = nil
	a.trainBootstrap = nil
	a.trainController = nil
	a.trainMu.Unlock()

	if oldCancel != nil {
		oldCancel()
	}
}

func (a *Application) clearBootstrapState() {
	a.trainMu.Lock()
	a.trainBootstrap = nil
	a.trainMu.Unlock()
}

func (a *Application) isCurrentTrainRun(runID uint64) bool {
	a.trainMu.RLock()
	defer a.trainMu.RUnlock()
	if !a.trainMode {
		return false
	}
	_, ok := a.trainTasks[runID]
	return ok
}

func (a *Application) nextWorkspaceRunID() string {
	a.trainMu.RLock()
	defer a.trainMu.RUnlock()
	if !a.trainMode || len(a.trainReqs) == 0 {
		return "primary"
	}
	return fmt.Sprintf("run-%d", len(a.trainReqs)+1)
}

func (a *Application) isBootstrapRequest(req train.Request) bool {
	return strings.TrimSpace(req.Model) == "" || strings.TrimSpace(req.Method) == ""
}

func (a *Application) currentBootstrapAction() (string, string, bool) {
	a.trainMu.RLock()
	defer a.trainMu.RUnlock()
	if a.trainBootstrap == nil {
		return "", "", false
	}
	state := a.trainBootstrap[a.trainCurrentRun]
	if state == nil || strings.TrimSpace(state.PendingActionID) == "" {
		return "", "", false
	}
	return a.trainCurrentRun, state.PendingActionID, true
}

func (a *Application) setBootstrapPendingAction(runID, actionID string) {
	a.trainMu.Lock()
	defer a.trainMu.Unlock()
	if a.trainBootstrap == nil {
		a.trainBootstrap = map[string]*bootstrapRunState{}
	}
	state := a.trainBootstrap[runID]
	if state == nil {
		state = &bootstrapRunState{Applied: map[string]bool{}}
		a.trainBootstrap[runID] = state
	}
	state.PendingActionID = actionID
}

func (a *Application) markBootstrapActionApplied(runID, actionID string) {
	a.trainMu.Lock()
	defer a.trainMu.Unlock()
	if a.trainBootstrap == nil {
		a.trainBootstrap = map[string]*bootstrapRunState{}
	}
	state := a.trainBootstrap[runID]
	if state == nil {
		state = &bootstrapRunState{Applied: map[string]bool{}}
		a.trainBootstrap[runID] = state
	}
	if state.Applied == nil {
		state.Applied = map[string]bool{}
	}
	state.Applied[actionID] = true
	state.PendingActionID = ""
}

func (a *Application) bootstrapApplied(runID string) map[string]bool {
	a.trainMu.RLock()
	defer a.trainMu.RUnlock()
	if a.trainBootstrap == nil || a.trainBootstrap[runID] == nil || len(a.trainBootstrap[runID].Applied) == 0 {
		return nil
	}
	out := make(map[string]bool, len(a.trainBootstrap[runID].Applied))
	for actionID, done := range a.trainBootstrap[runID].Applied {
		out[actionID] = done
	}
	return out
}
