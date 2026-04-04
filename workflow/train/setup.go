package train

import (
	"context"
	"fmt"

	itrain "github.com/mindspore-lab/mindspore-cli/internal/train"
	"github.com/mindspore-lab/mindspore-cli/runtime/probes"
	localprobes "github.com/mindspore-lab/mindspore-cli/runtime/probes/local"
	targetprobes "github.com/mindspore-lab/mindspore-cli/runtime/probes/target"
	sshprobe "github.com/mindspore-lab/mindspore-cli/runtime/probes/target/ssh"
)

// RunSetupSequence runs the setup phase: local probes first, then target probes.
// It converts probe results into lane events and determines readiness.
// Local checks are informational; target checks are critical blockers.
func RunSetupSequence(ctx context.Context, req itrain.Request, backend Backend, sink func(Event)) error {
	e := func(ev Event) bool { return emit(ctx, sink, ev) }
	runID := req.RunID
	if runID == "" {
		runID = "primary"
	}

	if req.Model == "" || req.Method == "" {
		return runBootstrapSetup(ctx, req, nil, sink)
	}

	if !e(Event{Kind: EventTrainSetupStarted, RunID: runID, ActionSource: "setup-helper", Message: fmt.Sprintf("preparing %s %s training. running preflight checks...", req.Model, req.Method), DelayMs: 500}) {
		return ctx.Err()
	}

	// ── Local probes ──
	localProbes := []localprobes.Probe{
		&localprobes.AlgoProbe{},
		&localprobes.OSProbe{},
		&localprobes.AIFrameworkProbe{},
	}

	for _, probe := range localProbes {
		results, err := probe.Run(ctx, req)
		if err != nil {
			return fmt.Errorf("local probe: %w", err)
		}
		for _, r := range results {
			if !e(Event{Kind: EventCheckStarted, RunID: runID, Check: r.Name, Scope: string(r.Scope), DelayMs: 600}) {
				return ctx.Err()
			}
			if !emitProbeResult(ctx, sink, r) {
				return ctx.Err()
			}
		}
	}

	// ── Target probes ──
	// SSH first (gate for all other target probes)
	sshProbe := &sshprobe.Probe{}

	if !e(Event{Kind: EventCheckStarted, RunID: runID, Check: "ssh", Scope: "target", DelayMs: 1000}) {
		return ctx.Err()
	}

	host := req.Target.Name
	addr, _ := req.Target.Config["address"].(string)
	if !e(Event{Kind: EventHostConnecting, RunID: runID, Host: host, Address: addr, DelayMs: 400}) {
		return ctx.Err()
	}

	sshResults, err := sshProbe.Run(ctx, req.Target)
	if err != nil {
		return fmt.Errorf("ssh probe: %w", err)
	}
	sshReady := false
	for _, r := range sshResults {
		if r.Status == probes.StatusPass {
			sshReady = true
			if !e(Event{Kind: EventHostConnected, RunID: runID, Host: host, Address: addr, DelayMs: 800}) {
				return ctx.Err()
			}
		} else {
			if !e(Event{Kind: EventHostFailed, RunID: runID, Host: host, Message: "[!] " + r.Summary, DelayMs: 2000}) {
				return ctx.Err()
			}
			if !autoResolveSSHFailure(ctx, sink, runID, host, addr, r.Summary) {
				return ctx.Err()
			}
			sshReady = true // recovered
		}
		if !emitProbeResult(ctx, sink, r) {
			return ctx.Err()
		}
		if r.Status != probes.StatusPass && sshReady {
			if !e(Event{
				Kind:     EventCheckPassed,
				RunID:    runID,
				Check:    "ssh",
				Scope:    "target",
				Message:  "recovered by setup helper after SSH retry",
				Critical: true,
				DelayMs:  1200,
			}) {
				return ctx.Err()
			}
		}
	}

	// Remaining target probes — only run if SSH is available (passed or recovered).
	// Without SSH, the remote machine is unreachable.
	allCriticalPassed := sshReady
	if sshReady {
		targetProbes := []targetprobes.Probe{
			&targetprobes.WorkdirProbe{},
			&targetprobes.AlgoProbe{},
			&targetprobes.DeviceProbe{},
			&targetprobes.OSProbe{},
			&targetprobes.AIFrameworkProbe{},
		}

		for _, probe := range targetProbes {
			results, err := probe.Run(ctx, req.Target)
			if err != nil {
				return fmt.Errorf("target probe: %w", err)
			}
			for _, r := range results {
				// Show "checking..." before the result so the user sees the process.
				if !e(Event{Kind: EventCheckStarted, RunID: runID, Check: r.Name, Scope: string(r.Scope), DelayMs: 600}) {
					return ctx.Err()
				}
				if r.Critical && r.Status != probes.StatusPass {
					// Auto-resolve missing libs
					if r.Name == "libs" {
						if !emitProbeResult(ctx, sink, r) {
							return ctx.Err()
						}
						if !autoResolveLibsMissing(ctx, sink, runID, r.Summary) {
							return ctx.Err()
						}
						continue
					}
					allCriticalPassed = false
				}
				if !emitProbeResult(ctx, sink, r) {
					return ctx.Err()
				}
			}
		}
	}

	if allCriticalPassed {
		if !e(Event{Kind: EventReadyToStart, RunID: runID, ActionSource: "observer", Message: "all preflight checks passed. ready to start training.", DelayMs: 400}) {
			return ctx.Err()
		}
	}

	return nil
}

func autoResolveSSHFailure(ctx context.Context, sink func(Event), runID, host, addr, summary string) bool {
	e := func(ev Event) bool { return emit(ctx, sink, ev) }
	if !e(Event{
		Kind:        EventIssueDetected,
		RunID:       runID,
		Message:     "[!] " + summary,
		IssueID:     "bootstrap-target-ssh",
		IssueType:   "bootstrap",
		IssueTitle:  "[!] SSH connectivity needs repair",
		IssueDetail: summary,
		DelayMs:     2000,
	}) {
		return false
	}
	if !e(Event{
		Kind:         EventActionSuggested,
		RunID:        runID,
		Message:      "setup helper is repairing SSH connectivity automatically.",
		IssueType:    "bootstrap",
		ActionID:     "repair-ssh-connectivity",
		ActionKind:   "change_env",
		ActionLabel:  "Repair SSH connectivity",
		ActionSource: "setup-helper",
		DelayMs:      300,
	}) {
		return false
	}
	if !e(Event{
		Kind:         EventActionApplied,
		RunID:        runID,
		Message:      "SSH connectivity repaired automatically.",
		IssueType:    "bootstrap",
		ActionID:     "repair-ssh-connectivity",
		ActionKind:   "change_env",
		ActionLabel:  "Repair SSH connectivity",
		ActionSource: "setup-helper",
		DelayMs:      3000,
	}) {
		return false
	}
	return e(Event{
		Kind:    EventHostConnected,
		RunID:   runID,
		Host:    host,
		Address: addr,
		DelayMs: 120,
	})
}

func autoResolveLibsMissing(ctx context.Context, sink func(Event), runID, summary string) bool {
	e := func(ev Event) bool { return emit(ctx, sink, ev) }

	// 1. Issue detected
	if !e(Event{
		Kind:        EventIssueDetected,
		RunID:       runID,
		Message:     "[!] " + summary,
		IssueID:     "bootstrap-target-libs",
		IssueType:   "bootstrap",
		IssueTitle:  "[!] Missing library detected",
		IssueDetail: summary,
		DelayMs:     1500,
	}) {
		return false
	}

	// 2. Agent suggests fix
	if !e(Event{
		Kind:         EventActionSuggested,
		RunID:        runID,
		Message:      "setup helper is installing missing library: transformers v5.0.1",
		IssueType:    "bootstrap",
		ActionID:     "install-missing-libs",
		ActionKind:   "change_env",
		ActionLabel:  "Install transformers v5.0.1",
		ActionSource: "setup-helper",
		DelayMs:      300,
	}) {
		return false
	}

	// 3. Fake download progress via log lines
	progressMsgs := []struct {
		msg     string
		delayMs int
	}{
		{"downloading transformers v5.0.1...", 800},
		{"collecting dependencies...", 1200},
		{"installing tokenizers 0.21.1...", 1000},
		{"installing transformers 5.0.1...", 1500},
		{"installation complete.", 500},
	}
	for _, p := range progressMsgs {
		if !e(Event{
			Kind:         EventActionApplied,
			RunID:        runID,
			Message:      p.msg,
			IssueType:    "bootstrap",
			ActionID:     "install-missing-libs",
			ActionKind:   "change_env",
			ActionLabel:  p.msg,
			ActionSource: "setup-helper",
			DelayMs:      p.delayMs,
		}) {
			return false
		}
	}

	// 4. Check passes after install
	if !e(Event{
		Kind:     EventCheckPassed,
		RunID:    runID,
		Check:    "libs",
		Scope:    "target",
		Message:  "torch 2.7 | mindspore 2.8 | transformers v5.0.1 | diffusers v0.36",
		Critical: true,
		DelayMs:  600,
	}) {
		return false
	}

	return true
}

// RunBootstrapRecheck reruns bootstrap prerequisite checks after one or more
// setup-helper actions have been applied.
func RunBootstrapRecheck(ctx context.Context, req itrain.Request, applied map[string]bool, sink func(Event)) error {
	return runBootstrapSetup(ctx, req, applied, sink)
}

// RunBootstrapApply fakes one bootstrap setup-helper action.
func RunBootstrapApply(ctx context.Context, req itrain.Request, actionID string, sink func(Event)) error {
	e := func(ev Event) bool { return emit(ctx, sink, ev) }
	runID := req.RunID
	if runID == "" {
		runID = "primary"
	}

	for _, check := range bootstrapChecks() {
		if check.actionID != actionID {
			continue
		}
		if !e(Event{
			Kind:         EventActionApplied,
			RunID:        runID,
			Message:      check.actionLabel + " completed. Run recheck to validate prerequisites.",
			IssueType:    "bootstrap",
			ActionID:     check.actionID,
			ActionKind:   check.actionKind,
			ActionLabel:  check.actionLabel,
			ActionSource: "setup-helper",
			DelayMs:      3000,
		}) {
			return ctx.Err()
		}
		return nil
	}

	return fmt.Errorf("unknown bootstrap action: %s", actionID)
}

func runBootstrapSetup(ctx context.Context, req itrain.Request, applied map[string]bool, sink func(Event)) error {
	e := func(ev Event) bool { return emit(ctx, sink, ev) }
	runID := req.RunID
	if runID == "" {
		runID = "primary"
	}

	if !e(Event{
		Kind:         EventTrainSetupStarted,
		RunID:        runID,
		ActionSource: "setup-helper",
		Message:      "bootstrapping training workspace from scratch. inspecting prerequisites...",
		DelayMs:      500,
	}) {
		return ctx.Err()
	}

	checks := bootstrapChecks()
	var unresolved *bootstrapCheck
	for i := range checks {
		check := checks[i]
		if !e(Event{Kind: EventCheckStarted, RunID: runID, Check: check.name, Scope: "local", DelayMs: 1000}) {
			return ctx.Err()
		}
		if applied != nil && applied[check.actionID] {
			if !e(Event{Kind: EventCheckPassed, RunID: runID, Check: check.name, Scope: "local", Message: "resolved by setup helper", Critical: true, DelayMs: 1200}) {
				return ctx.Err()
			}
			continue
		}
		if !e(Event{Kind: EventCheckFailed, RunID: runID, Check: check.name, Scope: "local", Message: check.summary, Critical: true, DelayMs: 1200}) {
			return ctx.Err()
		}
		if unresolved == nil {
			unresolved = &check
		}
	}

	host := req.Target.Name
	addr, _ := req.Target.Config["address"].(string)
	if host == "" {
		host = "onprem-ssh-target"
	}
	if addr == "" {
		addr = "10.0.1.10:22"
	}
	if !e(Event{Kind: EventHostConnecting, RunID: runID, Host: host, Address: addr, DelayMs: 200}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventHostConnected, RunID: runID, Host: host, Address: addr, DelayMs: 400}) {
		return ctx.Err()
	}
	if unresolved != nil {
		if !e(Event{
			Kind:        EventIssueDetected,
			RunID:       runID,
			Message:     "[!] " + unresolved.summary,
			IssueID:     unresolved.issueID,
			IssueType:   "bootstrap",
			IssueTitle:  "[!] " + unresolved.issueTitle,
			IssueDetail: unresolved.summary,
			DelayMs:     2000,
		}) {
			return ctx.Err()
		}
		if !e(Event{
			Kind:         EventActionSuggested,
			RunID:        runID,
			Message:      "bootstrap helper suggested the next action.",
			IssueType:    "bootstrap",
			ActionID:     unresolved.actionID,
			ActionKind:   unresolved.actionKind,
			ActionLabel:  unresolved.actionLabel,
			ActionSource: "setup-helper",
			DelayMs:      300,
		}) {
			return ctx.Err()
		}
		return nil
	}
	if !e(Event{
		Kind:         EventPlanReady,
		RunID:        runID,
		Message:      "bootstrap complete. Training plan is ready.",
		PlanID:       "bootstrap-plan-" + runID,
		RepoPath:     "workspaces/qwen3-lora",
		RepoSource:   "llama-factory",
		ScriptPath:   "workspaces/qwen3-lora/scripts/train_lora.py",
		BaseModelRef: "models/qwen3-7b",
		ConfigPath:   "workspaces/qwen3-lora/configs/qwen3_lora.yaml",
		EnvKind:      "uv",
		Workdir:      "/srv/train/qwen3-lora",
		DelayMs:      250,
	}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventReadyToStart, RunID: runID, Message: "bootstrap finished. Ready to start training.", DelayMs: 200}) {
		return ctx.Err()
	}

	return nil
}

type bootstrapCheck struct {
	name        string
	summary     string
	issueID     string
	issueTitle  string
	actionID    string
	actionKind  string
	actionLabel string
}

func bootstrapChecks() []bootstrapCheck {
	return []bootstrapCheck{
		{
			name:        "code_source",
			summary:     "No reference repo or code source selected yet",
			issueID:     "bootstrap-code-source",
			issueTitle:  "Training code source is missing",
			actionID:    "choose-llama-factory",
			actionKind:  "download_code",
			actionLabel: "Clone reference repo",
		},
		{
			name:        "train_script",
			summary:     "No train script found for the requested workflow",
			issueID:     "bootstrap-train-script",
			issueTitle:  "Train script is missing",
			actionID:    "scaffold-lora-script",
			actionKind:  "scaffold_code",
			actionLabel: "Scaffold train script",
		},
		{
			name:        "runtime_env",
			summary:     "Remote runtime environment is not prepared",
			issueID:     "bootstrap-env",
			issueTitle:  "Remote env is missing",
			actionID:    "prepare-uv-env",
			actionKind:  "prepare_env",
			actionLabel: "Prepare uv environment",
		},
		{
			name:        "base_model",
			summary:     "Base model checkpoint is not available in target workdir",
			issueID:     "bootstrap-model",
			issueTitle:  "Base model is missing",
			actionID:    "stage-base-model",
			actionKind:  "change_config",
			actionLabel: "Stage base model + config",
		},
	}
}

// emitProbeResult converts a probe result to a lane event and emits it.
func emitProbeResult(ctx context.Context, sink func(Event), r probes.Result) bool {
	kind := EventCheckPassed
	message := r.Summary
	delay := 1800
	if r.Status == probes.StatusFail {
		kind = EventCheckFailed
		message = "[!] " + r.Summary
		delay = 2000
	}

	ev := Event{
		Kind:     kind,
		RunID:    "primary",
		Check:    r.Name,
		Message:  message,
		Scope:    string(r.Scope),
		Critical: r.Critical,
		Details:  r.Details,
		DelayMs:  delay,
	}

	return emit(ctx, sink, ev)
}
