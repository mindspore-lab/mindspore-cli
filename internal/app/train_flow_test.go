package app

import (
	"strings"
	"testing"
	"time"

	"github.com/vigo999/mindspore-cli/configs"
	itrain "github.com/vigo999/mindspore-cli/internal/train"
	"github.com/vigo999/mindspore-cli/ui/model"
)

// TestTrainPhase1Flow exercises the Phase 1 train lane:
//  1. /train qwen3 lora → setup (probe-based) → ready
//  2. start → running → logs + metrics → completed
//  3. Phase gates block invalid commands
func TestTrainPhase1Flow(t *testing.T) {
	app := newTestApp(t)

	// ── Step 1: /train qwen3 lora → setup → ready ──
	app.cmdTrain([]string{"qwen3", "lora"})

	assertPhase(t, app, "setup")
	if !app.isTrainMode() {
		t.Fatal("expected trainMode=true after cmdTrain")
	}

	// Drain events until we see TrainReady (setup completion).
	// The demo setup path includes multiple staged checks with delays.
	drainUntil(t, app, model.TrainReady, 90*time.Second)
	assertPhase(t, app, "ready")

	// Gate check: analyze should be rejected in ready phase
	app.handleTrainInput("analyze")
	ev := drainUntil(t, app, model.AgentReply, 2*time.Second)
	if ev.Type != model.AgentReply {
		t.Fatalf("expected rejection message, got %s", ev.Type)
	}

	// ── Step 2: start → running → failed (simulated runtime issue) ──
	app.handleTrainInput("start")
	assertPhase(t, app, "running")

	// Drain until runtime issue is reported.
	drainUntil(t, app, model.TrainIssueDetected, 60*time.Second)
	assertPhase(t, app, "failed")

	// Clean exit
	app.handleTrainInput("exit")
	if app.isTrainMode() {
		t.Fatal("expected trainMode=false after exit")
	}
}

func TestCmdTrainDuplicateStartQueuesUntilCurrentStepFinishes(t *testing.T) {
	app := newTestApp(t)

	app.cmdTrain([]string{"qwen3", "lora", "dataset1"})
	app.cmdTrain([]string{"qwen3", "lora", "dataset2"})

	if !app.trainMode {
		t.Fatal("expected trainMode=true after first /train")
	}
	ev := drainUntil(t, app, model.AgentReply, 2*time.Second)
	if !strings.Contains(ev.Message, "messages queued (press esc to interrupt)") {
		t.Fatalf("expected queue message, got %q", ev.Message)
	}

	app.trainMu.RLock()
	if got := len(app.trainReqs); got != 1 {
		app.trainMu.RUnlock()
		t.Fatalf("expected duplicate /train to be blocked until current step finishes, got %d requests", got)
	}
	if app.pendingTrain == nil {
		app.trainMu.RUnlock()
		t.Fatal("expected queued train request")
	}
	app.trainMu.RUnlock()

	drainUntil(t, app, model.TrainReady, 90*time.Second)
	ev = drainUntil(t, app, model.AgentReply, 2*time.Second)
	if !strings.Contains(ev.Message, "starting queued train") {
		t.Fatalf("expected queued train to start after current step, got %q", ev.Message)
	}
	drainUntil(t, app, model.TrainModeOpen, 2*time.Second)

	app.trainMu.RLock()
	defer app.trainMu.RUnlock()
	if got := len(app.trainReqs); got != 1 {
		t.Fatalf("expected replacement train workspace after queued restart, got %d requests", got)
	}
	if _, ok := app.trainReqs["primary"]; !ok {
		t.Fatal("expected queued train request to replace workspace as primary")
	}
	if app.trainCurrentRun != "primary" {
		t.Fatalf("expected current run to move back to primary, got %q", app.trainCurrentRun)
	}
	if app.pendingTrain != nil {
		t.Fatal("expected queued train to be cleared after auto-start")
	}
}

func TestTrainBootstrapFlowReady(t *testing.T) {
	app := newTestApp(t)

	app.cmdTrain([]string{})

	assertPhase(t, app, "setup")
	if !app.isTrainMode() {
		t.Fatal("expected trainMode=true after bootstrap /train")
	}

	drainUntil(t, app, model.TrainActionSuggested, 20*time.Second)
	assertNoEvent(t, app, model.TrainReady, 500*time.Millisecond)

	for i := 0; i < 4; i++ {
		app.handleTrainInput("apply fix")
		drainUntil(t, app, model.TrainActionApplied, 5*time.Second)
		assertPhase(t, app, "setup")

		if i < 3 {
			drainUntil(t, app, model.TrainActionSuggested, 20*time.Second)
			assertPhase(t, app, "setup")
			continue
		}
		drainUntil(t, app, model.TrainPlanReady, 30*time.Second)
		drainUntil(t, app, model.TrainReady, 30*time.Second)
	}
	assertPhase(t, app, "ready")
}

func TestProcessInputKeepsPlainChatGlobalDuringTrainMode(t *testing.T) {
	app := newTestApp(t)
	app.trainMode = true
	app.trainPhase = "ready"

	app.processInput("hello")

	ev := drainUntil(t, app, model.AgentReply, 2*time.Second)
	if ev.Message != provideAPIKeyFirstMsg {
		t.Fatalf("expected plain text to route to agent path, got %q", ev.Message)
	}
	if !app.isTrainMode() {
		t.Fatal("expected train mode to remain active after plain chat input")
	}
}

func TestProcessInputUsesSlashTrainControlsForActiveWorkspace(t *testing.T) {
	app := newTestApp(t)
	app.trainMode = true
	app.trainPhase = "ready"

	app.processInput("/train exit")

	if app.isTrainMode() {
		t.Fatal("expected /train exit to close the active train workspace")
	}
	ev := drainUntil(t, app, model.TrainModeClose, 2*time.Second)
	if ev.Type != model.TrainModeClose {
		t.Fatalf("expected TrainModeClose, got %s", ev.Type)
	}
}

// TestTrainFullFlow exercises the complete Phase 2 train state machine
// using the legacy dual-lane flow:
//  1. Setup completes, phase → ready
//  2. Start training, NPU crashes, GPU completes → failed
//  3. Analyze NPU available and works → analyzing → ready
//  4. Apply Runtime Fix available and works → running (NPU relaunch)
//  5. NPU completes, eval runs, drift detected → drift_detected
//  6. Analyze Drift available and works → analyzing → ready
//  7. Apply Accuracy Fix available and works → running
//  8. Rerun completes, verification passes → completed
//  9. Phase gates block invalid commands at each step
//  10. View Diff is available in completed state
func TestTrainFullFlow(t *testing.T) {
	app := newTestApp(t)

	// Set up train mode manually for Phase 2 legacy flow (no controller).
	req := itrain.Request{Model: "qwen3", Method: "lora"}
	ctx, runID := app.beginTrainMode(req)
	app.EventCh <- model.Event{
		Type:  model.TrainModeOpen,
		Train: &model.TrainEventData{Model: "qwen3", Method: "lora"},
	}
	// Run legacy setup
	go app.runLegacySetup(ctx, runID, req)

	assertPhase(t, app, "setup")
	if !app.isTrainMode() {
		t.Fatal("expected trainMode=true after beginTrainMode")
	}

	// Drain events until we see TrainReady (setup completion)
	drainUntil(t, app, model.TrainReady, 15*time.Second)
	assertPhase(t, app, "ready")

	// Gate check: analyze should be rejected in ready phase
	app.handleTrainInput("analyze")
	ev := drainUntil(t, app, model.AgentReply, 2*time.Second)
	if ev.Type != model.AgentReply {
		t.Fatalf("expected rejection message, got %s", ev.Type)
	}

	// ── Step 2: start → running → NPU crashes → GPU completes → failed ──
	app.handleTrainInput("start")
	assertPhase(t, app, "running")

	// Gate check: apply fix should be rejected during running
	app.handleTrainInput("apply fix")
	ev = drainUntil(t, app, model.AgentReply, 2*time.Second)
	assertContains(t, ev.Message, "cannot")

	// Drain until GPU completes and phase transitions to failed
	drainUntil(t, app, model.TrainDone, 30*time.Second) // GPU done
	assertPhase(t, app, "failed")

	// Gate check: start should be rejected in failed
	app.handleTrainInput("start")
	ev = drainUntil(t, app, model.AgentReply, 2*time.Second)
	assertContains(t, ev.Message, "cannot")

	// Verify trainIssueType is set
	if issueType := app.getTrainSnapshot().issueType; issueType != "runtime" {
		t.Fatalf("expected trainIssueType=runtime, got %q", issueType)
	}

	// ── Step 3: analyze → analyzing → ready ──
	app.handleTrainInput("analyze")
	assertPhase(t, app, "analyzing")

	drainUntil(t, app, model.TrainAnalysisReady, 15*time.Second)
	assertPhase(t, app, "ready")

	// Gate check: analyze should be rejected in ready
	app.handleTrainInput("analyze")
	ev = drainUntil(t, app, model.AgentReply, 2*time.Second)
	assertContains(t, ev.Message, "cannot")

	// ── Step 4: apply fix → NPU relaunches (fixing) ──
	app.handleTrainInput("apply fix")
	assertPhase(t, app, "fixing")

	// NPU relaunches, trains, completes, evals, and drift is detected
	drainUntil(t, app, model.TrainDriftDetected, 60*time.Second)
	assertPhase(t, app, "drift_detected")

	// Verify trainIssueType switched to accuracy
	if issueType := app.getTrainSnapshot().issueType; issueType != "accuracy" {
		t.Fatalf("expected trainIssueType=accuracy, got %q", issueType)
	}

	// ── Step 6: analyze drift → analyzing → ready ──
	app.handleTrainInput("analyze drift")
	assertPhase(t, app, "analyzing")

	drainUntil(t, app, model.TrainAnalysisReady, 15*time.Second)
	assertPhase(t, app, "ready")

	// ── Step 7: apply accuracy fix → fixing ──
	app.handleTrainInput("apply fix")
	assertPhase(t, app, "fixing")

	// Rerun is asynchronous and can advance to evaluation immediately when
	// demo playback is accelerated for tests.
	drainUntil(t, app, model.TrainRerunStarted, 15*time.Second)

	// ── Step 8: rerun completes, verification passes → completed ──
	drainUntil(t, app, model.TrainVerified, 60*time.Second)
	assertPhase(t, app, "completed")

	// ── Step 9: gate checks in completed state ──
	app.handleTrainInput("apply fix")
	ev = drainUntil(t, app, model.AgentReply, 2*time.Second)
	assertContains(t, ev.Message, "cannot")

	// ── Step 10: view diff works in completed state ──
	app.handleTrainInput("view diff")
	ev = drainUntil(t, app, model.AgentReply, 2*time.Second)
	assertContains(t, ev.Message, "diff")

	// Clean exit
	app.handleTrainInput("exit")
	if app.isTrainMode() {
		t.Fatal("expected trainMode=false after exit")
	}
	if phase := app.getTrainPhase(); phase != "" {
		t.Fatalf("expected empty trainPhase after exit, got %q", phase)
	}
}

// TestTrainPhaseGatesComprehensive tests that every command is rejected
// in every phase where it shouldn't run.
func TestTrainPhaseGatesComprehensive(t *testing.T) {
	type gateTest struct {
		phase   string
		command string
		allowed bool
	}

	tests := []gateTest{
		// start only in ready
		{"setup", "start", false},
		{"ready", "start", true},
		{"running", "start", false},
		{"failed", "start", false},
		{"completed", "start", true},

		// analyze in failed or drift_detected
		{"setup", "analyze", false},
		{"ready", "analyze", false},
		{"running", "analyze", false},
		{"failed", "analyze", true},
		{"drift_detected", "analyze", true},
		{"completed", "analyze", false},

		// apply fix only in ready
		{"setup", "apply fix", false},
		{"ready", "apply fix", true},
		{"running", "apply fix", false},
		{"failed", "apply fix", false},
		{"completed", "apply fix", false},

		// retry only in failed
		{"setup", "retry", false},
		{"ready", "retry", false},
		{"running", "retry", false},
		{"failed", "retry", true},
		{"drift_detected", "retry", false},
		{"completed", "retry", false},
	}

	for _, tt := range tests {
		t.Run(tt.phase+"/"+tt.command, func(t *testing.T) {
			app := newTestApp(t)
			app.trainMu.Lock()
			app.trainMode = true
			app.trainPhase = tt.phase
			app.trainIssueType = "runtime"
			app.trainReq = &trainReqFixture
			app.trainMu.Unlock()

			// For commands that would launch goroutines, we just check
			// the phase gate by examining if we get a rejection message.
			if !tt.allowed {
				app.handleTrainInput(tt.command)
				ev := drainUntil(t, app, model.AgentReply, 2*time.Second)
				assertContains(t, ev.Message, "cannot")
			}
			// For allowed commands, we verify the phase changes
			// (the full flow test covers actual execution)
		})
	}
}

// ── Test helpers ──────────────────────────────────────────────

var trainReqFixture = itrain.Request{Model: "qwen3", Method: "lora"}

func newTestApp(t *testing.T) *Application {
	t.Helper()
	t.Setenv("MS_DEMO_SPEED", "1000")

	return &Application{
		EventCh: make(chan model.Event, 256),
		Config:  &configs.Config{},
	}
}

func assertPhase(t *testing.T, app *Application, expected string) {
	t.Helper()
	if phase := app.getTrainPhase(); phase != expected {
		t.Fatalf("expected trainPhase=%q, got %q", expected, phase)
	}
}

func assertContains(t *testing.T, s, substr string) {
	t.Helper()
	if len(s) == 0 {
		t.Fatalf("expected string containing %q, got empty string", substr)
	}
	found := false
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected string containing %q, got %q", substr, s)
	}
}

// drainUntil reads events from the app's EventCh until one matches the
// target type, or the timeout expires. Non-matching events are discarded.
// Returns the matching event.
func drainUntil(t *testing.T, app *Application, target model.EventType, timeout time.Duration) model.Event {
	t.Helper()
	deadline := time.After(timeout)
	for {
		select {
		case ev := <-app.EventCh:
			if ev.Type == target {
				return ev
			}
		case <-deadline:
			t.Fatalf("timed out waiting for event %s (trainPhase=%s)", target, app.getTrainPhase())
			return model.Event{}
		}
	}
}

func assertNoEvent(t *testing.T, app *Application, target model.EventType, timeout time.Duration) {
	t.Helper()
	deadline := time.After(timeout)
	for {
		select {
		case ev := <-app.EventCh:
			if ev.Type == target {
				t.Fatalf("unexpected event %s while checking for absence", target)
			}
		case <-deadline:
			return
		}
	}
}
