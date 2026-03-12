package train

import (
	"context"
	"fmt"
	"strings"
	"time"

	itrain "github.com/vigo999/ms-cli/internal/train"
)

func withDefaultRunID(ev Event) Event {
	if ev.RunID != "" {
		return ev
	}
	switch ev.Lane {
	case "gpu":
		ev.RunID = "torch_npu"
	case "npu":
		ev.RunID = "mindspore_npu"
	default:
		ev.RunID = "primary"
	}
	return ev
}

// emit sends an event via the sink, respecting delays and context cancellation.
func emit(ctx context.Context, sink func(Event), ev Event) bool {
	if ev.DelayMs > 0 {
		select {
		case <-ctx.Done():
			return false
		case <-time.After(time.Duration(ev.DelayMs) * time.Millisecond):
		}
	}
	sink(ev)
	return true
}

// DemoBackend provides demo/scenario playback for Phase 1.
// It implements the Backend interface.
type DemoBackend struct{}

// Setup runs the demo setup phase using probe-based sequencing.
// The actual setup is driven by RunSetupSequence which calls probes;
// this method is only called if the controller bypasses the probe pipeline.
func (d *DemoBackend) Setup(ctx context.Context, req itrain.Request, sink func(Event)) error {
	return RunSetupSequence(ctx, req, d, sink)
}

// Run plays back the demo training scenario: log lines + metrics + completion.
func (d *DemoBackend) Run(ctx context.Context, session *Session, sink func(Event)) error {
	return d.runTraining(ctx, session.Model, session.Method, sink, session.FailAtStep)
}

// runTraining plays demo training: loading, log lines, metrics, completion.
// If failAtStep > 0, training crashes at that step with a DSA operator error.
func (d *DemoBackend) runTraining(ctx context.Context, model, method string, sink func(Event), failAtStep ...int) error {
	failStep := 0
	if len(failAtStep) > 0 {
		failStep = failAtStep[0]
	}
	e := func(ev Event) bool { return emit(ctx, sink, withDefaultRunID(ev)) }

	runID := fmt.Sprintf("train-%s", time.Now().Format("0102-150405"))

	if !e(Event{Kind: EventLogLine, Message: fmt.Sprintf("[%s] Loading base model %s-7b...", runID, model), DelayMs: 600}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Message: fmt.Sprintf("[%s] LoRA config: rank=16, alpha=32, dropout=0.05", runID), DelayMs: 300}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Message: fmt.Sprintf("[%s] Loading dataset: alpaca_gpt4_zh (52,002 samples)", runID), DelayMs: 400}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Message: fmt.Sprintf("[%s] Graph compilation... OK", runID), DelayMs: 800}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Message: fmt.Sprintf("[%s] Begin training loop (300 steps)...", runID), DelayMs: 300}) {
		return ctx.Err()
	}

	type snap struct {
		step       int
		loss       float64
		lr         float64
		throughput float64
	}
	steps := []snap{
		{10, 2.891, 2.0e-4, 487.2},
		{25, 2.374, 2.0e-4, 512.8},
		{50, 1.956, 1.8e-4, 518.3},
		{75, 1.638, 1.5e-4, 516.7},
		{100, 1.401, 1.2e-4, 519.1},
		{150, 1.142, 8.0e-5, 517.4},
		{200, 0.964, 5.0e-5, 518.9},
		{250, 0.882, 3.0e-5, 519.2},
		{300, 0.831, 2.0e-5, 518.5},
	}

	totalSteps := 300
	for _, s := range steps {
		// Failure injection: crash at the specified step.
		if failStep > 0 && s.step >= failStep {
			crashMsg := fmt.Sprintf("[%s] FATAL: operator init failed at step %d — DSA operator (torch.ops.npu.dsa) not implemented in torch 2.7 for Ascend backend", runID, s.step)
			e(Event{Kind: EventLogLine, Message: crashMsg, DelayMs: 200})
			return fmt.Errorf("operator init failed: DSA operator not implemented in torch 2.7")
		}
		logMsg := fmt.Sprintf("[%s] step %3d/%d | loss %.4f | lr %.1e | %.0f tok/s",
			runID, s.step, totalSteps, s.loss, s.lr, s.throughput)
		if !e(Event{Kind: EventLogLine, Message: logMsg, DelayMs: 350}) {
			return ctx.Err()
		}
		if !e(Event{Kind: EventMetricUpdate, Step: s.step, TotalSteps: totalSteps, Loss: s.loss, LR: s.lr, Throughput: s.throughput, DelayMs: 50}) {
			return ctx.Err()
		}
	}

	if !e(Event{Kind: EventLogLine, Message: fmt.Sprintf("[%s] Training complete. Saving model...", runID), DelayMs: 600}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventTrainCompleted, Message: "Training completed. Final loss: 0.831", DelayMs: 300}) {
		return ctx.Err()
	}

	return nil
}

// ── Legacy Phase 2 functions (dual-lane Torch-NPU vs MindSpore-NPU demo) ────────
// These are preserved for the advanced demo scenario and are called
// directly from internal/app/train.go for Phase 2 flows.

// RunSetup plays the legacy demo setup phase for both Torch-NPU and MindSpore-NPU hosts.
func RunSetup(ctx context.Context, model, method string, sink func(Event)) error {
	e := func(ev Event) bool { return emit(ctx, sink, withDefaultRunID(ev)) }

	if !e(Event{Kind: EventMessage, Message: fmt.Sprintf("Preparing %s %s compare training. Running preflight checks...", model, method), DelayMs: 400}) {
		return ctx.Err()
	}

	// Local checks
	checks := []struct {
		name    string
		delayMs int
		detail  string
	}{
		{"local_repo", 600, "git:main @ a3f2c1e, clean working tree"},
		{"train_script", 500, fmt.Sprintf("scripts/train_%s.py (142 lines)", method)},
		{"base_model", 900, fmt.Sprintf("models/%s-7b (13.2 GB, sha256 verified)", model)},
	}
	for _, c := range checks {
		if !e(Event{Kind: EventCheckStarted, Check: c.name, DelayMs: 200}) {
			return ctx.Err()
		}
		if !e(Event{Kind: EventCheckPassed, Check: c.name, Message: c.detail, DelayMs: c.delayMs}) {
			return ctx.Err()
		}
	}

	// SSH — GPU host
	if !e(Event{Kind: EventCheckStarted, Check: "ssh_gpu", DelayMs: 200}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventHostConnecting, Host: "gpu-a100-0", Address: "10.0.2.10:22", DelayMs: 300}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventHostConnected, Host: "gpu-a100-0", Address: "10.0.2.10:22", DelayMs: 800}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventCheckPassed, Check: "ssh_gpu", Message: "gpu-a100-0 (key auth, latency 1ms)", DelayMs: 200}) {
		return ctx.Err()
	}

	// SSH — NPU host
	if !e(Event{Kind: EventCheckStarted, Check: "ssh_npu", DelayMs: 200}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventHostConnecting, Host: "npu-910b-0", Address: "10.0.1.15:22", DelayMs: 300}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventHostConnected, Host: "npu-910b-0", Address: "10.0.1.15:22", DelayMs: 1200}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventCheckPassed, Check: "ssh_npu", Message: "npu-910b-0 (key auth, latency 2ms)", DelayMs: 200}) {
		return ctx.Err()
	}

	// Remote workspace
	if !e(Event{Kind: EventCheckStarted, Check: "remote_workdir", DelayMs: 200}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "gpu", Message: "rsync: gpu-a100-0 incremental file list (3 files)", DelayMs: 300}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: "rsync: npu-910b-0 incremental file list (3 files)", DelayMs: 400}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventCheckPassed, Check: "remote_workdir", Message: "synced to both hosts", DelayMs: 500}) {
		return ctx.Err()
	}

	// Runtime env
	if !e(Event{Kind: EventCheckStarted, Check: "runtime_env", DelayMs: 200}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "gpu", Message: "GPU: Python 3.10.14, PyTorch 2.3.1, CUDA 12.4", DelayMs: 400}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: "NPU: Python 3.10.14, MindSpore 2.5.0, CANN 8.0.RC2", DelayMs: 400}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "gpu", Message: "GPU devices: 8x A100 (80 GB HBM each)", DelayMs: 300}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: "NPU devices: 8x Ascend 910B (64 GB HBM each)", DelayMs: 300}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventCheckPassed, Check: "runtime_env", Message: "Torch/NPU: PyTorch+Ascend, MindSpore/NPU: MindSpore/CANN", DelayMs: 400}) {
		return ctx.Err()
	}

	if !e(Event{Kind: EventReadyToStart, Message: "All preflight checks passed. Ready to launch compare training.", DelayMs: 300}) {
		return ctx.Err()
	}

	return nil
}

// ── Concurrent training: GPU healthy, NPU crashes ───────────

// RunConcurrentTraining starts both lanes. NPU crashes at launch, GPU completes.
func RunConcurrentTraining(ctx context.Context, model, method string, sink func(Event)) error {
	e := func(ev Event) bool { return emit(ctx, sink, withDefaultRunID(ev)) }

	gpuRunID := fmt.Sprintf("gpu-%s", time.Now().Format("0102-150405"))
	npuRunID := fmt.Sprintf("npu-%s", time.Now().Format("0102-150405"))

	// Start both lanes
	if !e(Event{Kind: EventTrainStarted, Lane: "gpu", Message: fmt.Sprintf("Torch/NPU baseline launched. Run: %s", gpuRunID), RunLabel: "run1", DelayMs: 300}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventTrainStarted, Lane: "npu", Message: fmt.Sprintf("MindSpore/NPU candidate launched. Run: %s", npuRunID), RunLabel: "run1", DelayMs: 200}) {
		return ctx.Err()
	}

	// Interleaved loading phase
	if !e(Event{Kind: EventLogLine, Lane: "gpu", Message: fmt.Sprintf("[%s] Loading base model %s-7b (PyTorch)...", gpuRunID, model), DelayMs: 500}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: fmt.Sprintf("[%s] Loading base model %s-7b (MindSpore)...", npuRunID, model), DelayMs: 400}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "gpu", Message: fmt.Sprintf("[%s] LoRA config: rank=16, alpha=32, dropout=0.05", gpuRunID), DelayMs: 300}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: fmt.Sprintf("[%s] LoRA config: rank=16, alpha=32, dropout=0.05", npuRunID), DelayMs: 300}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: fmt.Sprintf("[%s] Initializing NPU graph compilation...", npuRunID), DelayMs: 600}) {
		return ctx.Err()
	}

	// ── NPU crash ──
	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: fmt.Sprintf("[%s] ERROR: RuntimeError during graph compilation on Ascend 910B", npuRunID), DelayMs: 400}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: fmt.Sprintf("[%s] CANN Error: kernel 'FlashAttentionScore' not found in oplib (CANN 8.0.RC2)", npuRunID), DelayMs: 300}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: fmt.Sprintf("[%s] Expected CANN >= 8.0.RC3 for FlashAttention on Ascend 910B", npuRunID), DelayMs: 200}) {
		return ctx.Err()
	}

	if !e(Event{
		Kind:        EventTrainFailed,
		Lane:        "npu",
		Message:     "NPU candidate failed to launch. CANN kernel 'FlashAttentionScore' not found.",
		IssueType:   "runtime",
		IssueTitle:  "CANN kernel version mismatch",
		IssueDetail: "FlashAttentionScore op requires CANN >= 8.0.RC3 but host has CANN 8.0.RC2",
		DelayMs:     300,
	}) {
		return ctx.Err()
	}

	// ── GPU continues to completion ──
	if !e(Event{Kind: EventLogLine, Lane: "gpu", Message: fmt.Sprintf("[%s] Loading dataset: alpaca_gpt4_zh (52,002 samples)", gpuRunID), DelayMs: 400}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "gpu", Message: fmt.Sprintf("[%s] CUDA graph compilation... OK", gpuRunID), DelayMs: 600}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "gpu", Message: fmt.Sprintf("[%s] Begin training loop (300 steps)...", gpuRunID), DelayMs: 300}) {
		return ctx.Err()
	}

	type snap struct {
		step       int
		loss       float64
		lr         float64
		throughput float64
	}
	gpuSteps := []snap{
		{10, 2.891, 2.0e-4, 487.2},
		{25, 2.374, 2.0e-4, 512.8},
		{50, 1.956, 1.8e-4, 518.3},
		{75, 1.638, 1.5e-4, 516.7},
		{100, 1.401, 1.2e-4, 519.1},
		{150, 1.142, 8.0e-5, 517.4},
		{200, 0.964, 5.0e-5, 518.9},
		{250, 0.882, 3.0e-5, 519.2},
		{300, 0.831, 2.0e-5, 518.5},
	}

	totalSteps := 300
	for _, s := range gpuSteps {
		logMsg := fmt.Sprintf("[%s] step %3d/%d | loss %.4f | lr %.1e | %.0f tok/s",
			gpuRunID, s.step, totalSteps, s.loss, s.lr, s.throughput)
		if !e(Event{Kind: EventLogLine, Lane: "gpu", Message: logMsg, DelayMs: 350}) {
			return ctx.Err()
		}
		if !e(Event{Kind: EventMetricUpdate, Lane: "gpu", Step: s.step, TotalSteps: totalSteps, Loss: s.loss, LR: s.lr, Throughput: s.throughput, RunLabel: "run1", DelayMs: 50}) {
			return ctx.Err()
		}
	}

	if !e(Event{Kind: EventLogLine, Lane: "gpu", Message: fmt.Sprintf("[%s] Training complete. Saving model...", gpuRunID), DelayMs: 600}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventTrainCompleted, Lane: "gpu", Message: "Torch/NPU baseline completed. Final loss: 0.831", RunLabel: "run1", DelayMs: 300}) {
		return ctx.Err()
	}

	return nil
}

// ── Problem 1: NPU launch failure analysis & fix ─────────────

// RunNPUAnalysis diagnoses the runtime launch failure.
func RunNPUAnalysis(ctx context.Context, model, method string, sink func(Event)) error {
	e := func(ev Event) bool { return emit(ctx, sink, withDefaultRunID(ev)) }

	if !e(Event{Kind: EventAnalysisStarted, Message: "Analyzing NPU runtime failure...", DelayMs: 500}) {
		return ctx.Err()
	}

	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: "[analysis] Checking CANN op registry on npu-910b-0...", DelayMs: 700}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: "[analysis] CANN version: 8.0.RC2 (installed) vs 8.0.RC3 (required by FlashAttention)", DelayMs: 600}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: "[analysis] FlashAttentionScore op was added in CANN 8.0.RC3 changelog", DelayMs: 500}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: "[analysis] Fallback available: disable FlashAttention, use standard SDPA path", DelayMs: 400}) {
		return ctx.Err()
	}

	if !e(Event{
		Kind:         EventActionSuggested,
		Message:      "Suggested action: switch to SDPA fallback and relaunch.",
		IssueType:    "runtime",
		ActionID:     "runtime-fallback",
		ActionKind:   "change_env",
		ActionLabel:  "Switch to SDPA fallback",
		ActionSource: "failure-agent",
		FixSummary:   "Disable FlashAttention, use standard SDPA path",
		DelayMs:      250,
	}) {
		return ctx.Err()
	}

	diffText := `--- a/configs/qwen3_lora.yaml
+++ b/configs/qwen3_lora.yaml
@@ -15,7 +15,8 @@
 attention:
-  use_flash_attention: True
+  use_flash_attention: False
+  attn_implementation: "sdpa"  # fallback for CANN < 8.0.RC3`

	if !e(Event{
		Kind:        EventAnalysisReady,
		Message:     "Root cause identified. Runtime fix prepared.",
		IssueType:   "runtime",
		IssueTitle:  "CANN kernel version mismatch (FlashAttentionScore)",
		IssueDetail: "FlashAttentionScore kernel requires CANN >= 8.0.RC3 but host npu-910b-0 has CANN 8.0.RC2. Standard SDPA path is available as a compatible fallback.",
		Confidence:  "high",
		FixSummary:  "Disable FlashAttention, use standard SDPA path",
		DiffText:    diffText,
		DelayMs:     500,
	}) {
		return ctx.Err()
	}

	return nil
}

// RunNPUFixAndResume applies the runtime fix, runs NPU successfully, then evals.
func RunNPUFixAndResume(ctx context.Context, model, method string, sink func(Event)) error {
	e := func(ev Event) bool { return emit(ctx, sink, withDefaultRunID(ev)) }

	diffText := `--- a/configs/qwen3_lora.yaml
+++ b/configs/qwen3_lora.yaml
@@ -15,7 +15,8 @@
 attention:
-  use_flash_attention: True
+  use_flash_attention: False
+  attn_implementation: "sdpa"  # fallback for CANN < 8.0.RC3`

	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: "[fix] Patching configs/qwen3_lora.yaml...", DelayMs: 400}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: "[fix] Syncing patched config to npu-910b-0...", DelayMs: 500}) {
		return ctx.Err()
	}
	if !e(Event{
		Kind:       EventFixApplied,
		Lane:       "npu",
		Message:    "Runtime fix applied. Relaunching NPU training...",
		FixSummary: "Disable FlashAttention, use standard SDPA path",
		DiffText:   diffText,
		DelayMs:    500,
	}) {
		return ctx.Err()
	}

	// ── NPU successful training ──
	runID := fmt.Sprintf("npu-%s", time.Now().Format("0102-150405"))

	if !e(Event{Kind: EventTrainStarted, Lane: "npu", Message: fmt.Sprintf("MindSpore/NPU candidate relaunched with SDPA path. Run: %s", runID), RunLabel: "run1", DelayMs: 300}) {
		return ctx.Err()
	}

	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: fmt.Sprintf("[%s] Loading base model %s-7b (MindSpore)...", runID, model), DelayMs: 600}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: fmt.Sprintf("[%s] attention.use_flash_attention=False, attn_implementation=sdpa", runID), DelayMs: 200}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: fmt.Sprintf("[%s] NPU graph compilation... OK", runID), DelayMs: 800}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: fmt.Sprintf("[%s] Begin training loop (300 steps)...", runID), DelayMs: 300}) {
		return ctx.Err()
	}

	type snap struct {
		step       int
		loss       float64
		lr         float64
		throughput float64
	}
	npuSteps := []snap{
		{10, 2.847, 2.0e-4, 312.4},
		{25, 2.341, 2.0e-4, 338.7},
		{50, 1.924, 1.8e-4, 341.2},
		{75, 1.612, 1.5e-4, 339.8},
		{100, 1.387, 1.2e-4, 342.1},
		{150, 1.124, 8.0e-5, 340.6},
		{200, 0.982, 5.0e-5, 341.9},
		{250, 0.891, 3.0e-5, 342.3},
		{300, 0.847, 2.0e-5, 341.7},
	}

	totalSteps := 300
	for _, s := range npuSteps {
		logMsg := fmt.Sprintf("[%s] step %3d/%d | loss %.4f | lr %.1e | %.0f tok/s",
			runID, s.step, totalSteps, s.loss, s.lr, s.throughput)
		if !e(Event{Kind: EventLogLine, Lane: "npu", Message: logMsg, DelayMs: 400}) {
			return ctx.Err()
		}
		if !e(Event{Kind: EventMetricUpdate, Lane: "npu", Step: s.step, TotalSteps: totalSteps, Loss: s.loss, LR: s.lr, Throughput: s.throughput, RunLabel: "run1", DelayMs: 50}) {
			return ctx.Err()
		}
	}

	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: fmt.Sprintf("[%s] Training complete. Saving adapter weights...", runID), DelayMs: 600}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventTrainCompleted, Lane: "npu", Message: "MindSpore/NPU candidate completed. Final loss: 0.847", RunLabel: "run1", DelayMs: 300}) {
		return ctx.Err()
	}

	// ── Evaluation ──
	if !e(Event{Kind: EventEvalStarted, Message: "Evaluating both models on MMLU benchmark...", DelayMs: 800}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "gpu", Message: "[eval] Running MMLU (14,042 samples) on Torch/NPU model...", DelayMs: 600}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "gpu", Message: "[eval] Torch/NPU eval_acc: 71.4%", DelayMs: 800}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: "[eval] Running MMLU (14,042 samples) on MindSpore/NPU model...", DelayMs: 600}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: "[eval] MindSpore/NPU eval_acc: 54.8%", DelayMs: 600}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: "[eval] Accuracy drift: -16.6 pts", DelayMs: 400}) {
		return ctx.Err()
	}

	if !e(Event{
		Kind:         EventEvalCompleted,
		Message:      "Evaluation complete. Significant accuracy drift detected.",
		BaselineAcc:  71.4,
		CandidateAcc: 54.8,
		Drift:        -16.6,
		RunLabel:     "run1",
		DelayMs:      500,
	}) {
		return ctx.Err()
	}

	if !e(Event{
		Kind:         EventDriftDetected,
		Message:      "MindSpore/NPU accuracy is 16.6 pts lower than Torch/NPU baseline.",
		BaselineAcc:  71.4,
		CandidateAcc: 54.8,
		Drift:        -16.6,
		DelayMs:      300,
	}) {
		return ctx.Err()
	}

	return nil
}

// ── Problem 2: Accuracy drift analysis & fix ─────────────────

// RunDriftAnalysis diagnoses the attention mask / dtype mismatch.
func RunDriftAnalysis(ctx context.Context, model, method string, sink func(Event)) error {
	e := func(ev Event) bool { return emit(ctx, sink, withDefaultRunID(ev)) }

	if !e(Event{Kind: EventAnalysisStarted, Message: "Analyzing accuracy drift on Ascend eval path...", DelayMs: 500}) {
		return ctx.Err()
	}

	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: "[analysis] Inspecting attention op implementations on Ascend 910B...", DelayMs: 800}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: "[analysis] Comparing eval-path dtype behavior: Torch (NPU) vs MindSpore (NPU)...", DelayMs: 700}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: "[analysis] Found: attention mask cast to float16 before SDPA op on Ascend", DelayMs: 600}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: "[analysis] Torch/NPU path uses bool mask -> correct masking semantics", DelayMs: 400}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: "[analysis] NPU path casts mask to float16 -> precision loss in softmax", DelayMs: 400}) {
		return ctx.Err()
	}

	if !e(Event{
		Kind:         EventActionSuggested,
		Message:      "Suggested action: patch mask dtype and softmax precision before rerun.",
		IssueType:    "accuracy",
		ActionID:     "accuracy-mask-fix",
		ActionKind:   "apply_patch",
		ActionLabel:  "Patch attention mask + fp32 softmax",
		ActionSource: "acc-agent",
		FixSummary:   "Cast mask to bool, use fp32 softmax accumulation",
		DelayMs:      250,
	}) {
		return ctx.Err()
	}

	diffText := `--- a/models/qwen3/attention.py
+++ b/models/qwen3/attention.py
@@ -127,7 +127,7 @@
     def _attn(self, query, key, value, attention_mask=None):
-        attn_mask = attention_mask.astype(ms.float16)
+        attn_mask = attention_mask.astype(ms.bool_)

--- a/configs/qwen3_lora.yaml
+++ b/configs/qwen3_lora.yaml
@@ -18,5 +18,6 @@
 attention:
   use_flash_attention: False
   attn_implementation: "sdpa"
-  softmax_dtype: "float16"
+  softmax_dtype: "float32"
+  force_mask_bool: True  # align with GPU masking semantics`

	if !e(Event{
		Kind:        EventAnalysisReady,
		Message:     "Root cause identified. Accuracy fix prepared.",
		IssueType:   "accuracy",
		IssueTitle:  "Attention mask dtype mismatch on Ascend eval path",
		IssueDetail: "Attention mask is cast to float16 before SDPA op on Ascend. Torch/NPU path uses bool mask. The float16 cast causes precision loss in softmax.",
		Confidence:  "high",
		FixSummary:  "Cast mask to bool, use fp32 softmax accumulation",
		DiffText:    diffText,
		DelayMs:     500,
	}) {
		return ctx.Err()
	}

	return nil
}

// RunDriftFixAndRerun applies the accuracy fix, reruns NPU only, and verifies.
func RunDriftFixAndRerun(ctx context.Context, model, method string, sink func(Event)) error {
	e := func(ev Event) bool { return emit(ctx, sink, withDefaultRunID(ev)) }

	diffText := `--- a/models/qwen3/attention.py
+++ b/models/qwen3/attention.py
@@ -127,7 +127,7 @@
     def _attn(self, query, key, value, attention_mask=None):
-        attn_mask = attention_mask.astype(ms.float16)
+        attn_mask = attention_mask.astype(ms.bool_)

--- a/configs/qwen3_lora.yaml
+++ b/configs/qwen3_lora.yaml
@@ -18,5 +18,6 @@
 attention:
   use_flash_attention: False
   attn_implementation: "sdpa"
-  softmax_dtype: "float16"
+  softmax_dtype: "float32"
+  force_mask_bool: True  # align with GPU masking semantics`

	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: "[fix] Patching models/qwen3/attention.py...", DelayMs: 400}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: "[fix] Patching configs/qwen3_lora.yaml...", DelayMs: 300}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: "[fix] Syncing patched files to npu-910b-0...", DelayMs: 500}) {
		return ctx.Err()
	}

	if !e(Event{
		Kind:       EventFixApplied,
		Lane:       "npu",
		Message:    "Accuracy fix applied. Starting NPU rerun...",
		FixSummary: "Cast mask to bool, fp32 softmax, align GPU semantics",
		DiffText:   diffText,
		DelayMs:    500,
	}) {
		return ctx.Err()
	}

	// ── NPU Run 2 ──
	runID := fmt.Sprintf("npu-rerun-%s", time.Now().Format("0102-150405"))

	if !e(Event{Kind: EventRerunStarted, Lane: "npu", Message: fmt.Sprintf("[Run 2] NPU rerun with accuracy fix. Run: %s", runID), RunLabel: "run2", DelayMs: 300}) {
		return ctx.Err()
	}

	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: fmt.Sprintf("[%s] Loading patched model %s-7b...", runID, model), DelayMs: 600}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: fmt.Sprintf("[%s] attention.softmax_dtype=float32, force_mask_bool=True", runID), DelayMs: 300}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: fmt.Sprintf("[%s] Begin training loop (300 steps)...", runID), DelayMs: 400}) {
		return ctx.Err()
	}

	type snap struct {
		step       int
		loss       float64
		lr         float64
		throughput float64
	}
	npuSteps := []snap{
		{25, 2.398, 2.0e-4, 328.1},
		{75, 1.641, 1.5e-4, 331.4},
		{150, 1.092, 8.0e-5, 330.8},
		{225, 0.891, 4.0e-5, 331.2},
		{300, 0.812, 2.0e-5, 330.9},
	}

	totalSteps := 300
	for _, s := range npuSteps {
		logMsg := fmt.Sprintf("[%s] step %3d/%d | loss %.4f | lr %.1e | %.0f tok/s",
			runID, s.step, totalSteps, s.loss, s.lr, s.throughput)
		if !e(Event{Kind: EventLogLine, Lane: "npu", Message: logMsg, DelayMs: 400}) {
			return ctx.Err()
		}
		if !e(Event{Kind: EventMetricUpdate, Lane: "npu", Step: s.step, TotalSteps: totalSteps, Loss: s.loss, LR: s.lr, Throughput: s.throughput, RunLabel: "run2", DelayMs: 50}) {
			return ctx.Err()
		}
	}

	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: fmt.Sprintf("[%s] Training complete. Saving adapter weights...", runID), DelayMs: 600}) {
		return ctx.Err()
	}

	// ── Final evaluation ──
	if !e(Event{Kind: EventEvalStarted, Message: "[Run 2] Running final evaluation...", DelayMs: 500}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "gpu", Message: "[eval] Torch/NPU baseline eval_acc: 71.4% (cached)", DelayMs: 400}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: "[eval] Running MMLU on NPU model (run 2)...", DelayMs: 600}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: "[eval] MindSpore/NPU eval_acc: 69.8%", DelayMs: 500}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Lane: "npu", Message: "[eval] Accuracy drift: -1.6 pts (improved from -16.6 pts)", DelayMs: 400}) {
		return ctx.Err()
	}

	if !e(Event{
		Kind:         EventEvalCompleted,
		Message:      "[Run 2] Evaluation complete.",
		BaselineAcc:  71.4,
		CandidateAcc: 69.8,
		Drift:        -1.6,
		RunLabel:     "run2",
		DelayMs:      300,
	}) {
		return ctx.Err()
	}

	if !e(Event{
		Kind:         EventVerificationPassed,
		Message:      "Fix verified. NPU accuracy recovered from 54.8% to 69.8%. Drift: -16.6 pts -> -1.6 pts.",
		BaselineAcc:  71.4,
		CandidateAcc: 69.8,
		Drift:        -1.6,
		DelayMs:      500,
	}) {
		return ctx.Err()
	}

	return nil
}

// RunPerformanceAnalysis is a stub demo flow for performance recovery.
func RunPerformanceAnalysis(ctx context.Context, model, method string, sink func(Event)) error {
	e := func(ev Event) bool { return emit(ctx, sink, withDefaultRunID(ev)) }

	if !e(Event{Kind: EventAnalysisStarted, Message: "Analyzing throughput bottleneck on target...", IssueType: "performance", DelayMs: 400}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Message: "[analysis] Observed dataloader workers underutilized on remote host", DelayMs: 500}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Message: "[analysis] Gradient checkpointing is enabled on a small memory-safe batch", DelayMs: 400}) {
		return ctx.Err()
	}
	if !e(Event{
		Kind:         EventActionSuggested,
		Message:      "Suggested action: increase dataloader workers and disable checkpointing for this target.",
		IssueType:    "performance",
		ActionID:     "perf-workers-tuning",
		ActionKind:   "change_config",
		ActionLabel:  "Tune dataloader + checkpointing",
		ActionSource: "perf-agent",
		FixSummary:   "Increase loader workers, disable unnecessary checkpointing",
		DelayMs:      250,
	}) {
		return ctx.Err()
	}
	if !e(Event{
		Kind:         EventAnalysisReady,
		Message:      "Performance bottleneck identified. Throughput tuning is ready.",
		IssueType:    "performance",
		IssueTitle:   "Input pipeline bottleneck",
		IssueDetail:  "The target is spending excessive time waiting on data loading and recomputation.",
		FixSummary:   "Increase loader workers, disable unnecessary checkpointing",
		ActionID:     "perf-workers-tuning",
		ActionKind:   "change_config",
		ActionLabel:  "Apply throughput tuning",
		ActionSource: "perf-agent",
		DelayMs:      300,
	}) {
		return ctx.Err()
	}
	return nil
}

// RunPerformanceFixAndRerun is a stub demo flow for performance tuning.
func RunPerformanceFixAndRerun(ctx context.Context, model, method string, sink func(Event)) error {
	e := func(ev Event) bool { return emit(ctx, sink, withDefaultRunID(ev)) }

	if !e(Event{Kind: EventFixApplied, Message: "Performance tuning applied. Restarting run...", IssueType: "performance", FixSummary: "Increase dataloader workers, disable checkpointing", ActionID: "perf-workers-tuning", ActionKind: "change_config", ActionLabel: "Apply throughput tuning", ActionSource: "perf-agent", DelayMs: 300}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventRerunStarted, Message: "Rerunning after performance tuning...", RunLabel: "perf-rerun", DelayMs: 300}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventMetricUpdate, Step: 40, TotalSteps: 300, Loss: 2.102, LR: 1.9e-4, Throughput: 690, RunLabel: "perf-rerun", DelayMs: 200}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventMetricUpdate, Step: 120, TotalSteps: 300, Loss: 1.312, LR: 1.1e-4, Throughput: 712, RunLabel: "perf-rerun", DelayMs: 200}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventTrainCompleted, Message: "Performance rerun completed. Throughput improved to ~710 tok/s.", RunLabel: "perf-rerun", DelayMs: 300}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventVerificationPassed, Message: "Performance verified. Accuracy remains stable while throughput improved.", IssueType: "performance", DelayMs: 300}) {
		return ctx.Err()
	}
	return nil
}

// RunTrickIteration is a stub demo flow for applying a new training trick.
func RunTrickIteration(ctx context.Context, model, method, trick string, sink func(Event)) error {
	e := func(ev Event) bool { return emit(ctx, sink, withDefaultRunID(ev)) }
	if trick == "" {
		trick = "mhc"
	}
	if !e(Event{Kind: EventFixApplied, Message: fmt.Sprintf("Applying new trick: %s", trick), ActionID: "trick-" + trick, ActionKind: "apply_patch", ActionLabel: "Apply trick " + strings.ToUpper(trick), ActionSource: "trick-flow", FixSummary: "Patch train config with new trick", DelayMs: 300}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventRerunStarted, Message: fmt.Sprintf("Rerunning with trick %s enabled...", trick), RunLabel: "trick-rerun", DelayMs: 300}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Message: fmt.Sprintf("[trick] %s enabled in train config", trick), DelayMs: 300}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventDriftDetected, Message: fmt.Sprintf("After enabling %s, validation accuracy regressed. Accuracy analysis recommended.", trick), IssueType: "accuracy", DelayMs: 300}) {
		return ctx.Err()
	}
	return nil
}
