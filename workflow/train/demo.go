package train

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	itrain "github.com/mindspore-lab/mindspore-cli/internal/train"
)

// demoSpeed controls the delay multiplier for demo playback.
// Set MS_DEMO_SPEED=10 to run 10x faster (e.g. 2000ms → 200ms).
// Default is 1 (normal speed).
//
// Read it dynamically so tests can accelerate playback with t.Setenv.
func demoSpeed() float64 {
	if s := os.Getenv("MS_DEMO_SPEED"); s != "" {
		if v, err := strconv.ParseFloat(s, 64); err == nil && v > 0 {
			return v
		}
	}
	return 1
}

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
// Delays are divided by demoSpeed (MS_DEMO_SPEED env var).
func emit(ctx context.Context, sink func(Event), ev Event) bool {
	if ev.DelayMs > 0 {
		delay := time.Duration(float64(ev.DelayMs)/demoSpeed()) * time.Millisecond
		select {
		case <-ctx.Done():
			return false
		case <-time.After(delay):
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
	return d.runTraining(ctx, session.Model, session.Method, sink, session.FailAtStep, session.PerfFixed, session.TrickApplied)
}

// runTraining plays demo training: loading, log lines, metrics, completion.
// If failStep > 0, training crashes at that step with a DSA operator error.
// If perfFixed is true, throughput is ~10% higher (fused adam kernel).
// If trickApplied is true, loss converges faster (MHC contrastive loss).
func (d *DemoBackend) runTraining(ctx context.Context, model, method string, sink func(Event), failStep int, perfFixed, trickApplied bool) error {
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
	if perfFixed {
		if !e(Event{Kind: EventLogLine, Message: fmt.Sprintf("[%s] Optimizer: fused_adam (custom kernel)", runID), DelayMs: 300}) {
			return ctx.Err()
		}
	}
	if trickApplied {
		if !e(Event{Kind: EventLogLine, Message: fmt.Sprintf("[%s] Algo-feature: MHC (multi-head contrastive) enabled", runID), DelayMs: 300}) {
			return ctx.Err()
		}
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

	// Select step data based on applied optimizations.
	// Base throughput ~518, perfFixed ~571, trickApplied lowers final loss.
	var steps []snap
	switch {
	case perfFixed && trickApplied:
		steps = []snap{
			{10, 2.812, 2.0e-4, 536.1},
			{25, 2.241, 2.0e-4, 564.3},
			{50, 1.823, 1.8e-4, 570.5},
			{75, 1.487, 1.5e-4, 568.2},
			{100, 1.248, 1.2e-4, 571.4},
			{150, 0.981, 8.0e-5, 569.8},
			{200, 0.824, 5.0e-5, 571.1},
			{250, 0.762, 3.0e-5, 571.5},
			{300, 0.718, 2.0e-5, 570.7},
		}
	case perfFixed:
		steps = []snap{
			{10, 2.891, 2.0e-4, 536.1},
			{25, 2.374, 2.0e-4, 564.3},
			{50, 1.956, 1.8e-4, 570.5},
			{75, 1.638, 1.5e-4, 568.2},
			{100, 1.401, 1.2e-4, 571.4},
			{150, 1.142, 8.0e-5, 569.8},
			{200, 0.964, 5.0e-5, 571.1},
			{250, 0.882, 3.0e-5, 571.5},
			{300, 0.831, 2.0e-5, 570.7},
		}
	case trickApplied:
		steps = []snap{
			{10, 2.812, 2.0e-4, 487.2},
			{25, 2.241, 2.0e-4, 512.8},
			{50, 1.823, 1.8e-4, 518.3},
			{75, 1.487, 1.5e-4, 516.7},
			{100, 1.248, 1.2e-4, 519.1},
			{150, 0.981, 8.0e-5, 517.4},
			{200, 0.824, 5.0e-5, 518.9},
			{250, 0.762, 3.0e-5, 519.2},
			{300, 0.718, 2.0e-5, 518.5},
		}
	default:
		steps = []snap{
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
	if !e(Event{Kind: EventTrainCompleted, Message: "training completed. Final loss: 0.831", DelayMs: 300}) {
		return ctx.Err()
	}

	return nil
}

// RunSingleLaneEval simulates post-training evaluation.
// When driftFixed is false, it detects accuracy drift (first run).
// When driftFixed is true, evaluation passes (after accuracy fix).
// When perfFixed is true, a perf comparison summary is included.
// When trickApplied is true, accuracy exceeds baseline.
func RunSingleLaneEval(ctx context.Context, model, method string, driftFixed, perfFixed, trickApplied bool, sink func(Event)) error {
	e := func(ev Event) bool { return emit(ctx, sink, withDefaultRunID(ev)) }

	if !e(Event{Kind: EventEvalStarted, Message: "evaluating model on ceval-valid benchmark...", DelayMs: 800}) {
		return ctx.Err()
	}

	evalRunID := fmt.Sprintf("eval-%s", time.Now().Format("0102-150405"))

	if !e(Event{Kind: EventLogLine, Message: fmt.Sprintf("[%s] Running ceval-valid (1,346 samples)...", evalRunID), DelayMs: 600}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Message: fmt.Sprintf("[%s] Baseline accuracy (pre-train): 72.1%%", evalRunID), DelayMs: 500}) {
		return ctx.Err()
	}

	if driftFixed {
		// After accuracy fix: eval passes. Trick improves accuracy further.
		candidateAcc := 71.8
		driftPts := -0.3
		if trickApplied {
			candidateAcc = 73.6
			driftPts = 1.5
		}

		if !e(Event{Kind: EventLogLine, Message: fmt.Sprintf("[%s] Trained model accuracy: %.1f%%", evalRunID, candidateAcc), DelayMs: 500}) {
			return ctx.Err()
		}
		if trickApplied {
			if !e(Event{Kind: EventLogLine, Message: fmt.Sprintf("[%s] Accuracy gain: +%.1f pts over baseline (MHC algo-feature)", evalRunID, driftPts), DelayMs: 400}) {
				return ctx.Err()
			}
		} else {
			if !e(Event{Kind: EventLogLine, Message: fmt.Sprintf("[%s] Accuracy drift: %.1f pts (within tolerance)", evalRunID, driftPts), DelayMs: 400}) {
				return ctx.Err()
			}
		}
		if !e(Event{
			Kind:         EventEvalCompleted,
			Message:      "evaluation complete. Accuracy within tolerance.",
			BaselineAcc:  72.1,
			CandidateAcc: candidateAcc,
			Drift:        driftPts,
			DelayMs:      500,
		}) {
			return ctx.Err()
		}
		if perfFixed {
			if !e(Event{Kind: EventLogLine, Message: fmt.Sprintf("[%s] Perf: avg step latency 250ms (was 400ms), reduced 150ms with fused_adam", evalRunID), DelayMs: 500}) {
				return ctx.Err()
			}
		}

		// Verification message based on what was applied
		var verifyMsg string
		switch {
		case trickApplied && perfFixed:
			verifyMsg = fmt.Sprintf("Verified. MHC algo-feature boosted accuracy to %.1f%% (+%.1f pts). Latency reduced 150ms with fused_adam.", candidateAcc, driftPts)
		case trickApplied:
			verifyMsg = fmt.Sprintf("Verified. MHC algo-feature boosted accuracy to %.1f%% (+%.1f pts over 72.1%% baseline).", candidateAcc, driftPts)
		case perfFixed:
			verifyMsg = fmt.Sprintf("Verified. Latency reduced 150ms (400ms → 250ms) with fused_adam. Accuracy stable at %.1f%%.", candidateAcc)
		default:
			verifyMsg = fmt.Sprintf("Accuracy verified. Model scores %.1f%% vs 72.1%% baseline on ceval-valid. Drift: %.1f pts.", candidateAcc, driftPts)
		}
		if !e(Event{
			Kind:         EventVerificationPassed,
			ActionSource: "observer",
			Message:      verifyMsg,
			BaselineAcc:  72.1,
			CandidateAcc: candidateAcc,
			Drift:        driftPts,
			DelayMs:      500,
		}) {
			return ctx.Err()
		}
		return nil
	}

	// First run: detect drift
	if !e(Event{Kind: EventLogLine, Message: fmt.Sprintf("[%s] Trained model accuracy: 55.3%%", evalRunID), DelayMs: 500}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Message: fmt.Sprintf("[%s] Accuracy drift: -16.8 pts", evalRunID), DelayMs: 400}) {
		return ctx.Err()
	}

	if !e(Event{
		Kind:         EventEvalCompleted,
		Message:      "evaluation complete. Significant accuracy drift detected.",
		BaselineAcc:  72.1,
		CandidateAcc: 55.3,
		Drift:        -16.8,
		DelayMs:      500,
	}) {
		return ctx.Err()
	}

	if !e(Event{
		Kind:         EventDriftDetected,
		Message:      "accuracy regressed by 16.8 pts (72.1% → 55.3%) on ceval-valid.",
		BaselineAcc:  72.1,
		CandidateAcc: 55.3,
		Drift:        -16.8,
		IssueType:    "accuracy",
		IssueTitle:   "Accuracy regression on ceval-valid benchmark",
		IssueDetail:  "Trained model scores 55.3% vs 72.1% baseline on ceval-valid (1,346 samples). Regression of 16.8 pts suggests attention mask or dtype issue.",
		DelayMs:      300,
	}) {
		return ctx.Err()
	}

	return nil
}

// AnalyzeSingleLaneDrift diagnoses accuracy drift for the single-lane flow.
func AnalyzeSingleLaneDrift(ctx context.Context, model, method string, sink func(Event)) error {
	e := func(ev Event) bool { return emit(ctx, sink, withDefaultRunID(ev)) }

	if !e(Event{Kind: EventAnalysisStarted, ActionSource: "acc-agent", Message: "analyzing accuracy drift...", DelayMs: 500}) {
		return ctx.Err()
	}

	if !e(Event{Kind: EventMessage, ActionSource: "acc-agent", Message: "cause → fp16 dtype drift for softmax accumulation", DelayMs: 800}) {
		return ctx.Err()
	}

	if !e(Event{
		Kind:         EventActionSuggested,
		IssueType:    "accuracy",
		IssueID:      "accuracy-softmax-drift",
		ActionID:     "fix-softmax-dtype",
		ActionKind:   "apply_patch",
		ActionLabel:  "apply fix",
		ActionSource: "acc-agent",
		Message:      "suggested action: patch softmax dtype to bf16.",
		DelayMs:      500,
	}) {
		return ctx.Err()
	}

	diffText := `--- a/configs/qwen3_lora.yaml
+++ b/configs/qwen3_lora.yaml
@@ -18,5 +18,6 @@
 attention:
-  softmax_dtype: "float16"
+  softmax_dtype: "bfloat16"
+  force_upcast_softmax: True  # prevent fp16 accumulation drift`

	if !e(Event{
		Kind:        EventAnalysisReady,
		IssueType:   "accuracy",
		IssueID:     "accuracy-softmax-drift",
		IssueTitle:  "fp16 softmax accumulation drift",
		IssueDetail: "Softmax accumulation in fp16 causes precision loss during eval. Switching to bf16 preserves accuracy.",
		Confidence:  "high",
		FixSummary:  "Patch softmax dtype to bf16",
		DiffText:    diffText,
		ActionID:    "fix-softmax-dtype",
		ActionKind:  "apply_patch",
		ActionLabel: "apply fix",
		Message:     "analysis complete. Ready to apply fix.",
		DelayMs:     400,
	}) {
		return ctx.Err()
	}

	return nil
}

// ApplySingleLaneDriftFix applies the accuracy fix (no rerun — user clicks rerun).
func ApplySingleLaneDriftFix(ctx context.Context, model, method string, sink func(Event)) error {
	e := func(ev Event) bool { return emit(ctx, sink, withDefaultRunID(ev)) }

	if !e(Event{
		Kind:         EventActionApplied,
		IssueType:    "accuracy",
		ActionID:     "fix-softmax-dtype",
		ActionKind:   "apply_patch",
		ActionSource: "acc-agent",
		Message:      "adding patch to softmax with bf16...",
		DelayMs:      2000,
	}) {
		return ctx.Err()
	}

	if !e(Event{
		Kind:         EventFixApplied,
		IssueType:    "accuracy",
		ActionSource: "acc-agent",
		FixSummary:   "Softmax dtype patched to bf16",
		Message:      "patch added. please rerun experiment.",
		DelayMs:      1500,
	}) {
		return ctx.Err()
	}

	return nil
}

// ── Story 5: Single-lane algo-feature iteration ──────────────

// RunSingleLaneAlgoFeature applies an algo-feature (e.g. MHC) and prepares for rerun.
func RunSingleLaneAlgoFeature(ctx context.Context, model, method, feature string, sink func(Event)) error {
	e := func(ev Event) bool { return emit(ctx, sink, withDefaultRunID(ev)) }

	if feature == "" {
		feature = "mhc"
	}

	featureInfo := map[string]struct{ name, desc string }{
		"mhc":          {"MHC", "inject contrastive loss on attention heads to improve generalization."},
		"flash-attn":   {"Flash Attention", "replace standard attention with IO-aware fused kernel for faster forward pass."},
		"sparse-attn":  {"Sparse Attention", "apply block-sparse attention pattern to reduce quadratic complexity."},
		"lora-plus":    {"LoRA+", "use differential learning rates for LoRA A/B matrices to improve convergence."},
		"galore":       {"GaLore", "apply gradient low-rank projection to reduce optimizer memory footprint."},
		"ddpm-noise":   {"DDPM Noise Schedule", "apply denoising diffusion noise scheduling for improved sample quality."},
		"dpo":          {"DPO", "add direct preference optimization loss for human alignment."},
		"rope-scaling": {"RoPE Scaling", "extend context length via rotary position embedding interpolation."},
		"moe-routing":  {"MoE Routing", "enable mixture-of-experts with top-k dynamic routing."},
	}
	info := featureInfo[feature]
	if info.name == "" {
		info.name = feature
		info.desc = "applying algorithm optimization."
	}

	if !e(Event{Kind: EventMessage, ActionSource: "algo-agent", Message: fmt.Sprintf("evaluating %s for %s %s...", info.name, model, method), DelayMs: 800}) {
		return ctx.Err()
	}

	if !e(Event{Kind: EventMessage, ActionSource: "algo-agent", Message: fmt.Sprintf("%s selected. %s", info.name, info.desc), DelayMs: 800}) {
		return ctx.Err()
	}

	if !e(Event{Kind: EventMessage, ActionSource: "algo-agent", Message: "patching training config and model forward pass...", DelayMs: 600}) {
		return ctx.Err()
	}

	type diffEntry struct {
		msg     string
		delayMs int
	}
	var diffLines []diffEntry

	switch feature {
	case "mhc":
		diffLines = []diffEntry{
			{"--- a/configs/qwen3_lora.yaml", 300},
			{"+++ b/configs/qwen3_lora.yaml", 200},
			{"@@ -18,6 +18,9 @@", 200},
			{" loss:", 150},
			{"-  type: \"cross_entropy\"", 200},
			{"+  type: \"mhc_loss\"", 200},
			{"+  mhc_heads: 8", 150},
			{"+  contrastive_weight: 0.15", 150},
			{"+  temperature: 0.07", 150},
			{"", 400},
			{"--- a/models/qwen3_lora_forward.py", 300},
			{"+++ b/models/qwen3_lora_forward.py", 200},
			{"@@ -42,4 +42,12 @@", 200},
			{" def forward(self, input_ids, attention_mask):", 150},
			{"     hidden = self.base_model(input_ids, attention_mask)", 150},
			{"+    # MHC: multi-head contrastive branch", 200},
			{"+    head_outputs = self.mhc_project(hidden)  # [B, S, num_heads, D]", 200},
			{"+    contrastive_loss = mhc_contrastive(", 150},
			{"+        head_outputs, temperature=0.07", 150},
			{"+    )", 100},
			{"+    loss = ce_loss + 0.15 * contrastive_loss", 200},
			{"", 300},
			{"2 files changed, 9 insertions(+), 1 deletion(-)", 500},
		}
	case "flash-attn":
		diffLines = []diffEntry{
			{"--- a/models/qwen3_lora_forward.py", 300},
			{"+++ b/models/qwen3_lora_forward.py", 200},
			{"@@ -1,4 +1,5 @@", 200},
			{" import torch", 150},
			{"+from flash_attn import flash_attn_func", 200},
			{"", 300},
			{"@@ -28,7 +29,10 @@", 200},
			{" class Qwen3LoRAModel(nn.Module):", 150},
			{"     def _attention(self, q, k, v, mask):", 150},
			{"-        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)", 200},
			{"-        scores = scores.masked_fill(mask == 0, -1e9)", 150},
			{"-        attn = torch.softmax(scores, dim=-1)", 150},
			{"-        return torch.matmul(attn, v)", 150},
			{"+        # Flash Attention v2: fused IO-aware kernel", 200},
			{"+        return flash_attn_func(", 200},
			{"+            q, k, v,", 150},
			{"+            causal=True,", 150},
			{"+            softmax_scale=1.0 / math.sqrt(self.head_dim),", 150},
			{"+        )", 100},
			{"", 300},
			{"1 file changed, 6 insertions(+), 4 deletions(-)", 500},
		}
	case "sparse-attn":
		diffLines = []diffEntry{
			{"--- a/configs/qwen3_lora.yaml", 300},
			{"+++ b/configs/qwen3_lora.yaml", 200},
			{"@@ -12,3 +12,6 @@", 200},
			{" attention:", 150},
			{"-  type: \"full\"", 200},
			{"+  type: \"block_sparse\"", 200},
			{"+  block_size: 64", 150},
			{"+  num_global_tokens: 16", 150},
			{"+  sparsity_ratio: 0.75", 150},
			{"", 400},
			{"--- a/models/qwen3_lora_forward.py", 300},
			{"+++ b/models/qwen3_lora_forward.py", 200},
			{"@@ -30,5 +30,12 @@", 200},
			{"     def _attention(self, q, k, v, mask):", 150},
			{"+        # Block-sparse attention: only attend within local blocks", 200},
			{"+        sparse_mask = build_block_sparse_mask(", 200},
			{"+            seq_len=q.size(-2),", 150},
			{"+            block_size=64,", 150},
			{"+            global_tokens=16,", 150},
			{"+        )", 100},
			{"+        scores = torch.matmul(q, k.transpose(-2, -1)) * sparse_mask", 200},
			{"", 300},
			{"2 files changed, 9 insertions(+), 1 deletion(-)", 500},
		}
	case "lora-plus":
		diffLines = []diffEntry{
			{"--- a/configs/qwen3_lora.yaml", 300},
			{"+++ b/configs/qwen3_lora.yaml", 200},
			{"@@ -25,4 +25,7 @@", 200},
			{" lora:", 150},
			{"   rank: 16", 150},
			{"   alpha: 32", 150},
			{"-  lr: 2e-4", 200},
			{"+  # LoRA+: differential LR for A/B matrices", 200},
			{"+  lr_A: 2e-4", 150},
			{"+  lr_B: 2e-5       # 10x smaller for B matrix", 150},
			{"+  lr_scheduler: \"cosine_with_warmup\"", 150},
			{"", 400},
			{"--- a/models/lora_optimizer.py", 300},
			{"+++ b/models/lora_optimizer.py", 200},
			{"@@ -15,4 +15,11 @@", 200},
			{" def build_optimizer(model, config):", 150},
			{"-    return AdamW(model.parameters(), lr=config.lr)", 200},
			{"+    # LoRA+ param groups with differential learning rates", 200},
			{"+    param_groups = [", 200},
			{"+        {\"params\": lora_A_params(model), \"lr\": config.lr_A},", 150},
			{"+        {\"params\": lora_B_params(model), \"lr\": config.lr_B},", 150},
			{"+        {\"params\": non_lora_params(model), \"lr\": config.lr_A * 0.1},", 150},
			{"+    ]", 100},
			{"+    return AdamW(param_groups)", 150},
			{"", 300},
			{"2 files changed, 10 insertions(+), 2 deletions(-)", 500},
		}
	case "galore":
		diffLines = []diffEntry{
			{"--- a/configs/qwen3_lora.yaml", 300},
			{"+++ b/configs/qwen3_lora.yaml", 200},
			{"@@ -22,3 +22,6 @@", 200},
			{" optimizer:", 150},
			{"   type: \"adam\"", 150},
			{"+  galore_enabled: True", 200},
			{"+  galore_rank: 128", 150},
			{"+  galore_update_proj_gap: 200  # re-project every 200 steps", 150},
			{"", 400},
			{"--- a/models/galore_wrapper.py", 300},
			{"+++ b/models/galore_wrapper.py", 200},
			{"@@ -0,0 +1,18 @@", 200},
			{"+import torch", 150},
			{"+from galore_torch import GaLoreAdamW", 200},
			{"+", 100},
			{"+def build_galore_optimizer(model, config):", 200},
			{"+    target_modules = [\"attn.q_proj\", \"attn.k_proj\", \"attn.v_proj\", \"mlp.gate\"]", 200},
			{"+    param_groups = []", 150},
			{"+    for name, param in model.named_parameters():", 150},
			{"+        if any(t in name for t in target_modules):", 150},
			{"+            param_groups.append({", 150},
			{"+                \"params\": [param],", 150},
			{"+                \"rank\": config.galore_rank,", 150},
			{"+                \"update_proj_gap\": config.galore_update_proj_gap,", 150},
			{"+            })", 100},
			{"+    return GaLoreAdamW(param_groups, lr=config.lr)", 200},
			{"", 300},
			{"2 files changed, 21 insertions(+)", 500},
		}
	case "ddpm-noise":
		diffLines = []diffEntry{
			{"--- a/configs/qwen3_lora.yaml", 300},
			{"+++ b/configs/qwen3_lora.yaml", 200},
			{"@@ -35,2 +35,6 @@", 200},
			{" training:", 150},
			{"+  noise_schedule: \"cosine\"", 200},
			{"+  timesteps: 1000", 150},
			{"+  beta_start: 0.0001", 150},
			{"+  beta_end: 0.02", 150},
			{"", 400},
			{"--- a/models/noise_scheduler.py", 300},
			{"+++ b/models/noise_scheduler.py", 200},
			{"@@ -0,0 +1,16 @@", 200},
			{"+import torch", 150},
			{"+import math", 150},
			{"+", 100},
			{"+class DDPMScheduler:", 200},
			{"+    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, schedule=\"cosine\"):", 200},
			{"+        self.timesteps = timesteps", 150},
			{"+        if schedule == \"cosine\":", 150},
			{"+            steps = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps", 150},
			{"+            alpha_bar = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2", 200},
			{"+            betas = torch.clamp(1 - alpha_bar[1:] / alpha_bar[:-1], max=0.999)", 150},
			{"+        else:", 100},
			{"+            betas = torch.linspace(beta_start, beta_end, timesteps)", 150},
			{"+        self.betas = betas.float()", 150},
			{"+        self.alphas_cumprod = torch.cumprod(1.0 - self.betas, dim=0)", 200},
			{"", 300},
			{"2 files changed, 20 insertions(+)", 500},
		}
	case "dpo":
		diffLines = []diffEntry{
			{"--- a/configs/qwen3_lora.yaml", 300},
			{"+++ b/configs/qwen3_lora.yaml", 200},
			{"@@ -18,4 +18,7 @@", 200},
			{" loss:", 150},
			{"-  type: \"cross_entropy\"", 200},
			{"+  type: \"dpo_loss\"", 200},
			{"+  dpo_beta: 0.1", 150},
			{"+  ref_model: \"qwen3-base-frozen\"", 150},
			{"+  label_smoothing: 0.0", 150},
			{"", 400},
			{"--- a/models/dpo_trainer.py", 300},
			{"+++ b/models/dpo_trainer.py", 200},
			{"@@ -0,0 +1,18 @@", 200},
			{"+import torch", 150},
			{"+import torch.nn.functional as F", 150},
			{"+", 100},
			{"+def dpo_loss(policy_logps_chosen, policy_logps_rejected,", 200},
			{"+             ref_logps_chosen, ref_logps_rejected, beta=0.1):", 200},
			{"+    \"\"\"Direct Preference Optimization loss.\"\"\"", 150},
			{"+    pi_ratio = policy_logps_chosen - policy_logps_rejected", 150},
			{"+    ref_ratio = ref_logps_chosen - ref_logps_rejected", 150},
			{"+    logits = beta * (pi_ratio - ref_ratio)", 200},
			{"+    return -F.logsigmoid(logits).mean()", 200},
			{"", 300},
			{"2 files changed, 14 insertions(+), 1 deletion(-)", 500},
		}
	case "rope-scaling":
		diffLines = []diffEntry{
			{"--- a/configs/qwen3_lora.yaml", 300},
			{"+++ b/configs/qwen3_lora.yaml", 200},
			{"@@ -10,2 +10,6 @@", 200},
			{" model:", 150},
			{"+  position_encoding:", 200},
			{"+    type: \"rope\"", 150},
			{"+    rope_scaling:", 150},
			{"+      type: \"dynamic_ntk\"", 150},
			{"+      factor: 4.0  # extend context 4x", 150},
			{"", 400},
			{"--- a/models/rope_extend.py", 300},
			{"+++ b/models/rope_extend.py", 200},
			{"@@ -0,0 +1,16 @@", 200},
			{"+import torch", 150},
			{"+import math", 150},
			{"+", 100},
			{"+def apply_rope_scaling(freqs, seq_len, factor=4.0, base=10000):", 200},
			{"+    \"\"\"Dynamic NTK-aware RoPE scaling for context extension.\"\"\"", 200},
			{"+    if seq_len > freqs.shape[0]:", 150},
			{"+        base_scaled = base * (factor * seq_len / freqs.shape[0]", 150},
			{"+                              - (factor - 1)) ** (freqs.shape[-1] / (freqs.shape[-1] - 2))", 200},
			{"+        inv_freq = 1.0 / (base_scaled ** (torch.arange(0, freqs.shape[-1], 2).float() / freqs.shape[-1]))", 200},
			{"+        t = torch.arange(seq_len, device=freqs.device).float()", 150},
			{"+        freqs = torch.outer(t, inv_freq)", 150},
			{"+    return torch.cat((freqs, freqs), dim=-1)", 150},
			{"", 300},
			{"2 files changed, 17 insertions(+)", 500},
		}
	case "moe-routing":
		diffLines = []diffEntry{
			{"--- a/configs/qwen3_lora.yaml", 300},
			{"+++ b/configs/qwen3_lora.yaml", 200},
			{"@@ -14,2 +14,7 @@", 200},
			{" model:", 150},
			{"+  moe:", 200},
			{"+    enabled: True", 150},
			{"+    num_experts: 8", 150},
			{"+    top_k: 2", 150},
			{"+    capacity_factor: 1.25", 150},
			{"+    load_balance_loss_weight: 0.01", 150},
			{"", 400},
			{"--- a/models/moe_layer.py", 300},
			{"+++ b/models/moe_layer.py", 200},
			{"@@ -0,0 +1,20 @@", 200},
			{"+import torch", 150},
			{"+import torch.nn as nn", 150},
			{"+", 100},
			{"+class TopKRouter(nn.Module):", 200},
			{"+    def __init__(self, hidden_dim, num_experts, top_k=2):", 200},
			{"+        super().__init__()", 150},
			{"+        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)", 150},
			{"+        self.top_k = top_k", 150},
			{"+", 100},
			{"+    def forward(self, x):", 200},
			{"+        logits = self.gate(x)", 150},
			{"+        scores, indices = torch.topk(logits, self.top_k, dim=-1)", 150},
			{"+        weights = torch.softmax(scores, dim=-1)", 150},
			{"+        return weights, indices", 150},
			{"", 300},
			{"2 files changed, 26 insertions(+)", 500},
		}
	}

	for _, d := range diffLines {
		if !e(Event{Kind: EventDiffLine, Message: d.msg, DelayMs: d.delayMs}) {
			return ctx.Err()
		}
	}

	if !e(Event{
		Kind:         EventFixApplied,
		IssueType:    "algo-feature",
		ActionSource: "algo-agent",
		FixSummary:   fmt.Sprintf("%s applied to %s %s", info.name, model, method),
		Message:      "config patched. please rerun experiment.",
		DelayMs:      1000,
	}) {
		return ctx.Err()
	}

	return nil
}

// RunSingleLanePerfFeature applies a performance feature and prepares for rerun.
func RunSingleLanePerfFeature(ctx context.Context, model, method, feature string, sink func(Event)) error {
	e := func(ev Event) bool { return emit(ctx, sink, withDefaultRunID(ev)) }

	if feature == "" {
		feature = "fa2"
	}

	featureNames := map[string]string{
		"fa2":                 "Flash Attention v2",
		"fused-adam":          "Fused Adam",
		"gradient-ckpt":       "Gradient Checkpointing",
		"bf16-mixed":          "BF16 Mixed Precision",
		"graph-mod":           "Graph Mode",
		"comm-overlap":        "Communication Overlap",
		"zero-offload":        "ZeRO Offload",
		"sequence-parallel":   "Sequence Parallel",
		"selective-recompute": "Selective Recompute",
	}
	displayName := featureNames[feature]
	if displayName == "" {
		displayName = feature
	}

	if !e(Event{Kind: EventMessage, ActionSource: "perf-agent", Message: fmt.Sprintf("evaluating %s for %s %s...", displayName, model, method), DelayMs: 800}) {
		return ctx.Err()
	}

	if !e(Event{Kind: EventMessage, ActionSource: "perf-agent", Message: fmt.Sprintf("%s selected. applying kernel-level optimization.", displayName), DelayMs: 800}) {
		return ctx.Err()
	}

	if !e(Event{Kind: EventMessage, ActionSource: "perf-agent", Message: "patching training config and kernel registry...", DelayMs: 600}) {
		return ctx.Err()
	}

	// Feature-specific diffs
	var diffLines []struct {
		msg     string
		delayMs int
	}

	switch feature {
	case "fa2":
		diffLines = []struct {
			msg     string
			delayMs int
		}{
			{"--- a/configs/qwen3_lora.yaml", 300},
			{"+++ b/configs/qwen3_lora.yaml", 200},
			{"@@ -12,3 +12,5 @@", 200},
			{" attention:", 150},
			{"-  backend: \"default\"", 200},
			{"+  backend: \"flash_attention_2\"", 200},
			{"+  flash_attn_version: 2", 150},
			{"+  use_sliding_window: True", 150},
			{"", 400},
			{"--- a/kernels/attn_registry.py", 300},
			{"+++ b/kernels/attn_registry.py", 200},
			{"@@ -8,2 +8,6 @@", 200},
			{" ATTN_REGISTRY = {", 150},
			{"+    \"flash_attention_2\": FlashAttnV2Kernel(", 200},
			{"+        block_size=128,", 150},
			{"+        causal=True,", 150},
			{"+        softmax_scale=None,  # auto", 150},
			{"+    ),", 100},
			{"", 300},
			{"2 files changed, 7 insertions(+), 1 deletion(-)", 500},
		}
	case "fused-adam":
		diffLines = []struct {
			msg     string
			delayMs int
		}{
			{"--- a/configs/qwen3_lora.yaml", 300},
			{"+++ b/configs/qwen3_lora.yaml", 200},
			{"@@ -22,3 +22,4 @@", 200},
			{" optimizer:", 150},
			{"-  type: \"adam\"", 200},
			{"+  type: \"fused_adam\"", 200},
			{"+  fused_kernel: True  # Ascend 910B fused adam", 150},
			{"", 300},
			{"1 file changed, 2 insertions(+), 1 deletion(-)", 500},
		}
	case "gradient-ckpt":
		diffLines = []struct {
			msg     string
			delayMs int
		}{
			{"--- a/configs/qwen3_lora.yaml", 300},
			{"+++ b/configs/qwen3_lora.yaml", 200},
			{"@@ -30,2 +30,4 @@", 200},
			{" memory:", 150},
			{"+  gradient_checkpointing: True", 200},
			{"+  checkpoint_every_n_layers: 4", 150},
			{"", 300},
			{"1 file changed, 2 insertions(+)", 500},
		}
	case "bf16-mixed":
		diffLines = []struct {
			msg     string
			delayMs int
		}{
			{"--- a/configs/qwen3_lora.yaml", 300},
			{"+++ b/configs/qwen3_lora.yaml", 200},
			{"@@ -6,3 +6,5 @@", 200},
			{" precision:", 150},
			{"-  dtype: \"fp16\"", 200},
			{"+  dtype: \"bf16\"", 200},
			{"+  mixed_precision: True", 150},
			{"+  grad_dtype: \"fp32\"  # keep grads in fp32", 150},
			{"", 300},
			{"1 file changed, 3 insertions(+), 1 deletion(-)", 500},
		}
	case "graph-mod":
		diffLines = []struct {
			msg     string
			delayMs int
		}{
			{"--- a/configs/qwen3_lora.yaml", 300},
			{"+++ b/configs/qwen3_lora.yaml", 200},
			{"@@ -2,2 +2,6 @@", 200},
			{" runtime:", 150},
			{"+  mode: \"graph\"", 200},
			{"+  graph_compiler: \"ge\"  # Graph Engine for Ascend NPU", 150},
			{"+  compile_level: \"O2\"", 150},
			{"+  static_shape: True", 150},
			{"", 400},
			{"--- a/models/graph_mode_wrapper.py", 300},
			{"+++ b/models/graph_mode_wrapper.py", 200},
			{"@@ -0,0 +1,12 @@", 200},
			{"+import mindspore as ms", 150},
			{"+from mindspore import context", 150},
			{"+", 100},
			{"+def enable_graph_mode(compile_level=\"O2\"):", 200},
			{"+    context.set_context(mode=ms.GRAPH_MODE)", 200},
			{"+    context.set_context(jit_level=compile_level)", 150},
			{"+    context.set_context(device_target=\"Ascend\")", 150},
			{"+    context.set_context(enable_graph_kernel=True)", 150},
			{"", 300},
			{"2 files changed, 12 insertions(+)", 500},
		}
	case "comm-overlap":
		diffLines = []struct {
			msg     string
			delayMs int
		}{
			{"--- a/configs/qwen3_lora.yaml", 300},
			{"+++ b/configs/qwen3_lora.yaml", 200},
			{"@@ -38,2 +38,6 @@", 200},
			{" distributed:", 150},
			{"+  comm_overlap: True", 200},
			{"+  overlap_grad_reduce: True", 150},
			{"+  bucket_size_mb: 25", 150},
			{"+  allreduce_fusion_threshold: 2048", 150},
			{"", 400},
			{"--- a/models/distributed_wrapper.py", 300},
			{"+++ b/models/distributed_wrapper.py", 200},
			{"@@ -0,0 +1,14 @@", 200},
			{"+import torch", 150},
			{"+import torch.distributed as dist", 150},
			{"+", 100},
			{"+class OverlapAllReduceHook:", 200},
			{"+    \"\"\"Overlap allreduce with backward computation.\"\"\"", 200},
			{"+    def __init__(self, bucket_size_mb=25):", 150},
			{"+        self.bucket_bytes = bucket_size_mb * 1024 * 1024", 150},
			{"+", 100},
			{"+    def register(self, model):", 200},
			{"+        for param in model.parameters():", 150},
			{"+            if param.requires_grad:", 150},
			{"+                param.register_post_accumulate_grad_hook(", 150},
			{"+                    lambda p: dist.all_reduce(p.grad, async_op=True))", 150},
			{"", 300},
			{"2 files changed, 18 insertions(+)", 500},
		}
	case "zero-offload":
		diffLines = []struct {
			msg     string
			delayMs int
		}{
			{"--- a/configs/qwen3_lora.yaml", 300},
			{"+++ b/configs/qwen3_lora.yaml", 200},
			{"@@ -22,4 +22,8 @@", 200},
			{" optimizer:", 150},
			{"   type: \"adam\"", 150},
			{"+  zero_stage: 2", 200},
			{"+  offload_device: \"cpu\"", 150},
			{"+  offload_pin_memory: True", 150},
			{"+  overlap_comm: True", 150},
			{"", 400},
			{"--- a/models/zero_config.py", 300},
			{"+++ b/models/zero_config.py", 200},
			{"@@ -0,0 +1,14 @@", 200},
			{"+from deepspeed.runtime.zero import ZeroConfig", 150},
			{"+", 100},
			{"+def build_zero_offload_config(stage=2):", 200},
			{"+    return {", 200},
			{"+        \"zero_optimization\": {", 150},
			{"+            \"stage\": stage,", 150},
			{"+            \"offload_optimizer\": {", 150},
			{"+                \"device\": \"cpu\",", 150},
			{"+                \"pin_memory\": True,", 150},
			{"+            },", 100},
			{"+            \"overlap_comm\": True,", 150},
			{"+            \"contiguous_gradients\": True,", 150},
			{"+        }", 100},
			{"+    }", 100},
			{"", 300},
			{"2 files changed, 18 insertions(+)", 500},
		}
	case "sequence-parallel":
		diffLines = []struct {
			msg     string
			delayMs int
		}{
			{"--- a/configs/qwen3_lora.yaml", 300},
			{"+++ b/configs/qwen3_lora.yaml", 200},
			{"@@ -38,3 +38,7 @@", 200},
			{" parallelism:", 150},
			{"   tensor_parallel: 4", 150},
			{"+  sequence_parallel: True", 200},
			{"+  sp_size: 4  # split sequence across TP group", 150},
			{"+  sp_comm_overlap: True", 150},
			{"+  scatter_gather_tensors: True", 150},
			{"", 400},
			{"--- a/models/sequence_parallel.py", 300},
			{"+++ b/models/sequence_parallel.py", 200},
			{"@@ -0,0 +1,16 @@", 200},
			{"+import torch", 150},
			{"+import torch.distributed as dist", 150},
			{"+", 100},
			{"+def scatter_along_seq(x, sp_group):", 200},
			{"+    \"\"\"Scatter input along sequence dim across SP group.\"\"\"", 200},
			{"+    world_size = dist.get_world_size(sp_group)", 150},
			{"+    seq_len = x.shape[1]", 150},
			{"+    assert seq_len % world_size == 0", 150},
			{"+    chunk_size = seq_len // world_size", 150},
			{"+    rank = dist.get_rank(sp_group)", 150},
			{"+    return x[:, rank * chunk_size : (rank + 1) * chunk_size, :]", 200},
			{"+", 100},
			{"+def gather_along_seq(x, sp_group):", 200},
			{"+    world_size = dist.get_world_size(sp_group)", 150},
			{"+    gathered = [torch.empty_like(x) for _ in range(world_size)]", 150},
			{"+    dist.all_gather(gathered, x, group=sp_group)", 150},
			{"+    return torch.cat(gathered, dim=1)", 150},
			{"", 300},
			{"2 files changed, 21 insertions(+)", 500},
		}
	case "selective-recompute":
		diffLines = []struct {
			msg     string
			delayMs int
		}{
			{"--- a/configs/qwen3_lora.yaml", 300},
			{"+++ b/configs/qwen3_lora.yaml", 200},
			{"@@ -30,2 +30,5 @@", 200},
			{" memory:", 150},
			{"+  selective_recompute: True", 200},
			{"+  recompute_modules: [\"attention\"]  # only recompute attention layers", 150},
			{"+  recompute_granularity: \"selective\"", 150},
			{"", 400},
			{"--- a/models/recompute_wrapper.py", 300},
			{"+++ b/models/recompute_wrapper.py", 200},
			{"@@ -0,0 +1,16 @@", 200},
			{"+import torch", 150},
			{"+from torch.utils.checkpoint import checkpoint", 150},
			{"+", 100},
			{"+def apply_selective_recompute(model, modules=(\"attention\",)):", 200},
			{"+    \"\"\"Apply activation checkpointing only to attention layers.\"\"\"", 200},
			{"+    for name, module in model.named_modules():", 150},
			{"+        if any(m in name for m in modules):", 150},
			{"+            original_forward = module.forward", 150},
			{"+            def _recompute_forward(*args, _fn=original_forward, **kwargs):", 200},
			{"+                return checkpoint(_fn, *args, use_reentrant=False, **kwargs)", 200},
			{"+            module.forward = _recompute_forward", 150},
			{"", 300},
			{"2 files changed, 14 insertions(+)", 500},
		}
	}

	for _, d := range diffLines {
		if !e(Event{Kind: EventDiffLine, Message: d.msg, DelayMs: d.delayMs}) {
			return ctx.Err()
		}
	}

	if !e(Event{
		Kind:         EventFixApplied,
		IssueType:    "perf-feature",
		ActionSource: "perf-agent",
		FixSummary:   fmt.Sprintf("%s applied to %s %s", displayName, model, method),
		Message:      "config patched. please rerun experiment.",
		DelayMs:      1000,
	}) {
		return ctx.Err()
	}

	return nil
}

// ── Story 4: Single-lane performance recovery ────────────────

// AnalyzeSingleLanePerf diagnoses a performance bottleneck for the single-lane flow.
func AnalyzeSingleLanePerf(ctx context.Context, model, method string, sink func(Event)) error {
	e := func(ev Event) bool { return emit(ctx, sink, withDefaultRunID(ev)) }

	if !e(Event{Kind: EventAnalysisStarted, ActionSource: "perf-agent", Message: "analyzing performance...", IssueType: "performance", DelayMs: 500}) {
		return ctx.Err()
	}

	if !e(Event{Kind: EventMessage, ActionSource: "perf-agent", Message: "profiling whole models and ops...", DelayMs: 800}) {
		return ctx.Err()
	}

	if !e(Event{Kind: EventMessage, ActionSource: "perf-agent", Message: "root cause found, adam optimizer spent 400ms which is abnormal.", DelayMs: 800}) {
		return ctx.Err()
	}

	if !e(Event{
		Kind:         EventActionSuggested,
		IssueType:    "performance",
		IssueID:      "perf-adam-latency",
		ActionID:     "fix-fused-adam",
		ActionKind:   "apply_patch",
		ActionLabel:  "apply fix",
		ActionSource: "perf-agent",
		Message:      "suggest to replace with fused_adam kernel which will boost around 10%.",
		DelayMs:      500,
	}) {
		return ctx.Err()
	}

	diffText := `--- a/configs/qwen3_lora.yaml
+++ b/configs/qwen3_lora.yaml
@@ -22,3 +22,4 @@
 optimizer:
-  type: "adam"
+  type: "fused_adam"
+  fused_kernel: True  # custom fused adam kernel for Ascend 910B`

	if !e(Event{
		Kind:        EventAnalysisReady,
		IssueType:   "performance",
		IssueID:     "perf-adam-latency",
		IssueTitle:  "Adam optimizer latency bottleneck (400ms per step)",
		IssueDetail: "Standard adam optimizer takes 400ms per step. Fused adam kernel reduces this to ~250ms, boosting throughput by ~10%.",
		Confidence:  "high",
		FixSummary:  "Replace adam with fused_adam kernel",
		DiffText:    diffText,
		ActionID:    "fix-fused-adam",
		ActionKind:  "apply_patch",
		ActionLabel: "apply fix",
		Message:     "analysis complete. Ready to apply fix.",
		DelayMs:     400,
	}) {
		return ctx.Err()
	}

	return nil
}

// ApplySingleLanePerfFix applies the performance fix (no rerun — user clicks rerun).
func ApplySingleLanePerfFix(ctx context.Context, model, method string, sink func(Event)) error {
	e := func(ev Event) bool { return emit(ctx, sink, withDefaultRunID(ev)) }

	if !e(Event{
		Kind:         EventActionApplied,
		IssueType:    "performance",
		ActionID:     "fix-fused-adam",
		ActionKind:   "apply_patch",
		ActionSource: "op-agent",
		Message:      "replacing normal adam with fused kernel...",
		DelayMs:      2000,
	}) {
		return ctx.Err()
	}

	if !e(Event{
		Kind:         EventFixApplied,
		IssueType:    "performance",
		ActionSource: "op-agent",
		FixSummary:   "Replaced adam with fused_adam kernel",
		Message:      "kernel replaced. please rerun experiment.",
		DelayMs:      1500,
	}) {
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

	if !e(Event{Kind: EventMessage, ActionSource: "setup-helper", Message: fmt.Sprintf("preparing %s %s compare training. running preflight checks...", model, method), DelayMs: 400}) {
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

	if !e(Event{Kind: EventReadyToStart, ActionSource: "observer", Message: "all preflight checks passed. ready to launch compare training.", DelayMs: 300}) {
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

	if !e(Event{Kind: EventAnalysisStarted, Message: "analyzing NPU runtime failure...", DelayMs: 500}) {
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
		Message:      "suggested action: switch to SDPA fallback and relaunch.",
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
		Message:     "root cause identified. Runtime fix prepared.",
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
		Message:    "runtime fix applied. Relaunching NPU training...",
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
	if !e(Event{Kind: EventEvalStarted, Message: "evaluating both models on MMLU benchmark...", DelayMs: 800}) {
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
		Message:      "evaluation complete. Significant accuracy drift detected.",
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

	if !e(Event{Kind: EventAnalysisStarted, Message: "analyzing accuracy drift on Ascend eval path...", DelayMs: 500}) {
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
		Message:      "suggested action: patch mask dtype and softmax precision before rerun.",
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
		Message:     "root cause identified. Accuracy fix prepared.",
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
		Message:    "accuracy fix applied. Starting NPU rerun...",
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
		ActionSource: "observer",
		Message:      "fix verified. NPU accuracy recovered from 54.8% to 69.8%. Drift: -16.6 pts -> -1.6 pts.",
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

	if !e(Event{Kind: EventAnalysisStarted, Message: "analyzing throughput bottleneck on target...", IssueType: "performance", DelayMs: 400}) {
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
		Message:      "suggested action: increase dataloader workers and disable checkpointing for this target.",
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
		Message:      "performance bottleneck identified. Throughput tuning is ready.",
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

	if !e(Event{Kind: EventFixApplied, Message: "performance tuning applied. Restarting run...", IssueType: "performance", FixSummary: "Increase dataloader workers, disable checkpointing", ActionID: "perf-workers-tuning", ActionKind: "change_config", ActionLabel: "Apply throughput tuning", ActionSource: "perf-agent", DelayMs: 300}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventRerunStarted, Message: "rerunning after performance tuning...", RunLabel: "perf-rerun", DelayMs: 300}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventMetricUpdate, Step: 40, TotalSteps: 300, Loss: 2.102, LR: 1.9e-4, Throughput: 690, RunLabel: "perf-rerun", DelayMs: 200}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventMetricUpdate, Step: 120, TotalSteps: 300, Loss: 1.312, LR: 1.1e-4, Throughput: 712, RunLabel: "perf-rerun", DelayMs: 200}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventTrainCompleted, Message: "performance rerun completed. Throughput improved to ~710 tok/s.", RunLabel: "perf-rerun", DelayMs: 300}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventVerificationPassed, ActionSource: "observer", Message: "performance verified. Accuracy remains stable while throughput improved.", IssueType: "performance", DelayMs: 300}) {
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
	if !e(Event{Kind: EventFixApplied, Message: fmt.Sprintf("applying new trick: %s", trick), ActionID: "trick-" + trick, ActionKind: "apply_patch", ActionLabel: "Apply trick " + strings.ToUpper(trick), ActionSource: "trick-flow", FixSummary: "Patch train config with new trick", DelayMs: 300}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventRerunStarted, Message: fmt.Sprintf("rerunning with trick %s enabled...", trick), RunLabel: "trick-rerun", DelayMs: 300}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventLogLine, Message: fmt.Sprintf("[trick] %s enabled in train config", trick), DelayMs: 300}) {
		return ctx.Err()
	}
	if !e(Event{Kind: EventDriftDetected, Message: fmt.Sprintf("after enabling %s, validation accuracy regressed. Accuracy analysis recommended.", trick), IssueType: "accuracy", DelayMs: 300}) {
		return ctx.Err()
	}
	return nil
}
