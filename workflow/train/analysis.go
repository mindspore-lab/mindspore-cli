package train

import (
	"context"
	"fmt"
)

// AnalyzeFailure diagnoses a runtime training failure.
func AnalyzeFailure(ctx context.Context, model, method string, sink func(Event)) error {
	e := func(ev Event) bool { return emit(ctx, sink, withDefaultRunID(ev)) }

	if !e(Event{
		Kind:    EventAnalysisStarted,
		Message: "Analyzing training failure...",
		DelayMs: 500,
	}) {
		return ctx.Err()
	}

	if !e(Event{
		Kind:    EventMessage,
		Message: "Scanning crash logs and operator registry...",
		DelayMs: 800,
	}) {
		return ctx.Err()
	}

	if !e(Event{
		Kind:    EventMessage,
		Message: "Checking torch operator compatibility for Ascend 910B...",
		DelayMs: 600,
	}) {
		return ctx.Err()
	}

	if !e(Event{
		Kind:         EventActionSuggested,
		IssueType:    "failure",
		IssueID:      "failure-dsa-op",
		ActionID:     "fix-dsa-op",
		ActionKind:   "apply_patch",
		ActionLabel:  "apply fix",
		ActionSource: "op-agent",
		Message:      "Root cause: DSA operator (torch.ops.npu.dsa) is not implemented in torch 2.7 for Ascend backend. Need to implement DSA op and compile custom torch-npu package.",
		DelayMs:      500,
	}) {
		return ctx.Err()
	}

	diffText := fmt.Sprintf(`--- a/torch_npu/csrc/aten/ops/dsa_kernel.cpp
+++ b/torch_npu/csrc/aten/ops/dsa_kernel.cpp
@@ -0,0 +1,42 @@
+#include <ATen/native/npu/DsaKernel.h>
+#include <torch_npu/csrc/framework/OpCommand.h>
+
+namespace at_npu {
+namespace native {
+
+at::Tensor dsa_forward_npu(
+    const at::Tensor& query,
+    const at::Tensor& key,
+    const at::Tensor& value,
+    double scale) {
+  // DSA (Dynamic Sparse Attention) NPU kernel
+  auto output = at::empty_like(query);
+  OpCommand cmd;
+  cmd.Name("DSA")
+     .Input(query)
+     .Input(key)
+     .Input(value)
+     .Attr("scale", static_cast<float>(scale))
+     .Output(output)
+     .Run();
+  return output;
+}
+
+} // namespace native
+} // namespace at_npu

--- a/setup.py
+++ b/setup.py
@@ -112,6 +112,7 @@ EXT_SOURCES = [
     "torch_npu/csrc/aten/ops/flash_attention_kernel.cpp",
+    "torch_npu/csrc/aten/ops/dsa_kernel.cpp",
 ]`)

	if !e(Event{
		Kind:       EventAnalysisReady,
		IssueType:  "failure",
		IssueID:    "failure-dsa-op",
		IssueTitle: "DSA operator not implemented in torch 2.7",
		IssueDetail: "The training script calls torch.ops.npu.dsa() which requires the Dynamic Sparse Attention kernel. " +
			"This operator is not available in the current torch-npu 2.7 package. " +
			"A custom kernel implementation and torch-npu recompilation is required.",
		Confidence:   "high",
		FixSummary:   "Implement DSA kernel in torch_npu and recompile",
		DiffText:     diffText,
		ActionID:     "fix-dsa-op",
		ActionKind:   "apply_patch",
		ActionLabel:  "apply fix",
		ActionSource: "op-agent",
		Message:      "Analysis complete. Ready to apply fix.",
		DelayMs:      400,
	}) {
		return ctx.Err()
	}

	return nil
}

// ApplyFailureFix applies the fix for a runtime failure and reruns training.
func ApplyFailureFix(ctx context.Context, model, method string, sink func(Event)) error {
	e := func(ev Event) bool { return emit(ctx, sink, withDefaultRunID(ev)) }

	if !e(Event{
		Kind:       EventActionApplied,
		IssueType:  "failure",
		ActionID:   "fix-dsa-op",
		ActionKind: "apply_patch",
		Message:    "op-agent: implementing DSA operator and compiling custom torch-npu...",
		DelayMs:    2000,
	}) {
		return ctx.Err()
	}

	if !e(Event{
		Kind:       EventFixApplied,
		IssueType:  "failure",
		FixSummary: "DSA operator implemented and torch-npu recompiled",
		Message:    "op-agent: DSA operator finished. New torch wheel is ready. Please rerun experiment.",
		DelayMs:    1500,
	}) {
		return ctx.Err()
	}

	return nil
}
