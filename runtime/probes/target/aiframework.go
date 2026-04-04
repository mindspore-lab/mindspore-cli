package target

import (
	"context"

	"github.com/vigo999/mindspore-cli/internal/train"
	"github.com/vigo999/mindspore-cli/runtime/probes"
)

// AIFrameworkProbe checks remote AI framework and driver stack.
// Phase 1: returns demo-backed results.
type AIFrameworkProbe struct{}

func (p *AIFrameworkProbe) Run(_ context.Context, target train.TrainTarget) ([]probes.Result, error) {
	libsStatus := probes.StatusPass
	libsSummary := "torch 2.7 | mindspore 2.8 | transformers v5.0.1 | diffusers v0.36"
	if missing, _ := target.Config["demo_libs_missing"].(bool); missing {
		libsStatus = probes.StatusFail
		libsSummary = "torch 2.7 | mindspore 2.8 | transformers (missing) | diffusers v0.36"
	}

	return []probes.Result{
		{
			Scope:    probes.ScopeTarget,
			Name:     "libs",
			Status:   libsStatus,
			Summary:  libsSummary,
			Critical: true,
			Details: map[string]any{
				"host": target.Name,
			},
		},
		{
			Scope:    probes.ScopeTarget,
			Name:     "device driver",
			Status:   probes.StatusPass,
			Summary:  "cuda 13.1 | cann 8.5",
			Critical: true,
			Details: map[string]any{
				"host": target.Name,
			},
		},
	}, nil
}
