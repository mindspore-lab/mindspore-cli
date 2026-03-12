package target

import (
	"context"

	"github.com/vigo999/ms-cli/internal/train"
	"github.com/vigo999/ms-cli/runtime/probes"
)

// WorkdirProbe checks remote working directory existence and writability.
// Phase 1: returns demo-backed results.
type WorkdirProbe struct{}

func (p *WorkdirProbe) Run(_ context.Context, target train.TrainTarget) ([]probes.Result, error) {
	// TODO: real implementation would test remote path write access
	return []probes.Result{
		{
			Scope:    probes.ScopeTarget,
			Name:     "working dir",
			Status:   probes.StatusPass,
			Summary:  "~/work/qwen3_experiment",
			Critical: true,
			Details: map[string]any{
				"host": target.Name,
			},
		},
	}, nil
}
