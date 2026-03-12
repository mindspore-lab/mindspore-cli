package target

import (
	"context"

	"github.com/vigo999/ms-cli/internal/train"
	"github.com/vigo999/ms-cli/runtime/probes"
)

// OSProbe checks remote OS.
// Phase 1: returns demo-backed results.
type OSProbe struct{}

func (p *OSProbe) Run(_ context.Context, target train.TrainTarget) ([]probes.Result, error) {
	// TODO: real implementation would run remote commands via SSH
	return []probes.Result{
		{
			Scope:    probes.ScopeTarget,
			Name:     "os",
			Status:   probes.StatusPass,
			Summary:  "ubuntu 24.04",
			Critical: true,
			Details: map[string]any{
				"host": target.Name,
			},
		},
	}, nil
}
